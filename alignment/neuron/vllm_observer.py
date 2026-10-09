"""Observational public stock scheduler/worker subclasses; no model overrides.

Pinned integration: vllm-neuron 0.24.0.1.1.0, vLLM 0.24.0. Observations use
SchedulerOutput CPU metadata and tensor shapes only. No device synchronization.
"""

import importlib.metadata
import json
import os
import time

from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

from .vllm_cleanup import install_cleanup_adapter
from .vllm_records import (
    STOCK_VERSIONS,
    RequestTimings,
    iteration_metric,
    scheduler_snapshot,
    token_hash,
)


def _emit(tag: str, row: dict) -> None:
    print(tag + " " + json.dumps(row, separators=(",", ":")), flush=True)


def block_table_shapes(metadata: dict) -> list[tuple[int, ...]]:
    """Read tensor shapes only, including the stock decode-mask cache entry.

    llama3/model.py caches `_cached_decode_mask` in the layer metadata map.
    It is a tensor, while the named layers remain dictionaries. Never inspect
    that tensor's contents or treat it as another layer.
    """
    if not isinstance(metadata, dict):
        raise RuntimeError("unknown stock attention metadata schema")
    if "block_table_tensor" in metadata:
        layers = [metadata]
    else:
        layers = []
        for key, value in metadata.items():
            if key == "_cached_decode_mask":
                if not hasattr(value, "shape"):
                    raise RuntimeError("stock decode-mask cache is not a tensor")
            elif isinstance(value, dict) and "block_table_tensor" in value:
                layers.append(value)
            else:
                raise RuntimeError("unknown stock layer attention metadata")
    if not layers:
        raise RuntimeError("stock attention metadata has no block tables")
    return sorted({tuple(m["block_table_tensor"].shape) for m in layers})


class SchedulerObservation:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.log_stats:
            raise ValueError("stock alignment requires native EngineCore log_stats events")
        cleanup = install_cleanup_adapter()
        if cleanup is not None:
            _emit("VllmNeuronCleanup", cleanup)
        self._alignment_request_timings = RequestTimings()

    def schedule(self, *args, **kwargs):
        start = time.monotonic_ns()
        output = super().schedule(*args, **kwargs)
        snapshot = scheduler_snapshot(output)
        if snapshot["requests"]:
            for req in output.scheduled_new_reqs:
                _emit(
                    "VllmNeuronPrompt",
                    {
                        "request_id": req.req_id,
                        "prompt_tokens": len(req.prompt_token_ids),
                        "prompt_sha256": token_hash(req.prompt_token_ids),
                    },
                )
            index = getattr(self, "_alignment_iteration", 0)
            self._alignment_iteration = index + 1
            prompts = {
                r["request_id"]: self.requests[r["request_id"]].num_prompt_tokens
                for r in snapshot["requests"]
            }
            metric = iteration_metric(
                snapshot,
                prompt_lengths=prompts,
                iteration=index,
                start_ns=start,
                stop_ns=time.monotonic_ns(),
            )
            pending = getattr(self, "_alignment_pending", None)
            if pending is None:
                self._alignment_pending = pending = {}
            pending[id(output)] = metric
        return output

    def update_from_output(self, scheduler_output, model_output):
        output = super().update_from_output(scheduler_output, model_output)
        for timing in self._alignment_request_timings.observe(output):
            _emit("VibeSimAlignmentRequestTiming", timing)
        metric = getattr(self, "_alignment_pending", {}).pop(id(scheduler_output), None)
        if metric is not None:
            stop = time.monotonic_ns()
            metric["observed_end_monotonic_ns"] = stop
            metric["observed_elapsed_ms"] = (stop - metric["observed_start_monotonic_ns"]) / 1e6
            _emit("VibeSimAlignmentIteration", metric)
        return output


class ObservedAsyncScheduler(SchedulerObservation, NeuronAsyncScheduler):
    pass


class ObservedScheduler(SchedulerObservation, NeuronScheduler):
    def __new__(cls, *args, **kwargs):
        # Preserve the resolved stock async policy. The public platform replaces
        # only default scheduler class paths, so this observer performs that dispatch.
        config = kwargs.get("vllm_config", args[0] if args else None)
        if cls is ObservedScheduler and config.scheduler_config.async_scheduling:
            return ObservedAsyncScheduler(*args, **kwargs)
        return super().__new__(cls)


class ObservedWorker(NeuronWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        versions = {name: importlib.metadata.version(name) for name in STOCK_VERSIONS}
        if versions != STOCK_VERSIONS:
            raise RuntimeError("stock observation ABI requires the validated vLLM/Neuron versions")
        _emit(
            "VllmNeuronWorker",
            {
                "pid": os.getpid(),
                "rank": self.rank,
                "local_rank": self.local_rank,
                "versions": versions,
                "async_scheduling": self.vllm_config.scheduler_config.async_scheduling,
                "visible_devices": os.environ.get("NEURON_VISIBLE_DEVICES"),
            },
        )

    def execute_model(self, scheduler_output):
        snapshot = scheduler_snapshot(scheduler_output)
        start_epoch, start_mono = time.time_ns(), time.monotonic_ns()
        output = super().execute_model(scheduler_output)
        stop_epoch, stop_mono = time.time_ns(), time.monotonic_ns()
        if snapshot["requests"]:
            state = self.model_runner.execute_model_state
            if state is None or state.scheduler_output is not scheduler_output:
                raise RuntimeError(
                    "stock forward state does not belong to observed scheduler output"
                )
            index = getattr(self, "_alignment_forward", 0)
            self._alignment_forward = index + 1
            _emit(
                "VllmNeuronForward",
                {
                    **snapshot,
                    "worker_sequence": index,
                    "pid": os.getpid(),
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "start_epoch_ns": start_epoch,
                    "stop_epoch_ns": stop_epoch,
                    "start_monotonic_ns": start_mono,
                    "stop_monotonic_ns": stop_mono,
                    "input_ids_shape": list(state.input_ids.shape),
                    "block_table_shapes": block_table_shapes(state.attn_metadata),
                },
            )
        return output
