"""Public worker_cls adapter; delegates every model and sampling call unchanged."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from vllm_neuron.vllm.worker.neuron_model_runner import AsyncNeuronModelRunnerOutput
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

from profiling.runners.neuron import vllm_ops_control as control


class OpsNeuronWorker(NeuronWorker):
    def __init__(self, *args, **kwargs):
        site = Path(importlib.util.find_spec("vllm").origin).parent.parent
        source_hashes = control.verify_runtime_sources(site)
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        profiler = config.profiler_config
        if (
            not self.use_async_scheduling
            or config.parallel_config.tensor_parallel_size != 4
            or config.parallel_config.pipeline_parallel_size != 1
            or profiler.delay_iterations != 0
            or profiler.max_iterations != 0
            or config.cache_config.enable_prefix_caching
        ):
            raise ValueError("capture worker requires the pinned stock async TP4 configuration")
        root = Path(config.additional_config["neuron_profiler"]["output_dir"]).parent
        plan = json.loads((root / "ops-plan.json").read_text())
        if plan["shape"]["phase"] != "prefill" or plan["request"]["forwards"] != 1:
            raise ValueError("geometry capture worker only supports one selected prefill")
        if config.additional_config["neuron_profiler"]["neuron_cores"] != plan["request"]["ranks"]:
            raise ValueError("capture plan and native profiler rank selection differ")
        self._ops_control = control.GeometryCapture(
            prompts=plan["case"]["prompts"],
            bucket=plan["shape"]["token_bucket"],
            rank=self.rank,
            local_rank=self.local_rank,
            selected_ranks=plan["request"]["ranks"],
            decode_buckets=config.additional_config["neuron_config"]["num_seqs_buckets"],
            receipt_path=root / "capture-control" / f"rank-{self.rank}.json",
            source_hashes=source_hashes,
            hook_source_hashes={
                Path(path).name: control.file_hash(Path(path))
                for path in (__file__, control.__file__)
            },
        )
        self._ops_previous_output = None

    @staticmethod
    def _ops_check_output(output):
        if not isinstance(output, AsyncNeuronModelRunnerOutput) or output.is_all_partial_prefill():
            raise ValueError("capture requires a complete stock asynchronous sampled output")

    @classmethod
    def _ops_drain(cls, output):
        cls._ops_check_output(output)
        # The pinned implementation holds an object-local lock, materializes once,
        # and skips state updates if another thread already converted the tokens.
        output.get_output()

    def _ops_native_profile(self, is_start):
        super().profile(is_start)
        if self.local_rank in self._ops_control.selected_ranks:
            profiler = self._profiler
            if (
                profiler is None
                or profiler._running is not is_start
                or profiler._profiling is not is_start
            ):
                raise RuntimeError("stock Neuron profiler transition did not complete")

    def profile(self, is_start=True, profile_prefix=None):
        try:
            if is_start:
                self._ops_control.arm()
            else:
                self._ops_control.finish()
        except Exception as error:
            self._ops_control.abort(error, stop=lambda: self._ops_native_profile(False))
            raise

    def execute_model(self, scheduler_output):
        try:
            self._ops_control.before_execute(
                scheduler_output,
                self._ops_previous_output,
                start=lambda: self._ops_native_profile(True),
                drain=self._ops_drain,
            )
            result = super().execute_model(scheduler_output)
            if self._ops_control.pending is not None and result is not None:
                raise ValueError("stock execute_model did not defer sampling as expected")
            return result
        except Exception as error:
            self._ops_control.abort(error, stop=lambda: self._ops_native_profile(False))
            raise

    def sample_tokens(self, grammar_output):
        try:
            result = super().sample_tokens(grammar_output)
            self._ops_control.after_sample(
                result,
                check_output=self._ops_check_output,
                drain=self._ops_drain,
                stop=lambda: self._ops_native_profile(False),
            )
            self._ops_previous_output = result
            return result
        except Exception as error:
            self._ops_control.abort(error, stop=lambda: self._ops_native_profile(False))
            raise
