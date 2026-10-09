"""Loopback token completions around the public, compiled NxDI generation path.

The pinned NxDI sampler does not invoke Transformers streamers. A public
StoppingCriteria observes each appended token and emits a real SSE event.
Forward hooks only record public adapter inputs and host timing; they never
replace model code, sampling, weights, or the device-resident KV cache.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from alignment.neuron.normalize import iteration_metric


def validate_request(body: dict) -> tuple[list[int], int]:
    if not isinstance(body, dict):
        raise ValueError("completion request must be an object")
    prompt, count = body.get("prompt"), body.get("max_tokens")
    if (
        not isinstance(prompt, list)
        or not 1 <= len(prompt) <= 128
        or any(type(x) is not int or not 0 <= x < 128256 for x in prompt)
    ):
        raise ValueError("prompt must contain 1..128 Llama token IDs")
    if type(count) is not int or count < 1 or len(prompt) + count > 512:
        raise ValueError("prompt plus max_tokens must fit the 512-token KV capacity")
    if body.get("temperature", 0) != 0 or body.get("ignore_eos") is not True:
        raise ValueError("NxDI baseline requires temperature0 and ignore_eos")
    if body.get("stream") is not True:
        raise ValueError("NxDI alignment requires streaming token output")
    return prompt, count


class NativeRuntime:
    def __init__(self, args):
        from profiling.exec.neuron import LocalNeuronPool

        self.reservation = next(LocalNeuronPool(devices=[args.neuron_device]).acquire_chunks(k=1))
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(self.reservation.device.core_ids[0])
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
        from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
        from transformers import AutoTokenizer

        from tools.trainium2.validate_checkpoint import (
            build_config,
            checkpoint_manifest,
            package_versions,
        )

        torch.set_num_threads(4)
        config = build_config(args.model_dir)
        self.config = json.loads(config.to_json_string())
        if self.config != json.loads((args.compiled_dir / "neuron_config.json").read_text()):
            raise ValueError("compiled NxDI config differs from requested production path")
        self.model = NeuronLlamaForCausalLM(str(args.model_dir), config)
        self.model.load(str(args.compiled_dir))
        self.adapter = HuggingFaceGenerationAdapter(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.packages = package_versions()
        with (args.compiled_dir / "model.pt").open("rb") as compiled:
            model_digest = hashlib.file_digest(compiled, "sha256").hexdigest()
        (args.log_dir / "runtime-provenance.json").write_text(
            json.dumps(
                {
                    "producer_kind": "framework_capture",
                    "engine": "nxdi",
                    "packages": self.packages,
                    "config": self.config,
                    "compiled_model_sha256": model_digest,
                    "checkpoint": checkpoint_manifest(args.model_dir),
                    "hardware": {
                        "neuron_device": self.reservation.device.device_id,
                        "logical_core": self.reservation.device.core_ids[0],
                        "logical_nc_config": 2,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        self.lock = threading.Lock()
        self.busy = 0
        self.trace = None
        self.capture_rows: list[dict] = []
        self.capture_requests: list[dict] = []
        self.current = None
        self.next_iteration = 0
        self.log_dir = args.log_dir
        self.adapter.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        self.adapter.register_forward_hook(self._after_forward, with_kwargs=True)

    def _before_forward(self, module, args, kwargs):
        current = self.current
        ids = kwargs["input_ids"]
        position = kwargs["position_ids"]
        prefill = current["forwards"] == 0
        row = {
            "iteration_id": self.next_iteration,
            "request_id": current["request_id"],
            "phase": "prefill" if prefill else "decode",
            "q_tokens": ids.shape[1],
            "kv_len_before": 0 if prefill else int(position[0, -1]),
            "input_ids": ids.tolist(),
            "position_ids": position.tolist(),
            "start_realtime_ns": time.time_ns(),
            "start_monotonic_ns": time.monotonic_ns(),
        }
        self.next_iteration += 1
        current["row"] = row
        current["forwards"] += 1

    def _after_forward(self, module, args, kwargs, output):
        row = self.current["row"]
        row["stop_monotonic_ns"] = time.monotonic_ns()
        row["stop_realtime_ns"] = time.time_ns()
        self.current["rows"].append(row)
        print("VibeSimAlignmentIteration " + json.dumps(iteration_metric(row)), flush=True)
        if self.trace is not None:
            self.capture_rows.append(row)

    def start_trace(self):
        from nrtpy._nrtpy import SystemTraceSession

        with self.lock:
            if self.trace is not None:
                raise ValueError("native capture already active")
            self.capture_rows, self.capture_requests = [], []
            self.trace = SystemTraceSession(self.reservation.device.core_ids[0])
            self.trace.__enter__()
            self.trace.drain_events()

    def stop_trace(self):
        with self.lock:
            if self.trace is None:
                raise ValueError("native capture is not active")
            try:
                events = json.loads(self.trace.fetch_events_json())
                (self.log_dir / "system-trace.json").write_text(json.dumps(events))
                (self.log_dir / "forward-records.json").write_text(
                    json.dumps(
                        {
                            "producer_kind": "framework_capture",
                            "capture_kind": "neuron_system_trace",
                            "engine": "nxdi",
                            "iterations": self.capture_rows,
                            "requests": self.capture_requests,
                            "packages": self.packages,
                            "config": self.config,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                self.trace.__exit__(None, None, None)
                self.trace = None

    def generate(self, prompt, count, request_id, emit):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        accepted = time.monotonic_ns()
        self.busy += 1
        try:
            with self.lock:
                scheduled = time.monotonic_ns()
                captured = self.trace is not None
                self.current = {"request_id": request_id, "forwards": 0, "rows": []}
                emitted: list[int] = []
                token_times = []

                class EmitToken(StoppingCriteria):
                    def __call__(criterion, input_ids, scores, **kwargs):
                        token = int(input_ids[0, -1])
                        stamp = time.monotonic_ns()
                        emitted.append(token)
                        token_times.append(stamp)
                        self.current["rows"][-1]["greedy_token"] = token
                        emit(token)
                        return torch.zeros(
                            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
                        )

                ids = torch.tensor([prompt], dtype=torch.int64)
                with torch.inference_mode():
                    output = self.adapter.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=count,
                        min_new_tokens=count,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=None,
                        stopping_criteria=StoppingCriteriaList([EmitToken()]),
                    )
                if len(emitted) != count or output[0, len(prompt) :].tolist() != emitted:
                    raise RuntimeError("streamed token IDs differ from public NxDI output")
                stop = time.monotonic_ns()
                decode_ms = (token_times[-1] - token_times[0]) / 1e6
                timing = {
                    "schema_version": 2,
                    "request_id": request_id,
                    "engine_queue_wait_ms": (scheduled - accepted) / 1e6,
                    "engine_first_schedule_to_first_token_ms": (token_times[0] - scheduled) / 1e6,
                    "engine_core_ttft_ms": (token_times[0] - accepted) / 1e6,
                    "engine_core_decode_ms": decode_ms,
                    "engine_core_tpot_ms": decode_ms / (count - 1) if count > 1 else None,
                    "num_output_tokens": count,
                }
                print("VibeSimAlignmentRequestTiming " + json.dumps(timing), flush=True)
                request = {
                    "request_id": request_id,
                    "input_ids": [prompt],
                    "output_ids": output.tolist(),
                    "output_tokens": count,
                    "first_iteration": self.current["rows"][0]["iteration_id"],
                    "start_monotonic_ns": scheduled,
                    "stop_monotonic_ns": stop,
                }
                if captured:
                    self.capture_requests.append(request)
                print("NxdiOutputEvidence " + json.dumps(request), flush=True)
                return emitted
        finally:
            self.busy -= 1


class CompletionsHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    @property
    def runtime(self):
        return self.server.runtime

    def _json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in {"/health", "/healthz"}:
            self._json(200, {"status": "ok"})
        elif self.path == "/load":
            self._json(200, {"server_load": self.runtime.busy})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "llama3.1-8b-nxdi"}]})
        else:
            self._json(404, {"error": "unknown endpoint"})

    def do_POST(self):
        if self.path in {"/start_profile", "/stop_profile"}:
            try:
                (
                    self.runtime.start_trace
                    if self.path == "/start_profile"
                    else self.runtime.stop_trace
                )()
                self._json(200, {"status": "ok"})
            except (ValueError, RuntimeError) as error:
                self._json(409, {"error": str(error)})
            return
        if self.path == "/reset_prefix_cache":
            # This baseline resets private KV state per request and has no prefix cache.
            self._json(200, {"status": "ok", "prefix_caching": False})
            return
        if self.path != "/v1/completions":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            prompt, count = validate_request(body)
        except (ValueError, TypeError) as error:
            self._json(400, {"error": str(error)})
            return
        request_id = self.headers.get("X-Request-Id") or body.get("rid")
        if not isinstance(request_id, str) or not request_id:
            self._json(400, {"error": "request ID is required"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def event(data):
            self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
            self.wfile.flush()

        try:
            self.runtime.generate(
                prompt,
                count,
                request_id,
                lambda token: event({"choices": [{"text": "", "token_ids": [token]}]}),
            )
            event(
                {
                    "choices": [{"finish_reason": "length", "token_ids": [], "text": ""}],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": count,
                        "total_tokens": len(prompt) + count,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as error:
            print(
                "NxdiRequestError " + json.dumps({"request_id": request_id, "error": str(error)}),
                flush=True,
            )
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--neuron-device", type=int, default=0)
    args = parser.parse_args()
    for field in ("model_dir", "compiled_dir", "log_dir"):
        setattr(args, field, getattr(args, field).resolve())
    args.log_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CompletionsHandler)
    server.runtime = NativeRuntime(args)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        server.runtime.reservation.lock.close()


if __name__ == "__main__":
    main()
