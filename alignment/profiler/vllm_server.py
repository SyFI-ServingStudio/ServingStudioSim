"""Launch one instrumented-fork vLLM server for one TP replica.

A lean, single-GPU descendant of the reference harness's `launchers/vllm.py` +
`base.py`. Builds the server argv and a **clean subprocess env** (the vLLM/torch
runtime lives in a separate fork venv — this package's interpreter never imports
vLLM), records launch metadata, and offers ready/idle polling over the
OpenAI-compatible endpoints. The actual spawn is wrapped by `nsys_capture.py` and
orchestrated in `__main__.py`.

The fork (`alignment/profiler/vllm`, branch `moesim-profile`) adds the
`vllm_iteration(N): <phase>` NVTX scopes (gated by `VLLM_NVTX_SCOPES_FOR_PROFILING`)
and a versioned `VibeSimAlignmentIteration {json}` record containing the exact
model input shape consumed by the typed predictor adapter. Per-request
`VibeSimAlignmentRequestTiming {json}` records expose EngineCore TTFT and TPOT
separately from TraceLab's client-observed latency accounting.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import IdleWaitConfig, ProfileConfig, ServerConfig

VLLM_SOURCE_ROOT = Path(__file__).resolve().parent / "vllm"

HEALTH_ENDPOINTS = ("/health", "/v1/models")

# Structured fork-owned record. Do not parse the human `Iteration(...)` line:
# its wording is vLLM UI, while this JSON is the versioned analyzer contract.
_ALIGNMENT_ITERATION_RE = re.compile(r"VibeSimAlignmentIteration\s+(\{.*\})\s*$")
_ALIGNMENT_REQUEST_TIMING_RE = re.compile(r"VibeSimAlignmentRequestTiming\s+(\{.*\})\s*$")
_ALIGNMENT_EXPERT_LOAD_RE = re.compile(r"VibeSimAlignmentExpertLoad\s+(\{.*\})\s*$")
_COMPLETION_ENGINE_REQUEST_RE = re.compile(r"^cmpl-(.+)-0$")


def build_server_argv(fork_python: str, cfg: ServerConfig) -> list[str]:
    """The `python -m vllm.entrypoints.openai.api_server ...` argv."""
    argv = [
        fork_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        cfg.model_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tensor-parallel-size",
        str(cfg.tp_size),
        "--max-num-batched-tokens",
        str(cfg.chunk_size),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    if cfg.enforce_eager:
        argv.append("--enforce-eager")  # no CUDA graphs → clean per-kernel nsys records
    if cfg.use_ray_nsight:
        # Ray launches the model worker under nsys (runtime_env nsight,
        # cuda-graph-trace=node) — captures CUDA-graph replay per-op kernels.
        argv += ["--distributed-executor-backend", "ray", "--ray-workers-use-nsight"]
    if cfg.enable_server_load_tracking:
        argv.append("--enable-server-load-tracking")  # /load endpoint for idle checks
    if cfg.enable_iteration_metrics:
        argv.append("--enable-logging-iteration-details")
    if cfg.max_cudagraph_capture_size is not None:
        argv += ["--max-cudagraph-capture-size", str(cfg.max_cudagraph_capture_size)]
    if cfg.served_model_name:
        argv += ["--served-model-name", cfg.served_model_name]
    argv += cfg.extra_args
    return argv


def extract_expert_popularity(
    server_log: Path,
    out_jsonl: Path,
    out_json: Path,
    *,
    expert_parallel_size: int | None = None,
    experts_per_token: int | None = None,
) -> int:
    """Extract and aggregate rank-synchronized logical-expert token counts.

    The vLLM fork emits one record per model step only when EPLB balancedness
    logging is explicitly enabled.  Counts are already reduced across the EP
    group and mapped from physical replicas back to logical expert ids.
    """
    if expert_parallel_size is not None and expert_parallel_size <= 0:
        raise ValueError("expert_parallel_size must be positive")
    if experts_per_token is not None and experts_per_token <= 0:
        raise ValueError("experts_per_token must be positive")

    records: list[dict] = []
    expected_shape: tuple[int, int] | None = None
    expected_model: str | None = None
    aggregate_counts: list[list[int]] | None = None
    with Path(out_jsonl).open("w") as output_file:
        for line in Path(server_log).read_text(errors="replace").splitlines():
            match = _ALIGNMENT_EXPERT_LOAD_RE.search(line)
            if match is None:
                continue
            record = json.loads(match.group(1))
            required = {
                "schema_version",
                "model",
                "eplb_step",
                "logical_expert_counts",
            }
            missing = required - set(record)
            if missing:
                raise ValueError(f"alignment expert-load record missing {sorted(missing)}")
            if record["schema_version"] not in {1, 2}:
                raise ValueError(
                    f"unsupported alignment expert-load schema {record['schema_version']!r}"
                )
            if record["schema_version"] == 2:
                for field_name in ("expert_parallel_size", "experts_per_token"):
                    value = record.get(field_name)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError(
                            f"alignment expert-load {field_name} must be a positive integer"
                        )
                record_ep_size = record["expert_parallel_size"]
                record_top_k = record["experts_per_token"]
                if expert_parallel_size is None:
                    expert_parallel_size = record_ep_size
                elif record_ep_size != expert_parallel_size:
                    raise ValueError(
                        "expert-load expert_parallel_size changed from "
                        f"{expert_parallel_size} to {record_ep_size}"
                    )
                if experts_per_token is None:
                    experts_per_token = record_top_k
                elif record_top_k != experts_per_token:
                    raise ValueError(
                        "expert-load experts_per_token changed from "
                        f"{experts_per_token} to {record_top_k}"
                    )
            model = record["model"]
            if not isinstance(model, str) or not model:
                raise ValueError("alignment expert-load model must be a non-empty string")
            if expected_model is None:
                expected_model = model
            elif model != expected_model:
                raise ValueError(f"expert-load model changed from {expected_model!r} to {model!r}")
            eplb_step = record["eplb_step"]
            if isinstance(eplb_step, bool) or not isinstance(eplb_step, int) or eplb_step < 0:
                raise ValueError("alignment expert-load eplb_step must be a nonnegative integer")
            counts = record["logical_expert_counts"]
            if (
                not isinstance(counts, list)
                or not counts
                or not all(
                    isinstance(layer_counts, list) and layer_counts for layer_counts in counts
                )
            ):
                raise ValueError("logical_expert_counts must be a non-empty 2D array")
            shape = (len(counts), len(counts[0]))
            if any(len(layer_counts) != shape[1] for layer_counts in counts):
                raise ValueError("logical_expert_counts must be rectangular")
            if expected_shape is None:
                expected_shape = shape
                aggregate_counts = [[0] * shape[1] for _ in range(shape[0])]
            elif shape != expected_shape:
                raise ValueError(f"expert-load shape changed from {expected_shape} to {shape}")
            assert aggregate_counts is not None
            for layer_index, layer_counts in enumerate(counts):
                for expert_index, count in enumerate(layer_counts):
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError("expert-load counts must be nonnegative integers")
                    aggregate_counts[layer_index][expert_index] += count
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            records.append(record)

    if not records or expected_shape is None or aggregate_counts is None:
        raise ValueError("no VibeSimAlignmentExpertLoad records found in server log")
    if expert_parallel_size is None or experts_per_token is None:
        raise ValueError(
            "schema-v1 expert-load records require explicit expert_parallel_size "
            "and experts_per_token fallbacks"
        )
    if expected_shape[1] % expert_parallel_size != 0:
        raise ValueError(
            f"num_logical_experts {expected_shape[1]} must be divisible by "
            f"expert_parallel_size {expert_parallel_size}"
        )
    if experts_per_token > expected_shape[1]:
        raise ValueError(
            f"experts_per_token {experts_per_token} exceeds num_logical_experts {expected_shape[1]}"
        )

    def normalize(counts: list[int]) -> list[float]:
        total = sum(counts)
        return [count / total for count in counts] if total else [0.0] * len(counts)

    all_layer_counts = [
        sum(layer[expert] for layer in aggregate_counts) for expert in range(expected_shape[1])
    ]
    summary = {
        "schema_version": 2,
        "model": expected_model,
        "num_moe_layers": expected_shape[0],
        "num_logical_experts": expected_shape[1],
        "expert_parallel_size": expert_parallel_size,
        "experts_per_rank": expected_shape[1] // expert_parallel_size,
        "experts_per_token": experts_per_token,
        "count_semantics": "logical_routed_token_assignments",
        "aggregation": {
            "scope": "all_captured_eplb_steps",
            "observed_eplb_step_min": min(record["eplb_step"] for record in records),
            "observed_eplb_step_max": max(record["eplb_step"] for record in records),
            "record_count": len(records),
        },
        # The current simulator projects logical expert ids onto contiguous EP
        # rank shards before removing rank/expert identity. Name that modeling
        # assumption explicitly; this is not claimed to be an observed EPLB
        # physical placement map.
        "expert_partitioning": {
            "kind": "contiguous_logical_expert_ids",
            "layout": "rank_major",
        },
        "counts_by_layer": aggregate_counts,
        "probabilities_by_layer": [normalize(layer) for layer in aggregate_counts],
        "counts_all_layers": all_layer_counts,
        "probabilities_all_layers": normalize(all_layer_counts),
    }
    Path(out_json).write_text(json.dumps(summary, indent=2))
    return len(records)


def build_server_env(fork_python: str, cuda_visible_devices: str) -> dict[str, str]:
    """A subprocess env pinned to the canonical fork source and its venv.

    Mirrors `ref/profile/attention/vllm_flashattention_profiler.py`: drop main's
    inherited Python/uv markers, prepend the fork venv bin to PATH, and point
    LD_LIBRARY_PATH at the fork's torch libs. PYTHONPATH is then rebuilt to the
    checked-out `alignment/profiler/vllm` source so an already-provisioned venv
    cannot silently import a different editable checkout; binary extensions and
    third-party dependencies still come from the selected venv.
    """
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)

    venv_bin = str(Path(fork_python).parent)
    # Prepend the venv bin *and* the CUDA bin dir so Ray's nsight plugin finds
    # `nsys` on PATH (it shells out to `nsys profile ... python` for each worker).
    path_parts = [venv_bin]
    for cuda_bin in ("/usr/local/cuda-12.8/bin", "/usr/local/cuda/bin"):
        if Path(cuda_bin, "nsys").exists():
            path_parts.append(cuda_bin)
            break
    env["PATH"] = ":".join([*path_parts, env.get("PATH", "")])
    env["PYTHONPATH"] = str(VLLM_SOURCE_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

    torch_lib = Path(fork_python).parents[1] / "lib"
    # site-packages/torch/lib holds libc10/libtorch; find it under the venv.
    for cand in Path(fork_python).parents[1].glob("lib/python*/site-packages/torch/lib"):
        env["LD_LIBRARY_PATH"] = f"{cand}:{env.get('LD_LIBRARY_PATH', '')}"
        break
    else:
        if torch_lib.exists():
            env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def _git_info(repo: Path) -> dict | None:
    if not (repo / ".git").exists():
        return None

    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    return {
        "path": str(repo),
        "head": run(["rev-parse", "HEAD"]),
        "branch": run(["branch", "--show-current"]),
        "status_short": run(["status", "--short"]),
    }


def write_launch_metadata(
    path: Path,
    argv: list[str],
    env: dict,
    cfg: ProfileConfig,
    fork_python: str,
) -> None:
    """Persist argv + selected env + fork git state next to the server log.

    So an alignment result is self-describing without recovering flags from log
    text (reference `_write_launch_metadata` contract).
    """
    keep = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_NVTX_SCOPES_FOR_PROFILING",
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
    )
    metadata = {
        "schema_version": 1,
        "name": cfg.name,
        "argv": argv,
        "env": {k: env[k] for k in keep if k in env},
        "server_config": cfg.server.__dict__,
        "nsys_config": cfg.nsys.__dict__,
        "fork_python": fork_python,
        "fork_git": _git_info(VLLM_SOURCE_ROOT),
    }
    Path(path).write_text(json.dumps(metadata, indent=2))


def _get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except (urllib.error.URLError, OSError):
        return None, None


def set_cuda_profile(base_url: str, *, active: bool, timeout: float = 120.0) -> None:
    """Start or stop vLLM's worker-owned CUDA profiler through its HTTP API.

    Stopping can block while Nsight finalizes a large report inside the worker;
    the HTTP response is therefore allowed to outlive the ordinary health-check
    timeout. The launcher still owns process shutdown if this bound is exceeded.
    """
    action = "start_profile" if active else "stop_profile"
    request = urllib.request.Request(f"{base_url}/{action}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM /{action} returned HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"vLLM /{action} failed: {exc}") from exc


def wait_for_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    """Block until a health endpoint returns 200, or the server process exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM server exited with code {code} before becoming ready")
        for ep in HEALTH_ENDPOINTS:
            status, _ = _get(f"{base_url}{ep}")
            if status == 200:
                return
        time.sleep(2)
    raise TimeoutError(f"vLLM server not ready within {timeout}s")


def extract_metrics_jsonl(server_log: Path, out_jsonl: Path) -> int:
    """Extract canonical structured iteration records into a metrics JSONL.

    The fork emits one `VibeSimAlignmentIteration {json}` line per model step.
    Exact prefill/decode shapes are retained for the typed timing-predict input
    adapter; `nsys_parse` also uses `prefill_tokens` for stage tagging.
    """
    required = {
        "schema_version",
        "input_adapter",
        "iteration_index",
        "prefill_tokens",
        "decode_requests",
        "decode_tokens_scheduled",
        "prefill_chunk_pairs",
        "decode_kv_lens",
    }
    n = 0
    with Path(out_jsonl).open("w") as out:
        for line in Path(server_log).read_text(errors="replace").splitlines():
            m = _ALIGNMENT_ITERATION_RE.search(line)
            if not m:
                continue
            row = json.loads(m.group(1))
            missing = required - set(row)
            if missing:
                raise ValueError(f"alignment iteration record missing fields {sorted(missing)}")
            if row["schema_version"] != 1 or row["input_adapter"] != "vllm_text":
                raise ValueError(
                    "unsupported alignment iteration record "
                    f"schema={row['schema_version']!r} adapter={row['input_adapter']!r}"
                )
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def extract_request_timings_jsonl(
    server_log: Path,
    out_jsonl: Path,
    *,
    expected_request_ids: set[str] | None = None,
) -> int:
    """Extract one complete EngineCore timing record per request.

    Schema v1 contains TTFT only. Schema v2 adds first-token → last-token TPOT;
    all boundaries use EngineCore monotonic timestamps and therefore exclude
    HTTP/frontend ingress, SSE return, and client completion accounting.
    """
    ttft_duration_fields = {
        "engine_core_ttft_ms",
        "engine_queue_wait_ms",
        "engine_first_schedule_to_first_token_ms",
    }
    tpot_fields = {
        "engine_core_decode_ms",
        "engine_core_tpot_ms",
        "num_output_tokens",
    }
    required = {"schema_version", *ttft_duration_fields}
    request_ids: set[str] = set()
    n = 0
    with Path(out_jsonl).open("w") as out:
        for line in Path(server_log).read_text(errors="replace").splitlines():
            match = _ALIGNMENT_REQUEST_TIMING_RE.search(line)
            if not match:
                continue
            row = json.loads(match.group(1))
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"alignment request timing record missing fields {sorted(missing)}"
                )
            schema_version = row["schema_version"]
            if schema_version not in {1, 2}:
                raise ValueError(f"unsupported alignment request timing schema {schema_version!r}")
            if schema_version == 2:
                missing = tpot_fields - set(row)
                if missing:
                    raise ValueError(
                        "alignment request timing schema v2 record missing fields "
                        f"{sorted(missing)}"
                    )
            # The vLLM OpenAI completions frontend wraps X-Request-Id before
            # enqueueing it as `cmpl-<source-id>-0`. TraceLab sends one prompt
            # per request, so index 0 is the only supported alignment shape.
            # `request_id` was the raw field name in the first instrumented run;
            # accept it so an in-flight capture remains extractable.
            engine_request_id = row.get("engine_request_id", row.get("request_id"))
            if not isinstance(engine_request_id, str) or not engine_request_id:
                raise ValueError("alignment request timing engine_request_id must be non-empty")
            completion_match = _COMPLETION_ENGINE_REQUEST_RE.fullmatch(engine_request_id)
            request_id = (
                completion_match.group(1) if completion_match is not None else engine_request_id
            )
            # vLLM may issue frontend-owned prefix-cache probes before TraceLab
            # starts the replay. The replay's successful request ids are the
            # authoritative experiment population; unrelated timing records are
            # intentionally excluded instead of relying on a fixed probe count.
            if expected_request_ids is not None and request_id not in expected_request_ids:
                continue
            if request_id in request_ids:
                raise ValueError(f"duplicate alignment request timing for {request_id!r}")
            request_ids.add(request_id)
            row["engine_request_id"] = engine_request_id
            row["request_id"] = request_id
            for field in ttft_duration_fields:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"alignment request timing {field} must be nonnegative")
            components_ms = (
                row["engine_queue_wait_ms"] + row["engine_first_schedule_to_first_token_ms"]
            )
            if abs(row["engine_core_ttft_ms"] - components_ms) > 1e-6:
                raise ValueError(
                    "alignment request timing components do not sum to engine_core_ttft_ms"
                )
            if schema_version == 2:
                num_output_tokens = row["num_output_tokens"]
                if (
                    isinstance(num_output_tokens, bool)
                    or not isinstance(num_output_tokens, int)
                    or num_output_tokens <= 0
                ):
                    raise ValueError(
                        "alignment request timing num_output_tokens must be a positive integer"
                    )
                decode_ms = row["engine_core_decode_ms"]
                if (
                    isinstance(decode_ms, bool)
                    or not isinstance(decode_ms, (int, float))
                    or not math.isfinite(decode_ms)
                    or decode_ms < 0
                ):
                    raise ValueError(
                        "alignment request timing engine_core_decode_ms must be nonnegative"
                    )
                tpot_ms = row["engine_core_tpot_ms"]
                if num_output_tokens == 1:
                    if tpot_ms is not None:
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms must be null "
                            "for a one-token request"
                        )
                else:
                    if (
                        isinstance(tpot_ms, bool)
                        or not isinstance(tpot_ms, (int, float))
                        or not math.isfinite(tpot_ms)
                        or tpot_ms < 0
                    ):
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms must be "
                            "nonnegative for a multi-token request"
                        )
                    expected_tpot_ms = decode_ms / (num_output_tokens - 1)
                    if abs(tpot_ms - expected_tpot_ms) > 1e-6:
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms does not equal "
                            "engine_core_decode_ms/(num_output_tokens-1)"
                        )
            out.write(json.dumps(row) + "\n")
            n += 1
    if expected_request_ids is not None and request_ids != expected_request_ids:
        missing = sorted(expected_request_ids - request_ids)
        extra = sorted(request_ids - expected_request_ids)
        raise ValueError(
            "alignment request timing ids do not match successful replay ids: "
            f"missing={missing[:8]!r} extra={extra[:8]!r}"
        )
    return n


def wait_for_idle(base_url: str, idle: IdleWaitConfig) -> bool:
    """Poll /load until in-flight requests drain to zero (server_load == 0).

    Stock vLLM (no `--enable-server-load-tracking`) has no /load endpoint; there
    the caller has already awaited request completion, so a 404 just returns True.
    """
    if not idle.enabled:
        return True
    deadline = time.time() + idle.timeout
    saw_endpoint = False
    while time.time() < deadline:
        status, body = _get(f"{base_url}/load")
        if status == 404:
            return True  # endpoint absent → rely on completed requests
        if status == 200 and body is not None:
            saw_endpoint = True
            try:
                if json.loads(body).get("server_load", -1) == 0:
                    return True
            except json.JSONDecodeError:
                pass
        time.sleep(idle.poll_interval)
    return not saw_endpoint
