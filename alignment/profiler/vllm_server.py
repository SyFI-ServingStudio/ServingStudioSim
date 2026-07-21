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
