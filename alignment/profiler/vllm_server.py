"""Launch one instrumented-fork vLLM server on one GPU (single-GPU milestone).

A lean, single-GPU descendant of the reference harness's `launchers/vllm.py` +
`base.py`. Builds the server argv and a **clean subprocess env** (the vLLM/torch
runtime lives in a separate fork venv — this package's interpreter never imports
vLLM), records launch metadata, and offers ready/idle polling over the
OpenAI-compatible endpoints. The actual spawn is wrapped by `nsys_capture.py` and
orchestrated in `__main__.py`.

The fork (`alignment/profiler/vllm`, branch `moesim-profile`) adds the
`vllm_iteration(N): <phase>` NVTX scopes (gated by `VLLM_NVTX_SCOPES_FOR_PROFILING`)
and a versioned `VibeSimAlignmentIteration {json}` record containing the exact
model input shape consumed by the typed predictor adapter.
"""

from __future__ import annotations

import json
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


def build_server_argv(fork_python: str, cfg: ServerConfig) -> list[str]:
    """The `python -m vllm.entrypoints.openai.api_server ...` argv (single GPU)."""
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


def set_cuda_profile(base_url: str, *, active: bool, timeout: float = 30.0) -> None:
    """Start or stop vLLM's worker-owned CUDA profiler through its HTTP API."""
    action = "start_profile" if active else "stop_profile"
    request = urllib.request.Request(
        f"{base_url}/{action}", data=b"", method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"vLLM /{action} returned HTTP {response.status}"
                )
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
                raise ValueError(
                    f"alignment iteration record missing fields {sorted(missing)}"
                )
            if row["schema_version"] != 1 or row["input_adapter"] != "vllm_text":
                raise ValueError(
                    "unsupported alignment iteration record "
                    f"schema={row['schema_version']!r} adapter={row['input_adapter']!r}"
                )
            out.write(json.dumps(row) + "\n")
            n += 1
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
