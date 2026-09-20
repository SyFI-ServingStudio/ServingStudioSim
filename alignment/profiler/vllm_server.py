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
separately from req-frontend's client-observed latency accounting. Reading those
records back is engine-independent and lives in `record_extraction.py`; what
stays here is only what is specific to launching vLLM.
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

# Enables the /start_profile and /stop_profile routes. Those routes fan CUDA
# profiler control into the actual GPU worker, which is the reliable
# targeted-capture boundary for the spawned EngineCore.
CUDA_PROFILER_SERVER_ARGS = ("--profiler-config.profiler=cuda",)

# Nothing extra: vLLM's per-iteration NVTX range already brackets one forward,
# and the eager/CUDA-graph tradeoff a capture may want is a `ServerConfig` knob
# rather than something the driver decides.
NSYS_CAPTURE_SERVER_ARGS: tuple[str, ...] = ()

# What to switch off for a pass that must not carry timing instrumentation.
# In vLLM this one flag gates both the NVTX scopes and the EngineCore request
# timing records, so a pass launched with it produces neither.
TIMING_INSTRUMENTATION_OFF_ENV = {"VLLM_NVTX_SCOPES_FOR_PROFILING": "0"}


def validate_nsys_capture_environment(
    _fork_python: str,
    *,
    env: dict[str, str],
    cwd: Path,
) -> None:
    """vLLM's instrumented fork has no optional NSYS marker dependency."""


# vLLM's parallel-state banner, emitted once per worker process at startup. It
# is the only place the worker pid ↔ global rank identity is stated; nsys knows
# pid ↔ CUDA device, so the two together give rank ↔ device.
_WORKER_RANK_RE = re.compile(
    r"^\(Worker[^)]*\s+pid=(\d+)\).*\bworld_size=(\d+)\s+rank=(\d+)\s+local_rank=(\d+)"
)


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
        "--data-parallel-size",
        str(cfg.dp_size),
        "--max-num-batched-tokens",
        str(cfg.chunk_size),
        "--enable-chunked-prefill",
        # req-frontend's mandatory prefix-cache preflight reads
        # usage.prompt_tokens_details.cached_tokens. Keep the paired server
        # observable by default instead of requiring every experiment preset
        # to repeat this transport-level flag.
        "--enable-prompt-tokens-details",
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    if cfg.enforce_eager:
        argv.append("--enforce-eager")  # no CUDA graphs → clean per-kernel nsys records
    if cfg.enable_server_load_tracking:
        argv.append("--enable-server-load-tracking")  # /load endpoint for idle checks
    if cfg.enable_iteration_metrics:
        argv.append("--enable-logging-iteration-details")
    if cfg.max_cudagraph_capture_size is not None:
        argv += ["--max-cudagraph-capture-size", str(cfg.max_cudagraph_capture_size)]
    if cfg.served_model_name:
        argv += ["--served-model-name", cfg.served_model_name]
    # Historical presets may still carry the now-default flag. Preserve their
    # semantics without emitting a duplicate CLI option in launch metadata.
    argv += [
        argument for argument in cfg.extra_args if argument != "--enable-prompt-tokens-details"
    ]
    return argv


def build_server_env(
    fork_python: str,
    cuda_visible_devices: str,
    *,
    driver_compat_lib_dir: str = "",
) -> dict[str, str]:
    """A subprocess env pinned to the canonical fork source and its venv.

    Mirrors `ref/profile/attention/vllm_flashattention_profiler.py`: drop main's
    inherited Python/uv markers, prepend the fork venv bin to PATH, and point
    LD_LIBRARY_PATH at the fork's torch libs. PYTHONPATH is then rebuilt to the
    checked-out `alignment/profiler/vllm` source so an already-provisioned venv
    cannot silently import a different editable checkout; binary extensions and
    third-party dependencies still come from the selected venv.
    """
    env = dict(os.environ)
    # The profiler owns an unauthenticated loopback server and its paired load
    # generator. An unrelated shell-level VLLM_API_KEY would otherwise make
    # every replay preflight fail with HTTP 401; auth is not part of this local
    # measurement boundary.
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "VLLM_API_KEY",
    ):
        env.pop(key, None)

    venv_bin = str(Path(fork_python).parent)
    env["PATH"] = os.pathsep.join([venv_bin, env.get("PATH", "")])
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
    if driver_compat_lib_dir:
        # Ahead of the torch libs, so this `libcuda.so.1` is the one resolved.
        # Unused by the stock setup, where the host driver already satisfies the
        # venv's CUDA; kept symmetric with the SGLang driver so a config field
        # does not silently do nothing on one engine.
        env["LD_LIBRARY_PATH"] = f"{driver_compat_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
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
    launch_argv: list[str],
    server_argv: list[str],
    env: dict,
    cfg: ProfileConfig,
    fork_python: str,
    profiler_provenance: dict[str, str] | None,
    runtime_provenance: dict | None,
) -> None:
    """Persist argv + selected env + fork git state next to the server log.

    So an alignment result is self-describing without recovering flags from log
    text (reference `_write_launch_metadata` contract).
    """
    keep = (
        "CUDA_VISIBLE_DEVICES",
        "VLLM_NVTX_SCOPES_FOR_PROFILING",
        "VLLM_SERVER_DEV_MODE",
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
    )
    runtime_env = cfg.python_runtime.environment if cfg.python_runtime is not None else {}
    metadata = {
        "schema_version": 1,
        "name": cfg.name,
        "argv": launch_argv,
        "server_argv": server_argv,
        "env": {key: env[key] for key in (*keep, *runtime_env) if key in env},
        "server_config": cfg.server.__dict__,
        "nsys_config": cfg.nsys.__dict__,
        "nsys_profiler": profiler_provenance,
        "python_runtime": runtime_provenance,
        "fork_python": fork_python,
        "fork_git": _git_info(VLLM_SOURCE_ROOT),
    }
    Path(path).write_text(json.dumps(metadata, indent=2))


def speculative_decode_enabled(cfg: ServerConfig) -> bool:
    return any(arg.split("=", 1)[0] == "--speculative-config" for arg in cfg.extra_args)


def _speculative_config(cfg: ServerConfig) -> dict:
    args = list(cfg.extra_args)
    for index, arg in enumerate(args):
        key, sep, inline = arg.partition("=")
        if key != "--speculative-config":
            continue
        raw = inline if sep else (args[index + 1] if index + 1 < len(args) else "")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def draft_has_moe_experts(cfg: ServerConfig) -> bool:
    """Whether the draft model routes tokens to experts.

    Expert popularity is only defined for a draft that has experts. An MTP
    draft layer lives in the target checkpoint and shares its MoE, so it does.
    A draft named by its own `model` path is a separate checkpoint and may be
    dense -- DFlash2 is six dense GQA layers with a plain SwiGLU MLP -- and
    then EPLB has nothing to report for it. Asking anyway fails the whole
    popularity pass with `no VibeSimAlignmentExpertLoad records found`, which
    reads like broken instrumentation rather than an absent MoE.
    """
    if not speculative_decode_enabled(cfg):
        return False
    spec = _speculative_config(cfg)
    draft_path = spec.get("model")
    if not draft_path:
        # No separate checkpoint: the draft is part of the target (MTP), so it
        # routes through the target's experts.
        return True
    config_path = Path(draft_path) / "config.json"
    try:
        draft_config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        # Unreadable draft config is not evidence of absence; keep the previous
        # behaviour and let the extraction report what it finds.
        return True
    return any(
        key in draft_config
        for key in ("n_routed_experts", "num_experts", "num_local_experts")
    )


def _parse_spec_decode_metrics(text: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    names = {
        f"vllm:spec_decode_{name}_total": name
        for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    }
    position_name = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    totals: dict[str, int] = {}
    positions: dict[str, int] = {}
    seen: set[tuple] = set()
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in names and sample.name != position_name:
                continue
            identity = (sample.name, tuple(sorted(sample.labels.items())))
            if identity in seen:
                raise ValueError(f"duplicate spec-decode counter: {identity}")
            seen.add(identity)
            value = sample.value
            if not value.is_integer() or value < 0:
                raise ValueError(f"invalid spec-decode counter: {sample.name}={value}")
            if sample.name == position_name:
                position = sample.labels.get("position", "")
                if not position.isdecimal() or str(int(position)) != position:
                    raise ValueError(f"invalid spec-decode position: {position!r}")
                positions[position] = positions.get(position, 0) + int(value)
            else:
                name = names[sample.name]
                totals[name] = totals.get(name, 0) + int(value)
    if set(totals) != set(names.values()) or not positions:
        raise ValueError("missing required spec-decode Prometheus counters")
    return {**totals, "accepted_per_position": positions}


def fetch_spec_decode_metrics(base_url: str) -> dict:
    status, body = _get(f"{base_url}/metrics")
    if status != 200 or body is None:
        raise RuntimeError("cannot fetch vLLM spec-decode Prometheus counters")
    return _parse_spec_decode_metrics(body.decode())


def _get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except (urllib.error.URLError, OSError):
        return None, None


def set_cuda_profile(base_url: str, *, active: bool, timeout: float | None = None) -> None:
    """Start or stop vLLM's worker-owned CUDA profiler through its HTTP API.

    Stopping can block while Nsight finalizes a large report inside the worker;
    the HTTP response is therefore allowed to outlive the ordinary health-check
    timeout. The launcher still owns process shutdown if this bound is exceeded.
    """
    action = "start_profile" if active else "stop_profile"
    # Starting only arms CUPTI and should return promptly. Stopping also makes
    # NSYS finalize the report in the worker, which can take several minutes
    # for CUDA-graph-node traces with tens of millions of events.
    if timeout is None:
        timeout = 120.0 if active else 600.0
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


def extract_worker_device_ranks(server_log: Path) -> dict[int, int]:
    """worker pid → global rank, from vLLM's per-worker parallel-state banner.

    The caller pairs this with nsys's pid → CUDA device mapping to learn which
    device executed which rank. Returned empty when the log carries no banner
    (single-process captures), which the parser treats as "one rank on device 0".
    """
    ranks: dict[int, int] = {}
    world_sizes: set[int] = set()
    for line in Path(server_log).read_text(errors="replace").splitlines():
        match = _WORKER_RANK_RE.match(line)
        if not match:
            continue
        pid, world_size, rank = int(match.group(1)), int(match.group(2)), int(match.group(3))
        previous = ranks.setdefault(pid, rank)
        if previous != rank:
            raise ValueError(f"worker pid {pid} reports two ranks: {previous} and {rank}")
        world_sizes.add(world_size)
    if len(world_sizes) > 1:
        raise ValueError(f"workers disagree on world_size: {sorted(world_sizes)}")
    if ranks and len(set(ranks.values())) != len(ranks):
        raise ValueError(f"worker ranks are not unique: {sorted(ranks.items())}")
    if world_sizes and len(ranks) != next(iter(world_sizes)):
        raise ValueError(
            f"found {len(ranks)} worker rank banners but world_size={next(iter(world_sizes))}"
        )
    return ranks


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
