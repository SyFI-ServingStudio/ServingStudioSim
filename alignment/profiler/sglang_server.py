"""Launch one instrumented SGLang server, the sibling of `vllm_server.py`.

Same job, same contract: build the argv and a clean subprocess env for a server
that lives in its own venv, then offer ready / profile / idle control over HTTP.
The alignment records it emits are the same four the vLLM fork emits, so nothing
downstream of the log is duplicated here -- `record_extraction.py` reads both.

What is genuinely different is worth naming, because it is the whole reason this
file exists rather than a flag on the vLLM one:

* **Flag spellings.** `--model-path`, `--tp-size`, `--chunked-prefill-size`,
  `--mem-fraction-static` where vLLM says `--model`, `--tensor-parallel-size`,
  `--max-num-batched-tokens`, `--gpu-memory-utilization`.
* **One knob that is not a spelling difference.** `max_cudagraph_capture_size`
  counts tokens for vLLM and has no SGLang counterpart in the same unit, so
  `build_server_argv` rejects it instead of forwarding it to a request-count
  flag.
* **Targeted capture.** vLLM has a dedicated `--profiler-config.profiler=cuda`
  launch flag; SGLang instead takes the activity per request, so the capture is
  armed with `{"activities": ["CUDA_PROFILER"]}` on `/start_profile` and needs no
  launch-time flag at all.
* **Idle.** `/v1/loads` rather than `/load`, and it reports a running/waiting
  split per DP rank rather than a single scalar.
* **Rank provenance.** Two halves, both different from vLLM's. Which rank
  emitted a record comes from SGLang's own `[<time> DP<k>]` log prefix rather
  than vLLM's `(EngineCore_DPk pid=N)`. Which device that rank ran on is stated
  outright: each scheduler is handed its `gpu_id`, so the instrumented fork
  emits a `VibeSimAlignmentWorker` record pairing device with rank, and no
  pid -> rank banner has to be joined against the profiler at all. See
  `extract_worker_device_ranks`.

The CUDA forward-compatibility hook is not SGLang-specific in principle, but is
in practice: SGLang HEAD pins a CUDA 13 stack (`torch==2.13`, `flashinfer[cu13]`)
whose user-mode driver a CUDA 12.8 host driver cannot satisfy. Pointing
`driver_compat_lib_dir` at an unpacked `cuda-compat-13-x` gives the process a
newer `libcuda.so.1` that still talks to the installed kernel module -- the
supported forward-compatibility path on data-center GPUs.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import IdleWaitConfig, ProfileConfig, ServerConfig

SGLANG_SOURCE_ROOT = Path(__file__).resolve().parent / "sglang" / "python"

HEALTH_ENDPOINTS = ("/health", "/v1/models")

# Nothing to add at launch: SGLang takes the profiler activity per request, so
# the CUDA profiler is selected on /start_profile instead of by a launch flag.
CUDA_PROFILER_SERVER_ARGS: tuple[str, ...] = ()

# Nothing: a capture measures the server as it actually runs. Overlap
# scheduling in particular must stay on -- it is what puts scheduler CPU work
# underneath GPU work, so a capture without it cannot show a launch gap or a
# GPU bubble, which is most of what the timeline is read for. Iterations stay
# separable anyway because kernels are attributed by the correlation id of the
# runtime call that issued them, and the `sglang_iteration(N): forward` range
# closes when that iteration's launches are done rather than when its result is
# processed.
NSYS_CAPTURE_SERVER_ARGS: tuple[str, ...] = ()

# What to switch off for a pass that must not carry timing instrumentation.
# Unlike vLLM's single gate, SGLang's alignment records and its NVTX ranges are
# separately gated, so this turns off only the ranges -- the records a
# popularity pass needs keep flowing.
TIMING_INSTRUMENTATION_OFF_ENV = {"SGLANG_ENABLE_NVTX_SCHEDULER": "0"}


def validate_nsys_capture_environment(
    fork_python: str,
    *,
    env: dict[str, str],
    cwd: Path,
) -> None:
    """Require the optional package that emits SGLang's iteration ranges.

    SGLang only warns when ``nvtx`` is unavailable and then continues serving.
    An NSYS run would therefore finish with millions of kernels but no
    ``sglang_iteration(N): forward`` ranges to attribute them to. Refuse that
    expensive unusable capture before launching the server.
    """
    probe = subprocess.run(
        [fork_python, "-c", "import nvtx"],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        timeout=30,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip() or probe.stdout.strip() or "import failed"
        raise RuntimeError(
            f"the SGLang fork venv cannot import `nvtx`: {fork_python}\n"
            "SGLang would serve without iteration markers, leaving captured "
            "kernels unattributed. Install it with:\n"
            f"  uv pip install --python {fork_python} nvtx\n"
            f"Import error: {detail}"
        )


def build_server_argv(fork_python: str, cfg: ServerConfig) -> list[str]:
    """The `python -m sglang.launch_server ...` argv."""
    argv = [
        fork_python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "--tp-size",
        str(cfg.tp_size),
        "--dp-size",
        str(cfg.dp_size),
        # SGLang chunks prefill by default; this is the chunk budget, the same
        # quantity vLLM spells `--max-num-batched-tokens`.
        "--chunked-prefill-size",
        str(cfg.chunk_size),
        "--mem-fraction-static",
        str(cfg.gpu_memory_utilization),
        # req-frontend's mandatory prefix-cache preflight reads
        # usage.prompt_tokens_details.cached_tokens, which SGLang only fills in
        # behind this flag. vLLM's counterpart is --enable-prompt-tokens-details.
        "--enable-cache-report",
    ]
    if cfg.enforce_eager:
        argv.append("--disable-cuda-graph")  # every kernel is a normal launch
    if cfg.max_cudagraph_capture_size is not None:
        # vLLM's field is a token ceiling. SGLang's nearest flag,
        # --cuda-graph-max-bs-decode, counts concurrent decode requests. There
        # is no conversion between those independent quantities; forwarding a
        # 2,048-token chunk budget would ask SGLang to capture graphs for 2,048
        # requests and can make startup appear hung.
        raise ValueError(
            "server.max_cudagraph_capture_size is a vLLM token ceiling and has no "
            "SGLang equivalent: --cuda-graph-max-bs-decode counts requests, not "
            "tokens. Drop the field for engine: sglang and, if a decode-graph "
            "ceiling is wanted, pass --cuda-graph-max-bs-decode in server.extra_args."
        )
    if cfg.api_server_count is not None:
        raise ValueError(
            "server.api_server_count selects vLLM HTTP API processes and has no "
            "SGLang equivalent; drop the field for engine: sglang."
        )
    if cfg.served_model_name:
        argv += ["--served-model-name", cfg.served_model_name]
    argv += list(cfg.extra_args)
    return argv


def build_server_env(
    fork_python: str,
    cuda_visible_devices: str,
    *,
    driver_compat_lib_dir: str = "",
) -> dict[str, str]:
    """A subprocess env pinned to the checked-out SGLang source and its venv.

    Same shape as the vLLM driver's: drop this process's Python markers so the
    child cannot import main's packages, prepend the venv bin, and rebuild
    PYTHONPATH to the checked-out source so an editable install elsewhere cannot
    win. `driver_compat_lib_dir`, when set, goes ahead of the torch libs so its
    `libcuda.so.1` is the one the child resolves.
    """
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)

    venv_bin = str(Path(fork_python).parent)
    env["PATH"] = os.pathsep.join([venv_bin, env.get("PATH", "")])
    env["PYTHONPATH"] = str(SGLANG_SOURCE_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["SGLANG_ENABLE_VIBESIM_ALIGNMENT"] = "1"
    # The `sglang_iteration(N): forward` range rides SGLang's own per-subsystem
    # NVTX gate rather than a private one, so the scheduler gate has to be on for
    # an NSYS capture to carry iteration-named spans.
    env["SGLANG_ENABLE_NVTX_SCHEDULER"] = "1"

    library_paths: list[str] = []
    if driver_compat_lib_dir:
        compat = Path(driver_compat_lib_dir)
        if not (compat / "libcuda.so.1").is_file():
            raise FileNotFoundError(
                f"driver_compat_lib_dir has no libcuda.so.1: {compat}\n"
                "Unpack a cuda-compat package matching the CUDA major version the "
                "SGLang venv's torch was built against."
            )
        library_paths.append(str(compat))
    for torch_lib in Path(fork_python).parents[1].glob("lib/python*/site-packages/torch/lib"):
        library_paths.append(str(torch_lib))
        break
    if library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*library_paths, env.get("LD_LIBRARY_PATH", "")])
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
    """Persist argv + selected env + fork git state next to the server log."""
    keep = (
        "CUDA_VISIBLE_DEVICES",
        "SGLANG_ENABLE_VIBESIM_ALIGNMENT",
        "SGLANG_ENABLE_NVTX_SCHEDULER",
        "PYTHONPATH",
        "PATH",
        "LD_LIBRARY_PATH",
    )
    runtime_env = cfg.python_runtime.environment if cfg.python_runtime is not None else {}
    metadata = {
        "schema_version": 1,
        "engine": "sglang",
        "name": cfg.name,
        "argv": launch_argv,
        "server_argv": server_argv,
        "env": {key: env[key] for key in (*keep, *runtime_env) if key in env},
        "server_config": cfg.server.__dict__,
        "nsys_config": cfg.nsys.__dict__,
        "nsys_profiler": profiler_provenance,
        "python_runtime": runtime_provenance,
        "fork_python": fork_python,
        "fork_git": _git_info(SGLANG_SOURCE_ROOT.parent),
    }
    Path(path).write_text(json.dumps(metadata, indent=2))


def speculative_decode_enabled(cfg: ServerConfig) -> bool:
    # No speculative alignment capture contract is implemented for this driver.
    return False


def _get(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except (urllib.error.URLError, OSError):
        return None, None


def set_cuda_profile(base_url: str, *, active: bool, timeout: float | None = None) -> None:
    """Arm or disarm the worker-owned CUDA profiler through SGLang's HTTP API.

    `activities=["CUDA_PROFILER"]` selects the `cudaProfilerStart/Stop` profiler
    -- the one an external `nsys profile --capture-range=cudaProfilerApi` is
    waiting on -- instead of the torch profiler that would write its own trace.
    """
    action = "start_profile" if active else "stop_profile"
    # Starting only arms CUPTI and should return promptly. Stopping also makes
    # NSYS finalize the report in the worker, which can take several minutes
    # for CUDA-graph-node traces with tens of millions of events.
    if timeout is None:
        timeout = 120.0 if active else 600.0
    payload = json.dumps({"activities": ["CUDA_PROFILER"]}).encode() if active else b"{}"
    request = urllib.request.Request(
        f"{base_url}/{action}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"SGLang /{action} returned HTTP {response.status}")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"SGLang /{action} failed: {exc}") from exc


def wait_for_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    """Block until a health endpoint returns 200, or the server process exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"SGLang server exited with code {code} before becoming ready")
        for endpoint in HEALTH_ENDPOINTS:
            status, _ = _get(f"{base_url}{endpoint}")
            if status == 200:
                return
        time.sleep(2)
    raise TimeoutError(f"SGLang server not ready within {timeout}s")


def extract_worker_device_ranks(server_log: Path) -> dict[int, int]:
    """Always empty: SGLang needs no pid <-> rank banner, and prints none.

    vLLM's parallel-state banner exists so the reader can join pid -> rank with
    the profiler's pid -> device and arrive at device -> rank. SGLang's workers
    publish device -> rank themselves, in the `VibeSimAlignmentWorker` record
    `record_extraction.extract_dp_rank_by_device` reads, so the join has no work
    to do and inventing a pid banner to feed it would only add a step that can
    disagree with the answer already given.
    """
    return {}


def verify_server_started(server_log: Path, cfg: ServerConfig) -> None:
    """SGLang has one HTTP frontend; nothing to confirm."""


def wait_for_idle(base_url: str, idle: IdleWaitConfig, cfg: ServerConfig | None = None) -> bool:
    """Poll /v1/loads until every DP rank reports no running or waiting request."""
    if not idle.enabled:
        return True
    deadline = time.time() + idle.timeout
    saw_endpoint = False
    while time.time() < deadline:
        status, body = _get(f"{base_url}/v1/loads?include=core")
        if status == 404:
            return True  # endpoint absent → rely on completed requests
        if status == 200 and body is not None:
            try:
                loads = json.loads(body)["loads"]
            except (json.JSONDecodeError, KeyError, TypeError):
                loads = None
            if loads is not None:
                saw_endpoint = True
                if all(
                    load["num_running_reqs"] == 0 and load["num_waiting_reqs"] == 0
                    for load in loads
                ):
                    return True
        time.sleep(idle.poll_interval)
    return not saw_endpoint
