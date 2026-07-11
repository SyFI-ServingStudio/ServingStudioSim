"""Configuration for one real vLLM profiling run.

`ProfileConfig` pins the vLLM launch (fork venv / model / GPU / chunk budget),
the workload that drives it, the Nsight capture window, and this phase's single
artifact root. YAML/JSON parsing belongs to ``launcher.alignment_config``; this
module only defines the typed runtime values consumed by profiling code.

`ServerConfig` / `IdleWaitConfig` are the launch-level knobs consumed by
`vllm_server.py` (a lean, single-GPU descendant of the reference harness's
`launchers/config.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..load_generator.config import LoadGeneratorConfig


@dataclass
class ServerConfig:
    """How to launch one vLLM server on one GPU (single-GPU, tp=1 milestone)."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    # vLLM's `--max-num-batched-tokens`; also the chunked-prefill chunk budget.
    chunk_size: int = 2048
    enable_iteration_metrics: bool = True  # → --enable-logging-iteration-details
    gpu_memory_utilization: float = 0.85
    tp_size: int = 1
    served_model_name: str | None = None
    startup_timeout: float = 900.0
    # `--enforce-eager`: disable CUDA graphs so every kernel is a normal launch.
    # Under CUDA graphs an *external* `nsys profile` wrapper records replayed
    # kernels only at graph-capture time (no per-op busy time during serving);
    # eager sidesteps that but changes the kernel set (torch.compile fusion off).
    enforce_eager: bool = False
    # `--distributed-executor-backend ray --ray-workers-use-nsight`: run the model
    # worker as a Ray actor that Ray launches *under nsys* (its runtime_env sets
    # `cuda-graph-trace=node`). This is the only way to capture CUDA-graph *replay*
    # per-node kernels (the reference's method) — nsys attaches to the worker
    # process directly, so serving-time graph replays record kernels with
    # graphNodeId + correlationId inside the forward NVTX ranges. Mutually
    # exclusive with the external-nsys capture path in `alignment.runner.run_profile`.
    use_ray_nsight: bool = False
    # vLLM CUDA-graph capture ceiling; for a trace-derived chunk budget it must
    # cover chunk_size, else prefill iters fall back to a different path.
    max_cudagraph_capture_size: int | None = None
    # `--enable-server-load-tracking` (the /load idle endpoint). Present on the
    # fork; absent on stock vLLM 0.19.1 — leave False there and rely on request
    # completion + a drain wait for idle.
    enable_server_load_tracking: bool = False
    extra_args: list[str] = field(default_factory=list)


@dataclass
class IdleWaitConfig:
    """Poll vLLM's /load endpoint until in-flight requests drain to zero."""

    enabled: bool = True
    timeout: float = 120.0
    poll_interval: float = 0.2


@dataclass
class NsysConfig:
    """Nsight Systems targeted-capture knobs (see `nsys_capture.py`).

    ``cuda_profiler_api`` is the OpenAI-server default recommended by vLLM: the
    launcher calls `/start_profile` and `/stop_profile`, which arm CUPTI from the
    CUDA-owning worker rather than from the parent API process. CUDA graphs stay
    enabled and node tracing exposes their replayed kernels.
    """

    # "cuda_profiler_api" (worker-owned), "nvtx" (range trigger), or "full".
    capture_mode: str = "cuda_profiler_api"
    # NVTX range whose push starts the capture. Stock vLLM: "gpu_model_runner: forward";
    # the fork's indexed form: f"vllm_iteration({trigger_iteration}): forward".
    nvtx_trigger: str = "gpu_model_runner: forward"
    trigger_iteration: int = 8  # only used when nvtx_trigger embeds an index
    # Analysis window [start, end] over the (sequential, per-worker) forward index.
    analyze_iteration_start: int = 24
    analyze_iteration_end: int = 48
    sample: str = "none"
    cpuctxsw: str = "none"
    cuda_graph_trace: str = "node"

    def validate(self) -> None:
        allowed = {"cuda_profiler_api", "nvtx", "full"}
        if self.capture_mode not in allowed:
            raise ValueError(
                f"nsys.capture_mode must be one of {sorted(allowed)}, got {self.capture_mode!r}"
            )


@dataclass
class ProfileConfig:
    """One real-server ground-truth profiling run."""

    # --- identity / output ---
    name: str
    log_dir: str
    gpu: str  # DB/display GPU name, e.g. "NVIDIA H200"
    cuda_visible_devices: str = "0"  # the single physical GPU to run on

    # --- vLLM side ---
    fork_python: str = ""  # abs path to the instrumented-fork venv python
    server: ServerConfig = None  # type: ignore[assignment]
    idle: IdleWaitConfig = field(default_factory=IdleWaitConfig)
    nsys: NsysConfig = field(default_factory=NsysConfig)

    # --- workload: TraceLab consumes the same source trace as the simulator ---
    workload: LoadGeneratorConfig = None  # type: ignore[assignment]
