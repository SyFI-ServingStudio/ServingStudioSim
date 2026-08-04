"""Configuration for one real vLLM profiling run.

`ProfileConfig` pins the vLLM launch (fork venv / model / GPU / chunk budget),
the workload that drives it, the Nsight capture window, and this phase's single
artifact root. YAML/JSON parsing belongs to ``launcher.alignment_config``; this
module only defines the typed runtime values consumed by profiling code.

`ServerConfig` / `IdleWaitConfig` are the launch-level knobs consumed by
`vllm_server.py`. One profile represents one replica and may expose several
devices when tensor parallelism is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..load_generator.config import LoadGeneratorConfig


@dataclass
class ServerConfig:
    """How to launch one vLLM server for one TP x DP replica group."""

    model_path: str
    host: str = "127.0.0.1"
    port: int = 8000
    # vLLM's `--max-num-batched-tokens`; also the chunked-prefill chunk budget.
    chunk_size: int = 2048
    enable_iteration_metrics: bool = True  # → --enable-logging-iteration-details
    gpu_memory_utilization: float = 0.85
    tp_size: int = 1
    # Data-parallel ranks may also form the expert-parallel group.  Keep this
    # explicit instead of hiding it in ``extra_args`` because it determines the
    # visible-device and normalized-profile population contracts.
    dp_size: int = 1
    served_model_name: str | None = None
    startup_timeout: float = 900.0
    # `--enforce-eager`: disable CUDA graphs so every kernel is a normal launch.
    # Under CUDA graphs an *external* `nsys profile` wrapper records replayed
    # kernels only at graph-capture time (no per-op busy time during serving);
    # eager sidesteps that but changes the kernel set (torch.compile fusion off).
    enforce_eager: bool = False
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

    # Exact executable path. When omitted, the runner requires NSYS_BIN. It never
    # guesses from PATH or machine-specific installation directories because
    # profiler version is part of an alignment run's provenance.
    executable: str | None = None
    # "cuda_profiler_api" (worker-owned), "nvtx" (range trigger), or "full".
    capture_mode: str = "cuda_profiler_api"
    # Exact NVTX range whose push starts the capture. Stock vLLM commonly uses
    # "gpu_model_runner: forward"; indexed fork ranges can be supplied directly.
    nvtx_trigger: str = "gpu_model_runner: forward"
    # Analysis window [start, end] over the (sequential, per-worker) forward index.
    analyze_iteration_start: int = 24
    analyze_iteration_end: int = 48
    # Stop CUPTI after a bounded serving window while the replay continues.
    # Long graph-node captures can perturb or deadlock multi-rank collectives;
    # request TTFT/TPOT logging remains active after capture stops.
    capture_duration_seconds: float | None = None
    sample: str = "none"
    cpuctxsw: str = "none"
    cuda_graph_trace: str = "node"
    # Nsight 2025.1 device-side event tracing caused Xid 32 with FlashInfer's
    # multi-GPU NVLink all-to-all. Keep the safe default explicit and configurable.
    cuda_event_trace: bool = False

    def validate(self) -> None:
        allowed = {"cuda_profiler_api", "nvtx", "full"}
        if self.capture_mode not in allowed:
            raise ValueError(
                f"nsys.capture_mode must be one of {sorted(allowed)}, got {self.capture_mode!r}"
            )
        if self.capture_duration_seconds is not None:
            if self.capture_mode != "cuda_profiler_api":
                raise ValueError(
                    "nsys.capture_duration_seconds requires capture_mode=cuda_profiler_api"
                )
            if self.capture_duration_seconds <= 0:
                raise ValueError("nsys.capture_duration_seconds must be positive")


@dataclass
class ProfileConfig:
    """One real-server ground-truth profiling run."""

    # --- identity / output ---
    name: str
    log_dir: str
    gpu: str  # DB/display GPU name, e.g. "NVIDIA H200"
    cuda_visible_devices: str = "0"  # comma-separated physical GPUs; count equals tp_size

    # ``nsys`` is the timing/segment capture consumed by kernel alignment.
    # ``workload_metrics`` is a clean full-run scheduler/request timing pass.
    # ``expert_popularity`` is a separate, deliberately unprofiled pass whose
    # synchronization and D2H logging overhead must not contaminate timing.
    profile_kind: str = "nsys"

    # --- vLLM side ---
    fork_python: str = ""  # abs path to the instrumented-fork venv python
    server: ServerConfig = None  # type: ignore[assignment]
    idle: IdleWaitConfig = field(default_factory=IdleWaitConfig)
    nsys: NsysConfig = field(default_factory=NsysConfig)

    # --- workload: TraceLab consumes the same source trace as the simulator ---
    workload: LoadGeneratorConfig = None  # type: ignore[assignment]
