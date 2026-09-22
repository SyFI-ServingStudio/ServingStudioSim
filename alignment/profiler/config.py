"""Configuration for one real serving-engine profiling run.

`ProfileConfig` pins the engine launch (fork venv / model / GPU / chunk budget),
the workload that drives it, the Nsight capture window, and this phase's single
artifact root. YAML/JSON parsing belongs to ``launcher.alignment_config``; this
module only defines the typed runtime values consumed by profiling code.

`ServerConfig` / `IdleWaitConfig` are the launch-level knobs consumed by the
engine driver (`vllm_server.py` or `sglang_server.py`) selected by `engine`. The
fields are stated in vLLM's vocabulary because that came first; each driver
translates them into its own flag spellings. One profile represents one replica
and may expose several devices when tensor parallelism is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..load_generator.config import LoadGeneratorConfig


@dataclass(frozen=True)
class PythonPackageArtifact:
    """One wheel-backed runtime artifact required before model loading.

    ``version`` is the upstream release version passed to the installer.
    CUDA-specific wheels may append a local version such as ``+cu130``;
    ``local_version`` makes that suffix part of the validation contract without
    changing the upstream installation spelling.
    """

    name: str
    version: str
    index_url: str
    local_version: str | None = None
    required_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PythonRuntimeConfig:
    """Idempotent package preparation and immutable server-runtime policy."""

    packages: list[PythonPackageArtifact]
    environment: dict[str, str] = field(default_factory=dict)
    lock_timeout_seconds: float = 900.0
    install_timeout_seconds: float = 1800.0


@dataclass
class ServerConfig:
    """How to launch one server for one TP x DP replica group."""

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
    # Number of ranks across which the expert axis is sharded. Every
    # expert-popularity profile must state this in YAML. A token_corpus profile
    # may state it to get the marginal as well, and omitting it only means the
    # pass produces routes alone; other profile kinds do not consume it.
    expert_parallel_size: int | None = None
    # Number of ranks already represented by each synchronized expert-count
    # record. This is independent of expert sharding and follows the same rule.
    expert_count_reduction_group_size: int | None = None
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
    # Analysis window [start, end] over the (sequential, per-worker) forward
    # index. Open bounds analyze the entire capture; set them only when an
    # intentional numbered excerpt is wanted.
    analyze_iteration_start: int | None = None
    analyze_iteration_end: int | None = None
    # Stop CUPTI after a bounded serving window while the replay continues.
    # Long graph-node captures can perturb or deadlock multi-rank collectives;
    # request TTFT/TPOT logging remains active after capture stops.
    capture_duration_seconds: float | None = None
    sample: str = "none"
    cpuctxsw: str = "none"
    cuda_graph_trace: str = "node"
    # Periodically persist buffered CUDA records during long captures.
    cuda_flush_interval_ms: int | None = None
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
        if self.cuda_flush_interval_ms is not None and self.cuda_flush_interval_ms <= 0:
            raise ValueError("nsys.cuda_flush_interval_ms must be positive")
        if (
            self.analyze_iteration_start is not None
            and self.analyze_iteration_end is not None
            and self.analyze_iteration_start > self.analyze_iteration_end
        ):
            raise ValueError(
                "nsys.analyze_iteration_start must not exceed analyze_iteration_end; "
                "omit both to analyze the whole capture"
            )


#: Every `profile_kind` a profile config may name.
PROFILE_KINDS = frozenset({"nsys", "workload_metrics", "expert_popularity", "token_corpus"})

#: The passes that observe MoE routing. They launch the server bare with the
#: engine's timing instrumentation off, and both need the expert topology
#: declared because their records are already rank-reduced.
ROUTING_PROFILE_KINDS = frozenset({"expert_popularity", "token_corpus"})


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
    # ``token_corpus`` records the routed experts of every accepted token and
    # packs them into the corpus the simulator's ``routing: corpus`` samples. It
    # keeps the per-step expert-load stream on as well, so one pass yields the
    # corpus, the per-expert marginal, and the ground truth to score both.
    # ``expert_popularity`` is the same pass without the per-token routes and is
    # DEPRECATED: a marginal can only be resampled independently, and a serving
    # batch is not independent. Prefer ``token_corpus`` for anything that drafts.
    # Both are deliberately unprofiled -- their synchronization and D2H logging
    # overhead must not contaminate timing.
    profile_kind: str = "nsys"

    # --- engine side ---
    # Which serving engine to launch and whose records to expect. Both forks
    # emit the identical alignment records, so this selects the launch driver
    # and the log dialect, not the analysis that follows.
    engine: str = "vllm"
    fork_python: str = ""  # abs path to the instrumented-fork venv python
    # Directory holding a forward-compatible `libcuda.so.1` (an unpacked
    # `cuda-compat-<major>-<minor>` package), prepended to the child's library
    # path. Needed when the engine venv's torch was built against a newer CUDA
    # than the host driver's user-mode library provides; empty means the host
    # driver is already new enough. Recorded in launch metadata because it
    # changes which user-mode driver produced the measurement.
    driver_compat_lib_dir: str = ""
    # Declarative, automatically enforced binary dependencies for the fork
    # interpreter. Preparation installs only missing exact-version packages
    # under a per-venv lock; serving receives ``environment`` only after every
    # package and required file has been verified.
    python_runtime: PythonRuntimeConfig | None = None
    server: ServerConfig = None  # type: ignore[assignment]
    idle: IdleWaitConfig = field(default_factory=IdleWaitConfig)
    nsys: NsysConfig = field(default_factory=NsysConfig)

    # --- workload: req-frontend consumes the same source trace as the simulator ---
    workload: LoadGeneratorConfig = None  # type: ignore[assignment]

    @property
    def captures_expert_load(self) -> bool:
        """Whether this pass will ask the engine for EPLB's expert-load stream.

        That stream is the only per-step view of what the engine actually ran --
        one rank-synchronized record per forward -- so a routing pass takes it
        alongside its own product. vLLM's balancedness log is its only source
        and vLLM refuses EPLB without expert parallelism, so a deployment
        without either cannot produce it.

        Asking and requiring are the same question: a pass that asked for the
        stream and got nothing is a defect, not a deployment that happens to
        have no marginal.
        """
        return (
            self.profile_kind in ROUTING_PROFILE_KINDS
            and self.engine == "vllm"
            and "--enable-expert-parallel" in self.server.extra_args
        )
