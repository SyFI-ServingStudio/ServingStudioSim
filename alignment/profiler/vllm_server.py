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
_ALIGNMENT_API_REQUEST_TIMING_RE = re.compile(r"VibeSimAlignmentApiRequestTiming\s+(\{.*\})\s*$")
_ALIGNMENT_EXPERT_LOAD_RE = re.compile(r"VibeSimAlignmentExpertLoad\s+(\{.*\})\s*$")
_COMPLETION_ENGINE_REQUEST_RE = re.compile(r"^cmpl-(.+)-0$")
_COMPLETION_API_REQUEST_RE = re.compile(r"^cmpl-(.+)$")
_TOKENS_REQUEST_RE = re.compile(r"^generate-tokens-(.+)$")

# Data-parallel provenance recovered from vLLM's multiproc log prefixes rather
# than from the record body. Every DP rank runs its own EngineCore with its own
# independent scheduler and its own iteration numbering, so a bare iteration
# index is ambiguous once `dp_size > 1` — a rank tag is mandatory to join a
# measured batch shape to the device that executed it. Recovering it from the
# prefix (instead of adding a record field) keeps one code path that works on
# every capture ever taken, including the ones that predate this change.
_ENGINE_CORE_PREFIX_RE = re.compile(r"^\(EngineCore(?:_DP(\d+))?\s+pid=(\d+)\)")
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
        # TraceLab's mandatory prefix-cache preflight reads
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
        "argv": launch_argv,
        "server_argv": server_argv,
        "env": {k: env[k] for k in keep if k in env},
        "server_config": cfg.server.__dict__,
        "nsys_config": cfg.nsys.__dict__,
        "nsys_profiler": profiler_provenance,
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


def extract_metrics_jsonl(server_log: Path, out_jsonl: Path, *, dp_size: int = 1) -> int:
    """Extract canonical structured iteration records into a metrics JSONL.

    The fork emits one `VibeSimAlignmentIteration {json}` line per model step.
    Exact prefill/decode shapes are retained for the typed timing-predict input
    adapter; `nsys_parse` also uses `prefill_tokens` for stage tagging.

    Each row is stamped with the `dp_rank` of the EngineCore that emitted it.
    Under data parallelism every rank schedules an independent batch and numbers
    its own iterations, so `(dp_rank, iteration_index)` — not `iteration_index`
    alone — is the identity of one measured batch shape. A `dp_size > 1` capture
    therefore *requires* the rank tag and fails without it, while a single
    EngineCore needs no prefix and lands on rank 0.
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
            prefix = _ENGINE_CORE_PREFIX_RE.match(line)
            if prefix is None or prefix.group(1) is None:
                if dp_size > 1:
                    raise ValueError(
                        "alignment iteration record carries no EngineCore_DP<k> log prefix, so "
                        f"its DP rank cannot be established in a dp_size={dp_size} capture: "
                        f"{line[:120]!r}"
                    )
                dp_rank = 0
            else:
                dp_rank = int(prefix.group(1))
            row = json.loads(m.group(1))
            if row.get("dp_rank", dp_rank) != dp_rank:
                raise ValueError(
                    f"alignment iteration record claims dp_rank {row['dp_rank']} but was "
                    f"emitted by EngineCore_DP{dp_rank}"
                )
            row["dp_rank"] = dp_rank
            missing = required - set(row)
            if missing:
                raise ValueError(f"alignment iteration record missing fields {sorted(missing)}")
            if row["schema_version"] not in {1, 2} or row["input_adapter"] != "vllm_text":
                raise ValueError(
                    "unsupported alignment iteration record "
                    f"schema={row['schema_version']!r} adapter={row['input_adapter']!r}"
                )
            if row["schema_version"] == 2:
                timing_fields = {
                    "observed_start_monotonic_ns",
                    "observed_end_monotonic_ns",
                    "observed_elapsed_ms",
                }
                missing_timing = timing_fields - set(row)
                if missing_timing:
                    raise ValueError(
                        "alignment iteration schema-v2 record missing fields "
                        f"{sorted(missing_timing)}"
                    )
                if row["observed_end_monotonic_ns"] < row["observed_start_monotonic_ns"]:
                    raise ValueError("alignment iteration observation ends before it starts")
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

    Schema v1 contains TTFT only. Schema v2 adds first-token → last-token TPOT.
    When the server log also carries API/SSE timing records, this extractor
    joins them by request id and emits schema v3. Durations stay within their
    originating process clock; no cross-process absolute timestamps are mixed.
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
    api_duration_fields = {
        "api_frontend_prepare_ms",
        "api_first_output_wait_ms",
        "api_first_output_serialize_ms",
        "api_token_output_receive_span_ms",
        "api_token_sse_yield_span_ms",
        "api_terminal_tail_ms",
    }
    api_v2_duration_fields = {
        "api_stream_activation_ms",
        "api_add_request_ms",
        "api_collector_wait_ms",
        "api_collector_wakeup_ms",
        "api_generator_resume_ms",
    }
    api_v3_duration_fields = {
        "api_engine_output_wait_ms",
        "api_output_fanout_ms",
    }
    api_required = {
        "schema_version",
        "api_request_id",
        "output_tokens",
        "token_events",
        "first_token_event_tokens",
        "engine_core_ttft_ms",
        "engine_core_decode_ms",
        *api_duration_fields,
    }
    required = {"schema_version", *ttft_duration_fields}
    log_lines = Path(server_log).read_text(errors="replace").splitlines()
    api_rows: dict[str, dict] = {}
    for line in log_lines:
        api_match = _ALIGNMENT_API_REQUEST_TIMING_RE.search(line)
        if not api_match:
            continue
        api_row = json.loads(api_match.group(1))
        missing = api_required - set(api_row)
        if missing:
            raise ValueError(
                f"alignment API request timing record missing fields {sorted(missing)}"
            )
        api_schema_version = api_row["schema_version"]
        if api_schema_version not in {1, 2, 3}:
            raise ValueError(
                f"unsupported alignment API request timing schema {api_schema_version!r}"
            )
        if api_schema_version >= 2:
            missing = api_v2_duration_fields - set(api_row)
            if missing:
                raise ValueError(
                    "alignment API request timing schema v2 record missing fields "
                    f"{sorted(missing)}"
                )
        if api_schema_version == 3:
            missing = api_v3_duration_fields - set(api_row)
            if missing:
                raise ValueError(
                    "alignment API request timing schema v3 record missing fields "
                    f"{sorted(missing)}"
                )
        api_request_id = api_row["api_request_id"]
        if not isinstance(api_request_id, str) or not api_request_id:
            raise ValueError("alignment API request timing api_request_id must be non-empty")
        completion_match = _COMPLETION_API_REQUEST_RE.fullmatch(api_request_id)
        tokens_match = _TOKENS_REQUEST_RE.fullmatch(api_request_id)
        request_id = (
            completion_match.group(1)
            if completion_match is not None
            else tokens_match.group(1)
            if tokens_match is not None
            else api_request_id
        )
        if expected_request_ids is not None and request_id not in expected_request_ids:
            continue
        if request_id in api_rows:
            raise ValueError(f"duplicate alignment API request timing for {request_id!r}")
        duration_fields = set(api_duration_fields)
        if api_schema_version >= 2:
            duration_fields.update(api_v2_duration_fields)
        if api_schema_version == 3:
            duration_fields.update(api_v3_duration_fields)
        for field in {*duration_fields, "engine_core_ttft_ms", "engine_core_decode_ms"}:
            value = api_row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"alignment API request timing {field} must be nonnegative")
        for field in ("output_tokens", "token_events", "first_token_event_tokens"):
            value = api_row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"alignment API request timing {field} must be a positive integer")
        if api_row["token_events"] > api_row["output_tokens"]:
            raise ValueError("alignment API token_events cannot exceed output_tokens")
        if api_row["first_token_event_tokens"] > api_row["output_tokens"]:
            raise ValueError("alignment API first_token_event_tokens cannot exceed output_tokens")
        if api_schema_version >= 2:
            first_output_wait_components_ms = sum(
                api_row[field] for field in api_v2_duration_fields
            )
            if abs(api_row["api_first_output_wait_ms"] - first_output_wait_components_ms) > 1e-6:
                raise ValueError(
                    "alignment API first-output components do not sum to api_first_output_wait_ms"
                )
        if api_schema_version == 3:
            collector_wait_components_ms = sum(api_row[field] for field in api_v3_duration_fields)
            if abs(api_row["api_collector_wait_ms"] - collector_wait_components_ms) > 1e-6:
                raise ValueError(
                    "alignment API collector components do not sum to api_collector_wait_ms"
                )
        api_rows[request_id] = api_row

    api_request_ids = set(api_rows)
    request_ids: set[str] = set()
    n = 0
    with Path(out_jsonl).open("w") as out:
        for line in log_lines:
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
            tokens_match = _TOKENS_REQUEST_RE.fullmatch(engine_request_id)
            request_id = (
                completion_match.group(1)
                if completion_match is not None
                else tokens_match.group(1)
                if tokens_match is not None
                else engine_request_id
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
            api_row = api_rows.pop(request_id, None)
            if api_row is not None:
                if schema_version != 2:
                    raise ValueError(
                        "alignment API timing requires EngineCore request timing schema v2"
                    )
                if api_row["output_tokens"] != row["num_output_tokens"]:
                    raise ValueError(
                        "alignment API output_tokens does not match EngineCore num_output_tokens"
                    )
                for field in ("engine_core_ttft_ms", "engine_core_decode_ms"):
                    if abs(api_row[field] - row[field]) > 1e-6:
                        raise ValueError(
                            f"alignment API {field} does not match EngineCore timing record"
                        )
                row["schema_version"] = 3
                row["engine_timing_schema_version"] = 2
                row["api_timing_schema_version"] = api_row["schema_version"]
                row["api_request_id"] = api_row["api_request_id"]
                row["api_token_events"] = api_row["token_events"]
                row["api_first_token_event_tokens"] = api_row["first_token_event_tokens"]
                for field in api_duration_fields:
                    row[field] = api_row[field]
                if api_row["schema_version"] >= 2:
                    for field in api_v2_duration_fields:
                        row[field] = api_row[field]
                if api_row["schema_version"] == 3:
                    for field in api_v3_duration_fields:
                        row[field] = api_row[field]
            out.write(json.dumps(row) + "\n")
            n += 1
    if api_rows:
        raise ValueError(
            "alignment API request timings have no matching EngineCore records: "
            f"{sorted(api_rows)[:8]!r}"
        )
    if api_request_ids and api_request_ids != request_ids:
        missing = sorted(request_ids - api_request_ids)
        raise ValueError(
            "alignment API request timing ids do not cover EngineCore timing ids: "
            f"missing={missing[:8]!r}"
        )
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
