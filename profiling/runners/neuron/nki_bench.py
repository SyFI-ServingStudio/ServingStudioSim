"""Launch-unit microbench runner (worker only): backends ``vllm_neuron_nki`` and
``vllm_neuron_torch``.

Each row times one launch unit of the vLLM Neuron Llama 3.1 8B BF16 TP4
forward (``nki_units``) alone on the one reserved LNC2 logical core, inside
the vLLM Neuron 0.29 image (Neuron SDK 2.33). The unit is compiled through the
production path (one Dynamo full graph on the engine's compile backend with
the serving compiler flags), warmed, then executed ``ITERS`` times under the
NRT system profile, started through the same runtime binding the engine's
``/start_profile`` uses (``torch.classes.neuron.Runtime().start_profiling``).
The device time of one call is the union of the ``nc_exec_running`` spans of
the core's two physical cores. The row is the median call, raw. It includes
the fixed cost one standalone execution pays (the empty-graph floor). The
floor is measured once per batch and recorded in the row's provenance; it is
not subtracted. Small units run below the floor (the decode glue at B1), and
inside the engine's one graph the units do not each pay it, so what to charge
per launch is the composition's decision.

The two decode leaves of the default decode granularity
(``neuron_decoder_layer``, ``neuron_decode_fixed``) are validated here and
measured by ``forward_bench``: a reduced-depth TP4 decode forward on the whole
chip, not one unit alone.

The trace is exported with ``neuron-explorer`` as JSON, the format SDK 2.33
(the pinned image) still writes; SDK 2.34 exports Parquet only, so the next
image needs a Parquet reader behind ``export_trace`` and ``load_events``
(``neuron-explorer view --output-format parquet``, a folder of device tables).

The runner never chooses cores: the worker's reservation sets
``NEURON_RT_VISIBLE_CORES`` and ``NEURON_LOGICAL_NC_CONFIG``. Each batch leaves
its profiles and exports under ``<cache>/nki-runs/<uuid>/``; nothing prunes them.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

from profiling.db.args import DType
from profiling.runners.metrics import ComputeMetrics, RunnerResult
from profiling.runners.neuron.stock_contract import (
    BLOCK_SIZE,
    CONTEXTS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE_PER_RANK,
    KV_BLOCKS,
    KV_HEADS_PER_RANK,
    Q_HEADS_PER_RANK,
    QKV_OUT_PER_RANK,
    TP_SIZE,
    VOCAB_PER_RANK,
    largest_decode_bucket,
)

BYTES = 2  # BF16

# Which launch unit each kind measures in each phase. A kind absent from a phase
# is not measured there: in decode one attention call (projections included)
# does the work of the three prefill-only kinds, and decode attention has no
# prefill form.
UNITS = {
    "neuron_nki_qkv": {"prefill": "prefill.qkv_proj"},
    "neuron_nki_attention_cte": {"prefill": "prefill.flash_attention"},
    "neuron_nki_output_projection": {"prefill": "prefill.o_proj"},
    "neuron_nki_mlp": {"prefill": "prefill.mlp", "decode": "decode.mlp"},
    "neuron_attention_decode": {"decode": "decode.attention_nki"},
    # validated here, measured by forward_bench (a TP4 reduced-depth forward)
    "neuron_decoder_layer": {"decode": "decode.forward_nki"},
    "neuron_decode_fixed": {"decode": "decode.forward_nki"},
    "neuron_glue": {"prefill": "prefill.glue", "decode": "decode.glue"},
    "neuron_head": {"prefill": "prefill.head", "decode": "decode.head"},
}
# The second backend (`vllm_neuron_torch`) of neuron_attention_decode measures
# the stock call's torch fallback with the same rows and rules. The two decode
# leaves take their torch path in forward_bench.
TORCH_UNITS = {"neuron_attention_decode": {"decode": "decode.attention_torch"}}
# The kinds whose rows depend on the context (they read a block table over it).
CONTEXT_KINDS = ("neuron_attention_decode", "neuron_decoder_layer", "neuron_decode_fixed")
# The shape-only kinds read no block table, so no context of their own caps
# their decode buckets: the ladder's largest bucket does.
SHAPE_ONLY_KINDS = ("neuron_nki_mlp", "neuron_glue", "neuron_head")
# C512: the ladder context whose decode cap is the ladder's largest bucket. The
# shape-only kinds and the floor are built at it.
SHAPE_ONLY_CONTEXT = CONTEXTS[1]
ITERS = 50
WARM = 10
EXPLORER = "/opt/aws/neuron/bin/neuron-explorer"
EXPORT_TIMEOUT_S = 600
_PROVENANCE: dict[str, str] = {}


def key(spec):
    """The provenance map key of one row spec: its canonical JSON."""
    return json.dumps(spec, sort_keys=True, default=str)


def largest_ladder_decode_bucket():
    """The largest decode bucket any context of the ladder compiles."""
    return max(largest_decode_bucket(c) for c in CONTEXTS)


def validate_shape(kind, spec):
    """The row's shape must be one the stock contract compiles.

    Prefill token buckets are contexts of the ladder. Decode buckets are powers
    of two. The context's KV-resident capacity caps decode attention. The
    ladder's largest bucket caps the shape-only kinds. The identity fields pin
    BF16 and the stock TP4 pool. Returns ``(phase, tokens, ctx)``.
    """
    phases = UNITS[kind]
    phase = spec.get("phase", next(iter(phases)))
    if phase not in phases:
        raise ValueError(f"{kind} is not measured in the {phase} phase")
    if DType.from_value(spec["dtype"]) != DType.BF16:
        raise ValueError("standalone NKI leaves require BF16")
    if spec["tp_size"] != TP_SIZE:
        raise ValueError(f"standalone NKI leaves are measured at TP{TP_SIZE}")
    bucket = spec["token_bucket"]
    if phase == "prefill":
        if bucket not in CONTEXTS:
            raise ValueError(f"prefill token bucket must be one of {CONTEXTS}")
        return phase, bucket, bucket
    if bucket < 1 or bucket & (bucket - 1):
        raise ValueError("decode buckets are powers of two")
    if kind in CONTEXT_KINDS:
        if (spec["kv_blocks"], spec["block_size"]) != (KV_BLOCKS, BLOCK_SIZE):
            raise ValueError(f"stock pool is {KV_BLOCKS} blocks of {BLOCK_SIZE} tokens")
        ctx = spec["max_model_len"]
        if ctx not in CONTEXTS:
            raise ValueError(f"max_model_len must be one of {CONTEXTS}")
        if bucket > largest_decode_bucket(ctx):
            raise ValueError("decode bucket exceeds the context's KV-resident capacity")
        return phase, bucket, ctx
    if bucket > largest_ladder_decode_bucket():
        raise ValueError("decode bucket exceeds the ladder's largest bucket")
    return phase, bucket, SHAPE_ONLY_CONTEXT


def estimate_work(kind, phase, tokens, ctx):
    """(flops, bytes) of one unit per rank: matmul FLOPs and the operands moved
    once. Estimates for the rate columns, not device counters."""
    n, h, dh, nq, nkv, inter = (
        tokens,
        HIDDEN,
        HEAD_DIM,
        Q_HEADS_PER_RANK,
        KV_HEADS_PER_RANK,
        INTERMEDIATE_PER_RANK,
    )
    qkv_out = QKV_OUT_PER_RANK
    if kind == "neuron_nki_qkv":
        return 2 * n * h * qkv_out, BYTES * (n * h + h * qkv_out + n * qkv_out)
    if kind == "neuron_nki_attention_cte":
        # causal: half of the QK^T and PV products are computed
        return 2 * nq * n * n * dh, BYTES * 4 * nq * n * dh
    if kind == "neuron_nki_output_projection":
        return 2 * n * nq * dh * h, BYTES * (n * nq * dh + nq * dh * h + n * h)
    if kind == "neuron_nki_mlp":
        return 6 * n * h * inter, BYTES * (2 * n * h + 3 * h * inter)
    if kind == "neuron_attention_decode":
        flops = 2 * n * h * qkv_out + 4 * n * nq * ctx * dh + 2 * n * nq * dh * h
        kv = 2 * n * ctx * nkv * dh
        return flops, BYTES * (h * qkv_out + nq * dh * h + kv + 2 * n * h)
    if kind == "neuron_decoder_layer":
        attn_flops, attn_bytes = estimate_work("neuron_attention_decode", phase, tokens, ctx)
        mlp_flops, mlp_bytes = estimate_work("neuron_nki_mlp", phase, tokens, ctx)
        # the two norms and two residuals read and write the activation
        return attn_flops + mlp_flops, attn_bytes + mlp_bytes + BYTES * 6 * n * h
    if kind == "neuron_glue":
        kv_write = 2 * n * nkv * dh if phase == "prefill" else 0
        return 0, BYTES * (6 * n * h + kv_write)
    if kind == "neuron_head":
        sampled = 1 if phase == "prefill" else n
        flops = 2 * sampled * h * VOCAB_PER_RANK
        return flops, BYTES * (h * VOCAB_PER_RANK + n * h + sampled * VOCAB_PER_RANK)
    raise ValueError(kind)


def union_duration(spans):
    """Total length of the union of ``(start, stop)`` intervals."""
    merged = []
    for start, stop in sorted(spans):
        if stop <= start:
            raise ValueError("native execution interval must be positive")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return sum(stop - start for start, stop in merged)


def exec_times_ms(events):
    """Device time per execution, in ms, from ``nc_exec_running`` event dicts
    (keys ``exec_id``, ``timestamp``, ``duration``, ``device_core_idx``,
    optionally ``model_name`` and ``name``). For each execution, the spans of
    both physical cores merge into one union; an execution that did not run on
    exactly two distinct physical cores is refused (the LNC2 core is two cores,
    and one span would undercount it)."""
    by_exec = defaultdict(list)
    for e in events:
        if e.get("name", "nc_exec_running") != "nc_exec_running":
            continue
        by_exec[(e.get("model_name"), e["exec_id"])].append(
            (e["device_core_idx"], e["timestamp"], e["timestamp"] + e["duration"])
        )
    out = []
    for exec_key, spans in by_exec.items():
        cores = {core for core, _, _ in spans}
        if len(spans) != 2 or len(cores) != 2:
            raise ValueError(
                f"execution {exec_key[1]} ran on cores {sorted(cores)} with {len(spans)} spans; "
                "expected one span on each of the two physical cores"
            )
        out.append(union_duration([(s, e) for _, s, e in spans]) / 1e6)  # ns -> ms
    if not out:
        raise ValueError("NRT trace has no device executions")
    return out


def load_events(export):
    """The ``nc_exec_running`` events of one JSON export (its ``trace_event`` list)."""
    return json.loads(Path(export).read_text())["trace_event"]


def session_dir(prof_dir):
    """The one profile session written under ``prof_dir``: the exporter's
    ``<prof_dir>/<instance>_pid_<n>/<session>/`` for a runtime capture, or the
    directory itself when it holds ``ntrace.pb`` directly."""
    prof_dir = Path(prof_dir)
    if (prof_dir / "ntrace.pb").exists():
        return prof_dir
    sessions = sorted(p for p in prof_dir.glob("*/*") if p.is_dir())
    if len(sessions) != 1:
        raise ValueError(f"expected one profile session under {prof_dir}, found {len(sessions)}")
    return sessions[0]


def export_trace(prof_dir):
    """neuron-explorer export of one unit's system profile (worker side)."""
    prof_dir = Path(prof_dir)
    session = session_dir(prof_dir)
    output = prof_dir.parent / (prof_dir.name + ".json")
    command = [
        EXPLORER,
        "view",
        "-d",
        str(session),
        "--output-format",
        "json",
        "--ignore-device-profile",
        "--system-trace-filter-event-type",
        "nc_exec_running",
        "--output-file",
        str(output),
        "--disable-ui",
        "--force",
    ]
    with (prof_dir.parent / (prof_dir.name + ".export.log")).open("w") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=EXPORT_TIMEOUT_S
        )
    return output


@contextlib.contextmanager
def runtime_profile(prof_dir):
    """Capture the NRT system trace of the enclosed calls into ``prof_dir``
    (``system_profile`` only: ``nc_exec_running`` per graph execution, no
    instruction-level device trace). The binding and its arguments are those
    of the engine's own profiler wrapper (``vllm_neuron.vllm.worker.
    neuron_profiler``): output directory, activities, no core filter, the NRT
    default event budget, and no NEFF copy (the trace alone times the unit).
    Every unit call synchronizes on the host, so nothing is in flight at stop.
    """
    import torch

    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(prof_dir), ["system_profile"], None, None, None)
    try:
        yield
    finally:
        runtime.stop_profiling()


def measure_unit(unit, tokens, ctx, root):
    """Compile, warm and profile one unit; return its timing record."""
    from profiling.runners.neuron import nki_units

    module, args = nki_units.build(unit, tokens, ctx)
    fn = nki_units.compile_unit(module)
    started = time.perf_counter()
    nki_units.sync(fn(*args))
    compile_s = time.perf_counter() - started
    for _ in range(WARM):
        nki_units.sync(fn(*args))
    # unit names carry a dot; keep the directory name suffix-free
    prof_dir = root / f"prof_{unit.replace('.', '_')}_{tokens}_{ctx}"
    with runtime_profile(prof_dir):
        for _ in range(ITERS):
            nki_units.sync(fn(*args))
    times = sorted(exec_times_ms(load_events(export_trace(prof_dir))))
    if len(times) != ITERS:
        raise ValueError(f"{unit}: {len(times)} device executions for {ITERS} profiled calls")
    return {
        "median_ms": statistics.median(times),
        "min_ms": times[0],
        "p90_ms": times[int(0.9 * len(times)) - 1],
        "n": len(times),
        "compile_s": round(compile_s, 1),
    }


def _runs_root():
    cache = Path(os.environ["SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"])
    root = cache / "nki-runs" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def profile_batch(kind, kwargs_list, units=UNITS):
    """Return one result per spec, in order. A bad shape or a unit that fails
    to compile is that row's error. The floor is measured once per batch; if
    it fails, every row carries that error. ``units`` names the launch unit
    per phase (the kind's backend)."""
    shapes = []
    for spec in kwargs_list:
        try:
            shapes.append(validate_shape(kind, spec))
        except ValueError as error:
            shapes.append(error)
    if all(isinstance(shape, ValueError) for shape in shapes):
        return [RunnerResult(error=str(shape)) for shape in shapes]
    try:
        root = _runs_root()
        floor = measure_unit("noop", 1, SHAPE_ONLY_CONTEXT, root)
    except Exception as error:
        message = f"floor: {type(error).__name__}: {error}"
        return [RunnerResult(error=message) for _ in kwargs_list]
    results = []
    for spec, shape in zip(kwargs_list, shapes, strict=True):
        if isinstance(shape, ValueError):
            results.append(RunnerResult(error=str(shape)))
            continue
        phase, tokens, ctx = shape
        try:
            rec = measure_unit(units[kind][phase], tokens, ctx, root)
        except Exception as error:
            results.append(RunnerResult(error=f"{type(error).__name__}: {error}"))
            continue
        time_ms = rec["median_ms"]
        flops, moved = estimate_work(kind, phase, tokens, ctx)
        results.append(
            RunnerResult(
                metrics=ComputeMetrics(
                    time_ms=time_ms,
                    tflops=flops / (time_ms * 1e9),
                    memory_bandwidth_gbps=moved / (time_ms * 1e6),
                )
            )
        )
        _PROVENANCE[key(spec)] = (
            f"nki_bench median_ms={rec['median_ms']:.4f} min_ms={rec['min_ms']:.4f} "
            f"p90_ms={rec['p90_ms']:.4f} n={rec['n']} compile_s={rec['compile_s']} "
            f"floor_ms={floor['median_ms']:.4f}; artifacts={root}"
        )
    return results


def row_provenance(**spec):
    """The provenance note of a row this process measured, or None."""
    return _PROVENANCE.get(key(spec))


def profile_neuron_nki_qkv_batch(kwargs_list):
    """One result per spec for kind ``neuron_nki_qkv``; see ``profile_batch``."""
    return profile_batch("neuron_nki_qkv", kwargs_list)


def profile_neuron_nki_attention_cte_batch(kwargs_list):
    """One result per spec for kind ``neuron_nki_attention_cte``; see ``profile_batch``."""
    return profile_batch("neuron_nki_attention_cte", kwargs_list)


def profile_neuron_nki_output_projection_batch(kwargs_list):
    """One result per spec for kind ``neuron_nki_output_projection``; see ``profile_batch``."""
    return profile_batch("neuron_nki_output_projection", kwargs_list)


def profile_neuron_nki_mlp_batch(kwargs_list):
    """One result per spec for kind ``neuron_nki_mlp``; see ``profile_batch``."""
    return profile_batch("neuron_nki_mlp", kwargs_list)


def profile_neuron_attention_decode_batch(kwargs_list):
    """One result per spec for kind ``neuron_attention_decode`` on the NKI kernel path."""
    return profile_batch("neuron_attention_decode", kwargs_list)


def profile_neuron_attention_decode_torch_batch(kwargs_list):
    """``neuron_attention_decode`` on the stock call's torch fallback."""
    return profile_batch("neuron_attention_decode", kwargs_list, units=TORCH_UNITS)


def profile_neuron_glue_batch(kwargs_list):
    """One result per spec for kind ``neuron_glue``; see ``profile_batch``."""
    return profile_batch("neuron_glue", kwargs_list)


def profile_neuron_head_batch(kwargs_list):
    """One result per spec for kind ``neuron_head``; see ``profile_batch``."""
    return profile_batch("neuron_head", kwargs_list)
