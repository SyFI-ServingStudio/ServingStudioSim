"""Optimality floors — segmented and scope-fused necessary work, per level.

Called by the Rust `optimality` analyzer subject as a subprocess:

    uv run python -m model.work.floors <log_dir>

For unlocked analysis, Rust aggregates each level's workload (reusing the
`workload-conservation` subject's `groups` parser) and pipes it in on **stdin** as::

    {"levels": {"<level_key>": {matmul_tokens, prefill_tokens, decode_passes,
                                prefill_pairs, prefill_cached, decode_kv,
                                prefill_requests}, ...}}

where a public level_key is ``cluster`` | ``<pool_tag>`` | ``<pool_tag>/<worker_id>``
(exactly the `levels.rs` level keys). The analyzer may include private
``<pool_tag>/__saturated_worker__/<worker_id>`` rows in the same batched request;
the first path component still selects the model spec. This
script is the *model-aware* half: it never touches the cost_log / parquet. For
each level it loads that pool's model, assembles ONE aggregate ``Workload`` (the
whole run's tokens as a single mega-forward — weights counted once, the loosest
scope-fused floor), and emits the two roofline floors in **GPU·seconds** on stdout::

    {"levels": {"<level_key>": {"necessary": s, "segmented": s}, ...},
     "meta": {...}}

For batch-locked run analysis, Rust instead sends deduplicated iteration shapes::

    {"locked_compositions": {"<worker_key>": [
        {"occurrences": 3, "totals": {...}}, ...
    ]}}

Each fixed-batch roofline is evaluated before occurrence weighting. Dense affine
bases vectorize the common path; every basis is independently checked against one
direct ``model.label`` result, with a per-shape direct fallback on mismatch. The
worker response includes ``composition`` counters so Rust can validate the
shape/iteration contract and expose whether any fallback was required.

An independent row that cannot be labeled is returned as ``{"error": ...}``;
other rows in the same batch remain available.

- **necessary** = the stable wire name for the scope-fused roofline
  ``max(ΣFLOPs/peak, Σbytes/bw)`` (`roofline_ms`).
- **segmented** = ``Σ_seg max(compute, memory)`` (`segmented_lower_bound_ms`).

`num_gpus=1`: the labeler computes the FULL / unsharded ``F_min``, so
``total_work / peak`` is already the ×G GPU·seconds the Rust R5 rung is in.

The attention geometry is passed as pre-summed scalars, so one prefill and one
decode ``mask="full"`` interaction (``pairs() = q·k`` with ``q=1``,
``k=Σpairs``) reproduce the GQA work exactly without a 1.7-billion-element list.
``attention_step_count`` separately preserves the original state-transaction
count required by recurrent linear attention.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from .core import (
    AttnInteraction,
    Workload,
    gpu_mem_bandwidth_gbps,
    gpu_peak_tflops,
)
from .registry import load_model


@cache
def _model(config_path: str):
    """Load a model once per config path (levels within a pool share it)."""
    return load_model(config_path)


def _pool_specs(log_dir: Path) -> dict[str, dict]:
    """Map pool_tag -> {config, gpu, dtype} from the run's params.json.

    One group per pool assumed (PD/unified dense today); multi-group EP/HP needs
    the same per-group revisit as the rest of the analyzer.

    ``dtype`` is only the fallback for segments that make no precision claim; the
    real per-segment precisions come from the model config's own
    ``quantization_config``. The two must agree — see :func:`_check_precision`.
    """
    params = json.loads((log_dir / "raw" / "params.json").read_text())
    specs: dict[str, dict] = {}
    for pool_tag, pool in params.get("pools", {}).items():
        group = pool["groups"][0]
        arch = group["arch"]
        specs[pool_tag] = {
            "config": arch["model_config"],
            # `gpu` is a group-level field (the arch block carries model/tp/fp8).
            "gpu": group["gpu"],
            "arch_fp8": bool(arch.get("fp8")),
            "dtype": "fp8" if arch.get("fp8") else "bf16",
        }
    return specs


def _check_precision(model, spec: dict) -> None:
    """Refuse to label a run whose arch precision contradicts its model config.

    A run served from an FP8 checkpoint but labeled against the BF16 config
    reports weight bytes at 2x the traffic that actually moved — a "minimum"
    larger than the measured value, which silently turns redundancy into a number
    below 1. It is not detectable downstream, so it is rejected here.
    """
    config_is_quantized = model.quant is not None
    if spec["arch_fp8"] and not config_is_quantized:
        raise ValueError(
            f"arch declares fp8 but {spec['config']} has no quantization_config; "
            "point `model_config` at the checkpoint's FP8 config (e.g. glm52_fp8.json)"
        )
    if config_is_quantized and not spec["arch_fp8"]:
        raise ValueError(
            f"{spec['config']} declares a {model.quant.compute_dtype} quantization_config "
            "but the arch does not set fp8; the run and the accountant disagree on precision"
        )


def _spec_for_level(level_key: str, pool_specs: dict[str, dict]) -> dict:
    """Which (config, gpu, dtype) a level computes under.

    ``<pool_tag>`` / ``<pool_tag>/<worker_id>`` take that pool's spec; ``cluster``
    spans every pool, so it requires them to share one model (true for PD single
    -model runs) — otherwise a per-pool-summed cluster floor is needed (deferred).
    """
    if level_key == "cluster":
        distinct = {(s["config"], s["gpu"], s["dtype"]) for s in pool_specs.values()}
        if len(distinct) != 1:
            raise ValueError(f"cluster floor needs one shared model across pools, saw {distinct}")
        return next(iter(pool_specs.values()))
    # Pool level is the bare tag; worker level is "<pool_tag>/<worker_id>".
    pool_tag = level_key.split("/", 1)[0]
    return pool_specs[pool_tag]


def _aggregate_workload(totals: dict) -> Workload:
    """Assemble one giant-batch Workload from a level's pre-summed scalars.

    Prefill and decode attention each collapse to a single ``mask="full"``
    interaction carrying the summed pair / cached-KV counts (identity, since
    ``pairs()`` for full = ``q·k`` and we set ``q=1``). Decode reads all context
    keys from cache each step, so cached == pairs == ``decode_kv``; the self-key
    ``+1`` per step is negligible and omitted.
    """
    matmul_tokens = int(totals["matmul_tokens"])
    sampled = int(totals["decode_passes"]) + int(totals["prefill_requests"])
    attn: list[AttnInteraction] = []
    prefill_pairs = int(totals["prefill_pairs"])
    if prefill_pairs > 0:
        attn.append(
            AttnInteraction(
                1,
                prefill_pairs,
                int(totals["prefill_cached"]),
                "full",
                phase="prefill",
            )
        )
    decode_kv = int(totals["decode_kv"])
    if decode_kv > 0:
        attn.append(AttnInteraction(1, decode_kv, decode_kv, "full", phase="decode"))
    prefill_tokens = int(totals["prefill_tokens"])
    decode_passes = int(totals["decode_passes"])
    prefill_requests = int(totals["prefill_requests"])
    return Workload(
        matmul_tokens=matmul_tokens,
        head_positions=sampled,
        attn=attn,
        attention_step_count=sampled,
        attention_step_count_by_phase={
            "prefill": prefill_requests,
            "decode": decode_passes,
        },
        attention_tokens_by_phase={
            "prefill": prefill_tokens,
            "decode": decode_passes,
        },
    )


def _peak_resolver(spec: dict):
    """Memoized dtype -> peak TFLOP/s for this spec's GPU.

    ``spec["dtype"]`` is only the fallback for a segment that makes no precision
    claim; a mixed-precision checkpoint names its own dtype per segment.
    """
    peak_by_dtype: dict[str, float] = {}

    def peak(compute_dtype: str | None) -> float:
        key = compute_dtype or spec["dtype"]
        if key not in peak_by_dtype:
            peak_by_dtype[key] = gpu_peak_tflops(spec["gpu"], key)
        return peak_by_dtype[key]

    return peak


def _segment_dtypes(model, totals: dict, default_dtype: str) -> dict[str, str]:
    """Segment name -> compute dtype. Independent of workload size, but the set of
    segment names is not, so it is resolved for the same totals as the work."""
    label = model.label(_aggregate_workload(totals))
    return {
        segment.name: segment.compute_dtype or default_dtype for segment in label.segments
    }


def _label_payload(model, totals: dict, spec: dict) -> dict:
    """Serialize one workload label, including each semantic's own roofline."""
    peak = _peak_resolver(spec)
    bandwidth_gbps = gpu_mem_bandwidth_gbps(spec["gpu"])
    label = model.label(_aggregate_workload(totals))
    segment_work = {
        segment.name: (segment.flops_total, segment.bytes_total) for segment in label.segments
    }
    segment_dtypes = {
        segment.name: segment.compute_dtype or spec["dtype"] for segment in label.segments
    }
    return _payload_from_segment_work(segment_work, segment_dtypes, peak, bandwidth_gbps)


def _payload_from_segment_work(
    segment_work: dict[str, tuple[float, float]],
    segment_dtypes: dict[str, str],
    peak,
    bandwidth_gbps: float,
) -> dict:
    segments = []
    total_compute_seconds = 0.0
    total_bytes = 0.0
    for name in sorted(segment_work):
        flops, bytes_ = segment_work[name]
        compute_dtype = segment_dtypes[name]
        compute_seconds = flops / (peak(compute_dtype) * 1e12)
        memory_seconds = bytes_ / (bandwidth_gbps * 1e9)
        segments.append(
            {
                "name": name,
                "flops": flops,
                "bytes": bytes_,
                "compute_dtype": compute_dtype,
                "necessary": max(compute_seconds, memory_seconds),
            }
        )
        # Fusing every segment into one kernel still cannot fuse across precisions,
        # so the fused compute term sums per-segment seconds instead of dividing one
        # global FLOP total by one peak.
        total_compute_seconds += compute_seconds
        total_bytes += bytes_
    memory_seconds = total_bytes / (bandwidth_gbps * 1e9)
    return {
        "necessary": max(total_compute_seconds, memory_seconds),
        "segmented": sum(segment["necessary"] for segment in segments),
        "segments": segments,
    }


_WORKLOAD_FIELDS = (
    "matmul_tokens",
    "prefill_tokens",
    "decode_passes",
    "prefill_pairs",
    "prefill_cached",
    "decode_kv",
    "prefill_requests",
)
_LARGE_GEOMETRY_FIELDS = {"prefill_pairs", "prefill_cached", "decode_kv"}
_GEOMETRY_PROBE = 1_048_576


def _segment_work(model, totals: dict) -> dict[str, tuple[float, float]]:
    label = model.label(_aggregate_workload(totals))
    return {segment.name: (segment.flops_total, segment.bytes_total) for segment in label.segments}


def _subtract_segment_work(
    left: dict[str, tuple[float, float]], right: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    return {
        name: (
            (left.get(name) or (0.0, 0.0))[0] - (right.get(name) or (0.0, 0.0))[0],
            (left.get(name) or (0.0, 0.0))[1] - (right.get(name) or (0.0, 0.0))[1],
        )
        for name in left.keys() | right.keys()
    }


def _has_routed_matmul(model) -> bool:
    return any(
        group.routed
        for stack in model.layers
        for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups())
    )


def _basis_key(model, totals: dict, has_routed_matmul: bool) -> tuple[int | None, bool]:
    matmul_tokens = int(totals["matmul_tokens"])
    return (
        matmul_tokens if has_routed_matmul else None,
        matmul_tokens >= model.vocab,
    )


def _workload_basis(
    model, totals: dict, has_routed_matmul: bool
) -> dict[str, dict[str, tuple[float, float]] | dict[str, int]]:
    """Affine semantic-work basis, with nonlinear dimensions pinned in the key.

    Current model specs are affine in the seven compressed workload scalars except
    routed-expert weight loading and the embedding-table cap. Routed token counts are
    therefore pinned exactly; dense token counts use one basis on each side of the
    vocab cap. The first shape using a basis is independently checked below.
    """
    base_totals = {field_name: 0 for field_name in _WORKLOAD_FIELDS}
    matmul_tokens = int(totals["matmul_tokens"])
    if has_routed_matmul:
        base_totals["matmul_tokens"] = matmul_tokens
    elif matmul_tokens >= model.vocab:
        base_totals["matmul_tokens"] = model.vocab
    base = _segment_work(model, base_totals)
    coefficients = {}
    for field_name in _WORKLOAD_FIELDS:
        if has_routed_matmul and field_name == "matmul_tokens":
            continue
        probe = _GEOMETRY_PROBE if field_name in _LARGE_GEOMETRY_FIELDS else 1
        probe_totals = dict(base_totals)
        probe_totals[field_name] += probe
        if field_name == "prefill_cached":
            # `_aggregate_workload` materializes prefill attention only when pairs>0.
            probe_totals["prefill_pairs"] = _GEOMETRY_PROBE
            reference_totals = dict(base_totals, prefill_pairs=_GEOMETRY_PROBE)
            reference = _segment_work(model, reference_totals)
        else:
            reference = base
        delta = _subtract_segment_work(_segment_work(model, probe_totals), reference)
        coefficients[field_name] = {
            name: (flops / probe, bytes_ / probe) for name, (flops, bytes_) in delta.items()
        }
    return {"origin": base_totals, "base": base, "coefficients": coefficients}


def _reconstruct_segment_work(basis: dict, totals: dict) -> dict[str, tuple[float, float]]:
    coefficients = basis["coefficients"]
    segment_names = set(basis["base"])
    for coefficient in coefficients.values():
        segment_names.update(coefficient)
    segment_work = {}
    for name in segment_names:
        flops, bytes_ = basis["base"].get(name, (0.0, 0.0))
        for field_name, coefficient in coefficients.items():
            coefficient_flops, coefficient_bytes = coefficient.get(name, (0.0, 0.0))
            field_delta = int(totals[field_name]) - basis["origin"][field_name]
            flops += coefficient_flops * field_delta
            bytes_ += coefficient_bytes * field_delta
        segment_work[name] = (flops, bytes_)
    return segment_work


def _segment_work_matches(
    reconstructed: dict[str, tuple[float, float]], direct: dict[str, tuple[float, float]]
) -> bool:
    if reconstructed.keys() != direct.keys():
        return False
    for name, direct_values in direct.items():
        for reconstructed_value, direct_value in zip(
            reconstructed[name], direct_values, strict=True
        ):
            tolerance = max(abs(direct_value), 1.0) * 1e-9
            if abs(reconstructed_value - direct_value) > tolerance:
                return False
    return True


def _validated_basis(
    model,
    totals: dict,
    has_routed_matmul: bool,
    basis_cache: dict[tuple[int | None, bool], dict | None],
    default_dtype: str,
) -> dict | None:
    """Build and independently validate one basis before any batch reduction."""
    basis_key = _basis_key(model, totals, has_routed_matmul)
    if basis_key not in basis_cache:
        candidate = _workload_basis(model, totals, has_routed_matmul)
        reconstructed = _reconstruct_segment_work(candidate, totals)
        direct = _segment_work(model, totals)
        if _segment_work_matches(reconstructed, direct):
            # Validation just proved the basis spans exactly the direct label's
            # segment names, so one label at these totals resolves every dtype.
            candidate["dtypes"] = _segment_dtypes(model, totals, default_dtype)
            basis_cache[basis_key] = candidate
        else:
            basis_cache[basis_key] = None
    return basis_cache[basis_key]


def _reduce_affine_group(
    basis: dict,
    weighted_shapes: list[dict],
    peak,
    bandwidth_gbps: float,
) -> tuple[float, float, dict[str, dict]]:
    """Evaluate all shapes sharing one basis as dense array operations."""
    coefficients = basis["coefficients"]
    field_names = tuple(coefficients)
    segment_names = set(basis["base"])
    for coefficient in coefficients.values():
        segment_names.update(coefficient)
    segment_names = tuple(sorted(segment_names))

    base_flops = np.asarray(
        [basis["base"].get(name, (0.0, 0.0))[0] for name in segment_names], dtype=np.float64
    )
    base_bytes = np.asarray(
        [basis["base"].get(name, (0.0, 0.0))[1] for name in segment_names], dtype=np.float64
    )
    coefficient_flops = np.asarray(
        [
            [coefficients[field_name].get(name, (0.0, 0.0))[0] for name in segment_names]
            for field_name in field_names
        ],
        dtype=np.float64,
    )
    coefficient_bytes = np.asarray(
        [
            [coefficients[field_name].get(name, (0.0, 0.0))[1] for name in segment_names]
            for field_name in field_names
        ],
        dtype=np.float64,
    )
    workload_deltas = np.asarray(
        [
            [
                int(weighted_shape["totals"][field_name]) - basis["origin"][field_name]
                for field_name in field_names
            ]
            for weighted_shape in weighted_shapes
        ],
        dtype=np.float64,
    )
    occurrences = np.asarray(
        [int(weighted_shape["occurrences"]) for weighted_shape in weighted_shapes],
        dtype=np.float64,
    )
    # One peak per segment: a mixed-precision checkpoint runs some rows on the FP8
    # tensor cores and others (router, sparse MLA) at the master dtype.
    segment_peaks = np.asarray(
        [peak(basis["dtypes"][name]) for name in segment_names], dtype=np.float64
    )

    segment_flops = workload_deltas @ coefficient_flops + base_flops
    segment_bytes = workload_deltas @ coefficient_bytes + base_bytes
    compute_seconds = segment_flops / (segment_peaks * 1e12)
    memory_seconds = segment_bytes / (bandwidth_gbps * 1e9)
    segment_necessary = np.maximum(compute_seconds, memory_seconds)

    fused_seconds = float(
        occurrences
        @ np.maximum(
            compute_seconds.sum(axis=1),
            segment_bytes.sum(axis=1) / (bandwidth_gbps * 1e9),
        )
    )
    segmented_seconds = float(occurrences @ segment_necessary.sum(axis=1))
    aggregate_flops = occurrences @ segment_flops
    aggregate_bytes = occurrences @ segment_bytes
    aggregate_necessary = occurrences @ segment_necessary
    segments = {
        name: {
            "name": name,
            "flops": float(aggregate_flops[index]),
            "bytes": float(aggregate_bytes[index]),
            "compute_dtype": basis["dtypes"][name],
            "necessary": float(aggregate_necessary[index]),
        }
        for index, name in enumerate(segment_names)
    }
    return fused_seconds, segmented_seconds, segments


def _reduce_direct_group(
    model, weighted_shapes: list[dict], spec: dict
) -> tuple[float, float, dict[str, dict]]:
    """Correct fallback for a future model whose work is not affine in a basis."""
    fused_seconds = 0.0
    segmented_seconds = 0.0
    segments_by_name: dict[str, dict] = {}
    for weighted_shape in weighted_shapes:
        occurrences = int(weighted_shape["occurrences"])
        payload = _label_payload(model, weighted_shape["totals"], spec)
        fused_seconds += payload["necessary"] * occurrences
        segmented_seconds += payload["segmented"] * occurrences
        for segment in payload["segments"]:
            aggregate = _empty_aggregate(segments_by_name, segment)
            aggregate["flops"] += segment["flops"] * occurrences
            aggregate["bytes"] += segment["bytes"] * occurrences
            aggregate["necessary"] += segment["necessary"] * occurrences
    return fused_seconds, segmented_seconds, segments_by_name


def _empty_aggregate(target: dict[str, dict], segment: dict) -> dict:
    """Get-or-create the accumulator for one segment name, pinning its dtype.

    A segment name that arrived at two different precisions would make the summed
    compute floor meaningless, so the disagreement is raised rather than merged.
    """
    aggregate = target.setdefault(
        segment["name"],
        {
            "name": segment["name"],
            "flops": 0.0,
            "bytes": 0.0,
            "compute_dtype": segment["compute_dtype"],
            "necessary": 0.0,
        },
    )
    if aggregate["compute_dtype"] != segment["compute_dtype"]:
        raise ValueError(
            f"segment {segment['name']!r} reported both {aggregate['compute_dtype']!r} "
            f"and {segment['compute_dtype']!r} as its compute dtype"
        )
    return aggregate


def _merge_segment_totals(target: dict[str, dict], source: dict[str, dict]) -> None:
    for segment in source.values():
        aggregate = _empty_aggregate(target, segment)
        aggregate["flops"] += segment["flops"]
        aggregate["bytes"] += segment["bytes"]
        aggregate["necessary"] += segment["necessary"]


def compute_floors(log_dir: Path, levels: dict[str, dict]) -> dict[str, dict]:
    """Per-level {necessary, segmented} in GPU·seconds via the labeler roofline."""
    pool_specs = _pool_specs(log_dir)
    out: dict[str, dict] = {}
    for level_key, totals in levels.items():
        try:
            spec = _spec_for_level(level_key, pool_specs)
            model = _model(spec["config"])
            _check_precision(model, spec)
            out[level_key] = _label_payload(model, totals, spec)
        except Exception as error:  # noqa: BLE001 - isolate independent batch rows
            # One heterogeneous or unsupported scope must not discard valid
            # worker/pool labels in the same subprocess batch.
            out[level_key] = {"error": f"{type(error).__name__}: {error}"}
    return out


def compute_locked_compositions(log_dir: Path, compositions: dict[str, list[dict]]) -> dict:
    """Compose fixed-batch labels without fusing work across iteration boundaries.

    Rust deduplicates equal workloads. Each row here is one distinct shape plus its
    occurrence count; roofline times are evaluated before weighting and addition.
    The response is one compact semantic label per worker, independent of the number
    of source iterations.
    """
    pool_specs = _pool_specs(log_dir)
    out: dict[str, dict] = {}
    for level_key, weighted_shapes in compositions.items():
        try:
            spec = _spec_for_level(level_key, pool_specs)
            model = _model(spec["config"])
            _check_precision(model, spec)
            fused_seconds = 0.0
            segmented_seconds = 0.0
            segments_by_name: dict[str, dict] = {}
            iteration_count = sum(int(shape["occurrences"]) for shape in weighted_shapes)
            has_routed_matmul = _has_routed_matmul(model)
            basis_cache: dict[tuple[int | None, bool], dict | None] = {}
            shapes_by_basis: dict[tuple[int | None, bool], list[dict]] = defaultdict(list)
            for weighted_shape in weighted_shapes:
                occurrences = int(weighted_shape["occurrences"])
                if occurrences <= 0:
                    raise ValueError(f"occurrences must be positive, got {occurrences}")
                shapes_by_basis[
                    _basis_key(model, weighted_shape["totals"], has_routed_matmul)
                ].append(weighted_shape)
            peak = _peak_resolver(spec)
            bandwidth_gbps = gpu_mem_bandwidth_gbps(spec["gpu"])
            for basis_shapes in shapes_by_basis.values():
                basis = _validated_basis(
                    model,
                    basis_shapes[0]["totals"],
                    has_routed_matmul,
                    basis_cache,
                    spec["dtype"],
                )
                if basis is None:
                    group_fused, group_segmented, group_segments = _reduce_direct_group(
                        model, basis_shapes, spec
                    )
                else:
                    group_fused, group_segmented, group_segments = _reduce_affine_group(
                        basis, basis_shapes, peak, bandwidth_gbps
                    )
                fused_seconds += group_fused
                segmented_seconds += group_segmented
                _merge_segment_totals(segments_by_name, group_segments)
            out[level_key] = {
                "necessary": fused_seconds,
                "segmented": segmented_seconds,
                "segments": [segments_by_name[name] for name in sorted(segments_by_name)],
                "composition": {
                    "unique_shapes": len(weighted_shapes),
                    "iterations": iteration_count,
                    "affine_bases": sum(basis is not None for basis in basis_cache.values()),
                    "direct_fallback_bases": sum(basis is None for basis in basis_cache.values()),
                },
            }
        except Exception as error:  # noqa: BLE001 - isolate independent workers
            out[level_key] = {"error": f"{type(error).__name__}: {error}"}
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m model.work.floors <log_dir>  (scalars on stdin)")
    log_dir = Path(argv[0])
    request = json.load(sys.stdin)
    if "locked_compositions" in request:
        floors = compute_locked_compositions(log_dir, request["locked_compositions"])
    else:
        floors = compute_floors(log_dir, request["levels"])
    json.dump({"levels": floors, "meta": {"unit": "gpu_seconds"}}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
