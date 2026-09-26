"""Optimality floors — segmented and scope-fused necessary work, per level.

Called by the Rust `optimality` analyzer subject as a subprocess:

    uv run python -m model.work.floors <log_dir>

For unlocked analysis, Rust aggregates each level's workload (reusing the
`workload-conservation` subject's `groups` parser) and pipes it in on **stdin** as::

    {"levels": {"<level_key>": {matmul_tokens, prefill_tokens, decode_passes,
                                prefill_pairs, prefill_cached, decode_kv,
                                prefill_requests, prefill_stateful_requests}, ...}}

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

For batch-locked run analysis, Rust instead sends deduplicated iteration shapes,
one array per field (element ``i`` of every array is shape ``i``)::

    {"locked_compositions": {"<worker_key>": {
        "occurrences": [3, ...],
        "totals": {"matmul_tokens": [...], ..., "speculative_geometry": [{...}, ...]},
    }}}

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


def _model_for_spec(spec: dict):
    model = _model(spec["config"])
    if spec.get("arch_type") != "glm52_vllm_nvfp4_dsa_moe_speculative":
        return model
    mode = spec.get("mtp_mode", "index_share")
    if mode not in ("index_share", "full_index"):
        raise ValueError(f"unsupported speculative MTP mode {mode!r}")
    if mode == "full_index":
        from dataclasses import replace

        from .core import NormWeightGroup

        layers = [
            replace(stack, attn=replace(stack.attn, full_index=True))
            if stack.stage == "mtp_recurrent"
            else stack
            for stack in model.layers
        ]
        recurrent = next(stack for stack in layers if stack.stage == "mtp_recurrent")
        model = replace(
            model,
            layers=layers,
            norm_weights=[
                *model.norm_weights,
                NormWeightGroup(
                    "mtp_recurrent.indexer_k_norm",
                    recurrent.attn.index_head_dim,
                    1,
                    stage="mtp_recurrent",
                    param_count=0,
                ),
            ],
        )
    return model


def _validate_speculative_totals(spec: dict, totals: dict) -> None:
    if spec.get("arch_type") != "glm52_vllm_nvfp4_dsa_moe_speculative":
        if totals.get("speculative_geometry"):
            raise ValueError("speculative geometry requires a speculative architecture")
        return
    if not totals.get("speculative_geometry"):
        raise ValueError("speculative floors require per-stage workload geometry")
    for encoded in totals["speculative_geometry"]:
        if json.loads(encoded)["draft_tokens"] != spec["draft_tokens"]:
            raise ValueError("logged draft depth disagrees with params.json")


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
        arch_type = arch.get("type", "")
        arch_quant_dtype = (
            "fp4"
            if arch_type in ("glm52_vllm_nvfp4_dsa_moe", "glm52_vllm_nvfp4_dsa_moe_speculative")
            else None
        )
        if arch.get("fp8"):
            arch_quant_dtype = "fp8"
        specs[pool_tag] = {
            "arch_type": arch_type,
            "mtp_mode": arch.get("mtp_mode", "index_share"),
            "draft_tokens": arch.get("draft_tokens", 5),
            "config": arch["model_config"],
            # `gpu` is a group-level field (the arch block carries model/tp/fp8).
            "gpu": group["gpu"],
            "arch_fp8": bool(arch.get("fp8")),
            "arch_quant_dtype": arch_quant_dtype,
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
    config_quant_dtype = model.quant.compute_dtype if model.quant is not None else None
    arch_quant_dtype = spec.get("arch_quant_dtype", "fp8" if spec["arch_fp8"] else None)
    if arch_quant_dtype is not None and config_quant_dtype is None:
        raise ValueError(
            f"arch declares {arch_quant_dtype} but {spec['config']} has no "
            "quantization_config; point `model_config` at the matching quantized checkpoint"
        )
    if config_quant_dtype is not None and arch_quant_dtype is None:
        declaration = (
            "does not set fp8"
            if config_quant_dtype == "fp8"
            else "does not declare a matching quantized path"
        )
        raise ValueError(
            f"{spec['config']} declares a {model.quant.compute_dtype} quantization_config "
            f"but the arch {declaration}; "
            "the run and the accountant disagree on precision"
        )
    if config_quant_dtype != arch_quant_dtype:
        raise ValueError(
            f"{spec['config']} declares {config_quant_dtype} quantized compute but the arch "
            f"declares {arch_quant_dtype}; the run and the accountant disagree on precision"
        )


def _spec_for_level(level_key: str, pool_specs: dict[str, dict]) -> dict:
    """Which (config, gpu, dtype) a level computes under.

    ``<pool_tag>`` / ``<pool_tag>/<worker_id>`` take that pool's spec; ``cluster``
    spans every pool, so it requires them to share one model (true for PD single
    -model runs) — otherwise a per-pool-summed cluster floor is needed (deferred).
    """
    if level_key == "cluster":
        distinct = {
            (
                s["config"],
                s["gpu"],
                s["dtype"],
                s.get("arch_type"),
                s.get("mtp_mode"),
                s.get("draft_tokens"),
            )
            for s in pool_specs.values()
        }
        if len(distinct) != 1:
            raise ValueError(f"cluster floor needs one shared model across pools, saw {distinct}")
        spec = next(iter(pool_specs.values()))
    else:
        # Pool level is the bare tag; worker level is "<pool_tag>/<worker_id>".
        pool_tag = level_key.split("/", 1)[0]
        spec = pool_specs[pool_tag]
    return spec


def _aggregate_workload(totals: dict) -> Workload:
    """Assemble one giant-batch Workload from a level's pre-summed scalars.

    Prefill and decode attention each collapse to a single ``mask="full"``
    interaction carrying the summed pair / cached-KV counts (identity, since
    ``pairs()`` for full = ``q·k`` and we set ``q=1``). Decode reads all context
    keys from cache each step, so cached == pairs == ``decode_kv``; the self-key
    ``+1`` per step is negligible and omitted.
    """
    if totals.get("speculative_geometry"):
        from .speculative import aggregate_workload

        return aggregate_workload(totals)
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
    stateful_raw = totals.get("prefill_stateful_requests", 0)
    if (
        isinstance(stateful_raw, bool)
        or not isinstance(stateful_raw, (int, float))
        or not float(stateful_raw).is_integer()
        or stateful_raw < 0
    ):
        raise ValueError("prefill_stateful_requests must be a non-negative exact integer")
    prefill_stateful_requests = int(stateful_raw)
    if prefill_stateful_requests > prefill_requests:
        raise ValueError("prefill_stateful_requests cannot exceed prefill_requests")
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
        prefill_stateful_requests=prefill_stateful_requests,
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
    return {segment.name: segment.compute_dtype or default_dtype for segment in label.segments}


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
    "prefill_stateful_requests",
)
_LARGE_GEOMETRY_FIELDS = {"prefill_pairs", "prefill_cached", "decode_kv"}
_PREFILL_FIELDS = frozenset(
    {
        "prefill_tokens",
        "prefill_pairs",
        "prefill_cached",
        "prefill_requests",
        "prefill_stateful_requests",
    }
)
_DECODE_FIELDS = frozenset({"decode_passes", "decode_kv"})
_GEOMETRY_PROBE = 1_048_576

#: (routed matmul_tokens or None, past-vocab, prefill present, decode present) —
#: every nonlinearity the affine secant is not allowed to span.
_BasisKey = tuple[int | None, bool, bool, bool]


def _mode_presence(totals: dict) -> tuple[bool, bool]:
    """Whether this shape contains prefill / decode at all.

    A workload scalar reaching zero is not the edge of the same linear piece: the
    labeler emits no `*.attn.prefill` segments whatsoever for a pure-decode batch,
    so a secant anchored on a prefill-carrying origin cannot reconstruct one — the
    whole origin's prefill work survives as residue. Presence is part of the basis
    key rather than something the secant is asked to span.
    """
    prefill = int(totals["prefill_pairs"]) > 0 or int(totals["prefill_tokens"]) > 0
    decode = int(totals["decode_passes"]) > 0 or int(totals["decode_kv"]) > 0
    return prefill, decode


def _field_is_present(field_name: str, prefill_present: bool, decode_present: bool) -> bool:
    """Does this workload axis vary within a group carrying these modes?

    An absent mode's axes are pinned at zero for every member, so their deltas are
    always zero and their coefficients are dead. Probing them anyway is not merely
    wasted work: the probe materializes that mode's segments, which then enter the
    basis's segment-name set and make every reconstruction disagree with the direct
    label on `keys()`.
    """
    if field_name in _PREFILL_FIELDS:
        return prefill_present
    if field_name in _DECODE_FIELDS:
        return decode_present
    return True


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


def _basis_key(model, totals: dict, has_routed_matmul: bool) -> tuple[int | None, bool, bool, bool]:
    matmul_tokens = int(totals["matmul_tokens"])
    prefill_present, decode_present = _mode_presence(totals)
    return (
        matmul_tokens if has_routed_matmul else None,
        matmul_tokens >= model.vocab,
        prefill_present,
        decode_present,
    )


def _workload_basis(
    model, totals: dict, has_routed_matmul: bool
) -> dict[str, dict[str, tuple[float, float]] | dict[str, int]]:
    """Affine semantic-work basis, with nonlinear dimensions pinned in the key.

    Current model specs are affine in the eight compressed workload scalars except
    routed-expert weight loading, the embedding-table cap, and sparse attention's
    selected-key cap. Routed token counts are therefore pinned exactly; dense token
    counts use one basis on each side of the vocab cap. The shapes using a basis
    are independently checked below.

    The geometry origin is deliberately NOT zero, and it carries exactly the modes
    the group carries. Two separate cliffs make a zero origin wrong:

    - Saturation. A sparse-attention spec's work is `min(selected_k, pairs)`, so a
      secant anchored at zero geometry crosses the kink and reports a slope of
      `selected_k / probe` where the true slope past the cap is 0 — on GLM-5.2 DSA
      (`index_topk` 2048, probe 1,048,576) that over-stated every attention segment
      by ~3-4x. Anchoring inside the saturated regime keeps the secant in one piece.
    - Mode presence. Probing an absent mode's axes materializes segments the group's
      shapes do not have, so the origin only probes present modes (see
      `_field_is_present`); presence itself is pinned in the basis key.

    Shapes below the saturation cap still simply fail validation and take the exact
    direct path, as before.
    """
    prefill_present, decode_present = _mode_presence(totals)
    base_totals = {field_name: 0 for field_name in _WORKLOAD_FIELDS}
    for field_name in _LARGE_GEOMETRY_FIELDS:
        if _field_is_present(field_name, prefill_present, decode_present):
            base_totals[field_name] = _GEOMETRY_PROBE
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
        if not _field_is_present(field_name, prefill_present, decode_present):
            continue
        probe = _GEOMETRY_PROBE if field_name in _LARGE_GEOMETRY_FIELDS else 1
        probe_totals = dict(base_totals)
        probe_totals[field_name] += probe
        if field_name == "prefill_stateful_requests":
            # Stateful prefill requests are a strict subset of prefill requests.
            # Hold one parent request in both points to isolate this coefficient.
            probe_totals["prefill_requests"] = 1
            reference_totals = dict(base_totals, prefill_requests=1)
            reference = _segment_work(model, reference_totals)
        else:
            # Every large geometry field is already present at the probe scale in
            # the origin, so all other coefficients share the base reference.
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
            field_delta = int(totals.get(field_name, 0)) - basis["origin"][field_name]
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


class _ShapeColumns:
    """One worker's deduplicated shapes, as the parallel arrays Rust sends.

    Element `i` of every column is shape `i`. The vectorized paths read `counts`,
    one int64 array per workload field; the per-shape paths call `totals(row)`,
    which rebuilds exactly the row object the wire used to carry, so `model.label`
    sees the same values either way. A 2000 s PD run sends 1.6M shapes; a Python
    dict per shape spent more time in bookkeeping than the labeler spent on math.
    """

    def __init__(self, composition: dict):
        self.occurrences = np.asarray(composition["occurrences"], dtype=np.int64)
        self._columns = composition["totals"]
        size = len(self.occurrences)
        for field_name, column in self._columns.items():
            if len(column) != size:
                raise ValueError(
                    f"column {field_name!r} has {len(column)} rows, occurrences has {size}"
                )
        # Older analyzer payloads predate optional workload axes such as
        # `prefill_stateful_requests`; their wire default is zero.
        self.counts = {
            field_name: (
                np.asarray(self._columns[field_name]).astype(np.int64)
                if field_name in self._columns
                else np.zeros(size, dtype=np.int64)
            )
            for field_name in _WORKLOAD_FIELDS
        }

    def __len__(self) -> int:
        return len(self.occurrences)

    def totals(self, row: int) -> dict:
        return {field_name: column[row] for field_name, column in self._columns.items()}

    def carries_speculative_geometry(self) -> bool:
        return any(self._columns.get("speculative_geometry", ()))


def _validation_rows(shapes: _ShapeColumns, rows: np.ndarray) -> list[int]:
    """The group members a basis must reproduce exactly to be accepted.

    The first member, plus the argmin and argmax of every workload field. A basis
    is a secant fitted inside one linear piece, so a piece boundary crossed by
    this group shows up at a field extreme — checking the corners is what makes
    accepting the basis for the interior defensible. Any member may still be
    checked cheaply later; the corners are the ones that must be.
    """
    selected = {int(rows[0]): None}
    for field_name in _WORKLOAD_FIELDS:
        axis = shapes.counts[field_name][rows]
        for extreme in (np.argmin(axis), np.argmax(axis)):
            selected.setdefault(int(rows[extreme]), None)
    return list(selected)


#: Below this many members, fitting a basis costs more labels than it saves.
#: Building one is 1 base + up to 8 probes + up to 17 validation shapes (~26
#: `model.label` calls) to then evaluate the group as array math; the direct path
#: is exactly one call per shape. Routed models pin `matmul_tokens` in the basis
#: key, so prefill groups are near-singletons — on the 8h GLM-5.2 run, 7,441
#: prefill shapes spread over ~2,400 groups and paid for a basis apiece.
_MIN_BASIS_GROUP = 24


def _validated_basis(
    model,
    shapes: _ShapeColumns,
    rows: np.ndarray,
    has_routed_matmul: bool,
    basis_cache: dict[_BasisKey, dict | None],
    default_dtype: str,
) -> dict | None:
    """Build and independently validate one basis before any batch reduction."""
    totals = shapes.totals(int(rows[0]))
    basis_key = _basis_key(model, totals, has_routed_matmul)
    if totals.get("speculative_geometry") or len(rows) < _MIN_BASIS_GROUP:
        # Counted with the validation failures: both mean "this group was reduced
        # one shape at a time".
        basis_cache[basis_key] = None
    if basis_key not in basis_cache:
        candidate = _workload_basis(model, totals, has_routed_matmul)
        matches = all(
            _segment_work_matches(
                _reconstruct_segment_work(candidate, shapes.totals(row)),
                _segment_work(model, shapes.totals(row)),
            )
            for row in _validation_rows(shapes, rows)
        )
        if matches:
            # Validation just proved the basis spans exactly the direct label's
            # segment names, so one label at these totals resolves every dtype.
            candidate["dtypes"] = _segment_dtypes(model, totals, default_dtype)
            basis_cache[basis_key] = candidate
        else:
            basis_cache[basis_key] = None
    return basis_cache[basis_key]


def _reduce_affine_group(
    basis: dict,
    shapes: _ShapeColumns,
    rows: np.ndarray,
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
    workload_deltas = np.empty((len(rows), len(field_names)), dtype=np.float64)
    for column_index, field_name in enumerate(field_names):
        workload_deltas[:, column_index] = (
            shapes.counts[field_name][rows] - basis["origin"][field_name]
        )
    occurrences = shapes.occurrences[rows].astype(np.float64)
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
    model, shapes: _ShapeColumns, rows: np.ndarray, spec: dict, peak, bandwidth_gbps: float
) -> tuple[float, float, dict[str, dict]]:
    """Correct fallback for a model whose work is not affine in a basis.

    `peak` / `bandwidth_gbps` are passed in rather than rebuilt per shape: the
    resolver memoizes per dtype, so constructing one per shape threw the memo
    away and re-read the GPU spec for every row of a 170k-shape run.
    """
    fused_seconds = 0.0
    segmented_seconds = 0.0
    segments_by_name: dict[str, dict] = {}
    default_dtype = spec["dtype"]
    for row in rows:
        occurrences = int(shapes.occurrences[row])
        label = model.label(_aggregate_workload(shapes.totals(int(row))))
        payload = _payload_from_segment_work(
            {
                segment.name: (segment.flops_total, segment.bytes_total)
                for segment in label.segments
            },
            {segment.name: segment.compute_dtype or default_dtype for segment in label.segments},
            peak,
            bandwidth_gbps,
        )
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
            model = _model_for_spec(spec)
            _check_precision(model, spec)
            _validate_speculative_totals(spec, totals)
            out[level_key] = _label_payload(model, totals, spec)
        except Exception as error:  # noqa: BLE001 - isolate independent batch rows
            # One heterogeneous or unsupported scope must not discard valid
            # worker/pool labels in the same subprocess batch.
            out[level_key] = {"error": f"{type(error).__name__}: {error}"}
    return out


def _rows_by_basis(model, shapes: _ShapeColumns, has_routed_matmul: bool) -> list[np.ndarray]:
    """Shape rows grouped by `_basis_key`, groups in first-appearance order.

    Rows keep their input order inside a group, so every reduction sums in the
    same order as a per-shape loop would.
    """
    counts = shapes.counts
    matmul_tokens = counts["matmul_tokens"]
    prefill_present = (counts["prefill_pairs"] > 0) | (counts["prefill_tokens"] > 0)
    decode_present = (counts["decode_passes"] > 0) | (counts["decode_kv"] > 0)
    # One integer per `_basis_key` tuple: the routed token count (or 0), then the
    # three flags in the low bits.
    key = (
        (matmul_tokens if has_routed_matmul else np.zeros_like(matmul_tokens)) * 8
        + (matmul_tokens >= model.vocab) * 4
        + prefill_present * 2
        + decode_present
    )
    _unique, first_rows, group_of_row = np.unique(key, return_index=True, return_inverse=True)
    group_of_row = group_of_row.reshape(-1)
    rows_in_group_order = np.argsort(group_of_row, kind="stable")
    groups = np.split(rows_in_group_order, np.cumsum(np.bincount(group_of_row))[:-1])
    return [groups[group] for group in np.argsort(first_rows, kind="stable")]


def compute_locked_compositions(log_dir: Path, compositions: dict[str, dict]) -> dict:
    """Compose fixed-batch labels without fusing work across iteration boundaries.

    Rust deduplicates equal workloads. Each row here is one distinct shape plus its
    occurrence count; roofline times are evaluated before weighting and addition.
    The response is one compact semantic label per worker, independent of the number
    of source iterations.
    """
    pool_specs = _pool_specs(log_dir)
    out: dict[str, dict] = {}
    for level_key, composition in compositions.items():
        try:
            spec = _spec_for_level(level_key, pool_specs)
            model = _model_for_spec(spec)
            _check_precision(model, spec)
            shapes = _ShapeColumns(composition)
            if not len(shapes):
                _validate_speculative_totals(spec, {})
            nonpositive = np.flatnonzero(shapes.occurrences <= 0)
            first_nonpositive = int(nonpositive[0]) if len(nonpositive) else len(shapes)
            if (
                spec.get("arch_type") == "glm52_vllm_nvfp4_dsa_moe_speculative"
                or shapes.carries_speculative_geometry()
            ):
                # Up to and including the first bad count, so the error a caller
                # sees is the first bad row's, whichever check it fails.
                for row in range(min(first_nonpositive + 1, len(shapes))):
                    _validate_speculative_totals(spec, shapes.totals(row))
            if first_nonpositive < len(shapes):
                raise ValueError(
                    f"occurrences must be positive, got {shapes.occurrences[first_nonpositive]}"
                )
            fused_seconds = 0.0
            segmented_seconds = 0.0
            segments_by_name: dict[str, dict] = {}
            iteration_count = int(shapes.occurrences.sum())
            has_routed_matmul = _has_routed_matmul(model)
            basis_cache: dict[_BasisKey, dict | None] = {}
            peak = _peak_resolver(spec)
            bandwidth_gbps = gpu_mem_bandwidth_gbps(spec["gpu"])
            for rows in _rows_by_basis(model, shapes, has_routed_matmul):
                basis = _validated_basis(
                    model,
                    shapes,
                    rows,
                    has_routed_matmul,
                    basis_cache,
                    spec["dtype"],
                )
                if basis is None:
                    group_fused, group_segmented, group_segments = _reduce_direct_group(
                        model, shapes, rows, spec, peak, bandwidth_gbps
                    )
                else:
                    group_fused, group_segmented, group_segments = _reduce_affine_group(
                        basis, shapes, rows, peak, bandwidth_gbps
                    )
                fused_seconds += group_fused
                segmented_seconds += group_segmented
                _merge_segment_totals(segments_by_name, group_segments)
            out[level_key] = {
                "necessary": fused_seconds,
                "segmented": segmented_seconds,
                "segments": [segments_by_name[name] for name in sorted(segments_by_name)],
                "composition": {
                    "unique_shapes": len(shapes),
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
