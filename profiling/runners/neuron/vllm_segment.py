"""Stock vLLM Neuron Llama forward segmented at its collectives (worker only).

Each context group runs the UNCHANGED stock engine under
``segment-runs/<uuid>/`` through four stages: ``accuracy`` and ``reference``
(vendor full-logit check), ``profile`` (untraced system-profile whole-forward
timing) and ``capture`` (native device instruction trace of the same engine
configuration, ``vllm_segment_engine``). Rows are emitted only after these
gates pass, in order:

1. precision: the unchanged vendor full-logit check passes for every row shape;
2. binary identity: every traced NEFF is byte-identical (sha256) to the sealed,
   numerically validated stock NEFF of its shape, and the compile cache still
   holds those bytes after the capture;
3. trace completeness and structure (``vllm_segment_trace``): both physical cores
   of every captured rank traced, no lost notification except DMA, the exact
   collective layout, every layer within 10% of its forward's layer mean, and
   segments summing exactly to each execution span;
4. timing: per shape, embedding + 32 x (attention + MLP) + head medians within 5%
   of the same-group untraced stock whole-forward median.

Gate inputs and verdicts stay in the run directory. Nothing is fitted or scaled.
"""

from __future__ import annotations

import json
import os
import statistics
import uuid
from pathlib import Path

from profiling.runners.metrics import ComputeMetrics, RunnerResult
from profiling.runners.neuron.vllm_forward import (
    key,
    precision_shape,
    run_stage,
    validate_shape,
    verify_identity,
    write_plan,
)

SEGMENT_ENGINE = "profiling.runners.neuron.vllm_segment_engine"
SEGMENTS = ("embedding", "attention_block", "mlp_block", "head")
VERIFIED_SHAPES = frozenset({("prefill", 512), ("decode", 1), ("decode", 16)})
TIMING_TOLERANCE = 0.05
CAPTURE_RANKS = [0, 1, 2, 3]
# Instruction buffer sized from the retained captures: the 128 MiB default held
# about 8.4M instruction rows; one window here traces about 17M per rank. Event
# notifications dropped alongside at their 64 MiB default. DMA loss is tolerated.
CAPTURE_ENV = {
    "NEURON_RT_PROFILE_BUF_INSTRUCTION_MB": "512",
    "NEURON_RT_PROFILE_BUF_EVENT_MB": "512",
}
DECODE_FORWARDS_PER_CALL = 7  # Eight greedy tokens: one prefill then seven decodes.

_PROVENANCE: dict[str, str] = {}


def validate_segment_shape(segment, **forward):
    """Initial verified scope only: C512, pool 6782/page 32, TP4 BF16, three shapes."""
    if segment not in SEGMENTS:
        raise ValueError("segment must be embedding, attention_block, mlp_block or head")
    validate_shape(**forward)
    if forward["max_model_len"] != 512:
        raise ValueError("collective segments are verified only at max_model_len 512")
    if (forward["phase"], forward["token_bucket"]) not in VERIFIED_SHAPES:
        raise ValueError("collective segments are verified only for prefill 512 and decode 1/16")


def capture_calls(specs) -> dict[str, int]:
    """Public calls traced per decode bucket: two B1 calls (14 decodes), one otherwise.

    A prefill row is traced through the B16 call's prompts, the same bucket the
    stock prefill row's numerical acceptance uses.
    """
    buckets = {spec["token_bucket"] for spec in specs if spec["phase"] == "decode"}
    if any(spec["phase"] == "prefill" for spec in specs):
        buckets.add(16)
    return {str(bucket): 2 if bucket == 1 else 1 for bucket in sorted(buckets)}


def check_capture_counts(combined: dict, captured_calls: list[dict], context: int) -> dict:
    """The trace must hold every prefill and decode the traced public calls ran."""
    counts = {f"{phase}:{bucket}": len(rows) for (phase, bucket), rows in combined.items()}
    prefills = sum(call["batch"] for call in captured_calls)
    if counts.get(f"prefill:{context}", 0) != prefills:
        raise ValueError(f"device trace holds {counts}, expected {prefills} prefills")
    for bucket in {call["bucket"] for call in captured_calls}:
        calls = sum(call["bucket"] == bucket for call in captured_calls)
        if counts.get(f"decode:{bucket}", 0) < DECODE_FORWARDS_PER_CALL * calls:
            raise ValueError(f"device trace lacks decode {bucket} forwards: {counts}")
    return counts


def check_traced_binaries(profiles: Path, sessions: list[dict], graphs: dict) -> dict:
    """Every traced graph's NEFF bytes must equal the numerically sealed stock NEFF."""
    from profiling.runners.neuron.vllm_identity import sha256

    traced = sorted({f["graph"] for s in sessions for f in s["forwards"]})
    receipt = {}
    for graph in traced:
        expected = graphs[graph]["binary"]["neff_sha256"]
        observed = sha256(profiles / "neffs" / graph / f"graph_{graph}.neff")
        if observed != expected:
            raise ValueError(f"traced NEFF {graph} differs from the validated stock binary")
        receipt[graph] = {
            "phase": graphs[graph]["phase"],
            "token_bucket": graphs[graph]["token_bucket"],
            "neff_sha256": observed,
        }
    return receipt


def _require_precision(root: Path, shapes: set[str]) -> None:
    precision = json.loads((root / "precision.json").read_text())
    failed = sorted(shape for shape in shapes if not precision["by_shape"][shape]["passed"])
    if failed:
        raise RuntimeError(f"stock full-logit vendor check failed for {failed}")


def _measure_group(root: Path, cache: Path, context: int, specs: list[dict]) -> dict:
    """Run every stage and gate for one context group; return per-forward segments."""
    from profiling.runners.neuron.vllm_forward_trace import export_and_measure
    from profiling.runners.neuron.vllm_identity import (
        seal_accuracy_binaries,
        verify_cache_binaries,
    )
    from profiling.runners.neuron.vllm_segment_trace import (
        analyze_session,
        combine_ranks,
        export_device_sessions,
        homogeneity_verdict,
        segment_times,
        timing_verdict,
    )

    compile_cache = cache / "cache/neuron/compile_cache"
    shapes = {(spec["phase"], spec["token_bucket"]) for spec in specs}
    run_stage(root, "accuracy")
    run_stage(root, "reference")
    _require_precision(root, {precision_shape(spec, context) for spec in specs})
    sealed = seal_accuracy_binaries(root)
    if not sealed["binary_binding_available"]:
        raise RuntimeError("stock NEFFs were not warm-loaded; binary identity is unproven")
    graphs = sealed["graphs"]
    run_stage(root, "profile")
    stock_times = export_and_measure(root, compile_cache)
    run_stage(root, "capture", engine=SEGMENT_ENGINE, env=CAPTURE_ENV)
    sessions_dirs = export_device_sessions(root / "device-profiles", root / "device-parquet")
    graph_shapes = {graph: (row["phase"], row["token_bucket"]) for graph, row in graphs.items()}
    sessions = [analyze_session(path, graph_shapes) for path in sessions_dirs]
    (root / "segment-sessions.json").write_text(json.dumps(sessions, indent=2))
    if len(sessions) != len(CAPTURE_RANKS):
        raise RuntimeError(f"expected {len(CAPTURE_RANKS)} rank sessions, got {len(sessions)}")
    identity = {
        "traced": check_traced_binaries(root / "device-profiles", sessions, graphs),
        "compile_cache_after_capture": verify_cache_binaries(compile_cache, graphs),
        "binary_provenance": str(root / "binary-provenance.json"),
    }
    (root / "identity-gate.json").write_text(json.dumps(identity, indent=2))
    homogeneity = homogeneity_verdict(sessions)
    (root / "homogeneity-gate.json").write_text(json.dumps(homogeneity, indent=2))
    combined = combine_ranks(sessions)
    captured = json.loads((root / "captured-outputs.json").read_text())
    counts = check_capture_counts(combined, captured, context)
    times = segment_times(combined)
    (root / "segment-timing.json").write_text(
        json.dumps({":".join(map(str, k)): v for k, v in times.items()}, indent=2)
    )
    timing = timing_verdict(times, stock_times, shapes, TIMING_TOLERANCE)
    (root / "timing-gate.json").write_text(json.dumps(timing, indent=2))
    return {
        "times": times,
        "timing": timing,
        "homogeneity": homogeneity,
        "sessions": sessions,
        "counts": counts,
    }


def _provenance(spec, root, group, gate) -> str:
    from profiling.runners.neuron.vllm_segment_trace import layer_statistics, rank_spread

    shape = (spec["phase"], spec["token_bucket"])
    times = group["times"][(spec["segment"], *shape)]
    spread = rank_spread(group["sessions"], shape)[spec["segment"]]
    text = (
        f"stock_forward_collective_segments; segment={spec['segment']}; artifacts={root}; "
        f"forwards={len(times)}; ranks={len(group['sessions'])}; "
        f"rank_spread_pct={spread['spread_pct']:.3f}; "
        f"timing_gate_error_pct={gate['error_pct']:.3f}; traced_neff_sha256_equal_stock"
    )
    block = {"attention_block": "attention", "mlp_block": "mlp"}.get(spec["segment"])
    if block:
        layers = layer_statistics(group["sessions"], shape)[block]
        text += (
            f"; per_layer_median_ms=[{layers['per_layer_median_min_ms']:.6f},"
            f"{layers['per_layer_median_max_ms']:.6f}]; "
            f"worst_layer_deviation={layers['worst_layer_deviation']:.4f}; "
            f"median_layer_cv={layers['median_cv']:.4f}"
        )
    return text + "; work=per_rank_semantic_split_estimates_not_counters"


def profile_segment_batch(kwargs_list):
    results = [None] * len(kwargs_list)
    groups = {}
    for index, spec in enumerate(kwargs_list):
        try:
            validate_segment_shape(**spec)
            groups.setdefault(spec["max_model_len"], []).append(index)
        except Exception as error:
            results[index] = RunnerResult(error=str(error))
    if not groups:
        return results
    cache = Path(os.environ["SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"])
    model = Path(os.environ["SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR"])
    try:
        identity = verify_identity(model)
    except Exception as error:
        return [result or RunnerResult(error=str(error)) for result in results]
    from profiling.runners.neuron.vllm_forward_work import estimate_segment_work

    for context, indices in groups.items():
        root = cache / "segment-runs" / uuid.uuid4().hex
        root.mkdir(parents=True)
        forward_specs = []
        for index in indices:
            spec = {name: v for name, v in kwargs_list[index].items() if name != "segment"}
            if spec not in forward_specs:
                forward_specs.append(spec)
        plan = write_plan(root, forward_specs, context, model, identity, cache)
        plan.update(capture_calls=capture_calls(forward_specs), capture_ranks=CAPTURE_RANKS)
        (root / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
        try:
            group = _measure_group(root, cache, context, forward_specs)
        except Exception as error:
            for index in indices:
                results[index] = RunnerResult(error=f"{error}; artifacts={root}")
            continue
        for index in indices:
            spec = kwargs_list[index]
            shape = (spec["phase"], spec["token_bucket"])
            gate = group["timing"][f"{shape[0]}:{shape[1]}"]
            homogeneity = group["homogeneity"][f"{shape[0]}:{shape[1]}"]
            if not homogeneity["passed"]:
                results[index] = RunnerResult(
                    error=(
                        "interior layer blocks deviate beyond "
                        f"{100 * homogeneity['tolerance']:.0f}% of the layer mean in "
                        f"{len(homogeneity['failures'])} rank block(s); the 32x layer fold is "
                        f"unsupported; see {root / 'homogeneity-gate.json'}"
                    )
                )
                continue
            if not gate["passed"]:
                results[index] = RunnerResult(
                    error=(
                        f"segment timing composition off by {gate['error_pct']:.2f}% "
                        f"(limit {100 * TIMING_TOLERANCE:.0f}%); see {root / 'timing-gate.json'}"
                    )
                )
                continue
            duration = statistics.median(group["times"][(spec["segment"], *shape)])
            work = estimate_segment_work(spec["segment"], *shape, context)
            results[index] = RunnerResult(
                metrics=ComputeMetrics(
                    duration,
                    work["contraction_flops_per_rank"] / (duration * 1e9),
                    work["persistent_operand_bytes_per_rank"] / (duration * 1e6),
                )
            )
            _PROVENANCE[key(spec)] = _provenance(spec, root, group, gate)
    return results


def row_provenance(**spec):
    return _PROVENANCE.get(key(spec))
