"""Stock vLLM Neuron Llama forward split into model/head regions (worker only).

Each context group runs the split engine and the unchanged stock engine on the
identical canonical workload, under ``region-runs/<uuid>/{split,stock}``. Rows
are emitted only after four gates pass, in order:

1. structural: every compile's partition receipt passed (``vllm_region_partition``),
   and each loaded region NEFF binds to exactly one (region, phase, bucket);
2. precision: the unchanged vendor full-logit check passes on both engines;
3. equivalence: generated tokens match the stock engine for every case, and
   RMS(split - FP32) <= 1.10 x RMS(stock - FP32) for every case;
4. timing: per shape, model median + head median is within 5% of the stock
   whole-forward median measured in the same group.

Gate inputs and verdicts stay in the run directory. Nothing is fitted or scaled.
"""

from __future__ import annotations

import json
import math
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

REGION_ENGINE = "profiling.runners.neuron.vllm_region_engine"
REGIONS = ("model", "head")
VERIFIED_SHAPES = frozenset({("prefill", 512), ("decode", 1), ("decode", 16)})
MAX_ERROR_RATIO = 1.10
TIMING_TOLERANCE = 0.05

_PROVENANCE: dict[str, str] = {}


def validate_region_shape(region, **forward):
    """Initial verified scope only: C512, pool 6782/page 32, TP4 BF16, three shapes."""
    if region not in REGIONS:
        raise ValueError("region must be model or head")
    validate_shape(**forward)
    if forward["max_model_len"] != 512:
        raise ValueError("region split is verified only at max_model_len 512")
    if (forward["phase"], forward["token_bucket"]) not in VERIFIED_SHAPES:
        raise ValueError("region split is verified only for prefill 512 and decode 1/16")


def _rms(values) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(np.square(values))))


def case_equivalence(split, stock, fp32) -> dict:
    """Compare one case's split and stock full logits against the FP32 reference."""
    import numpy as np

    split, stock, fp32 = (np.asarray(a, dtype=np.float64) for a in (split, stock, fp32))
    if not split.shape == stock.shape == fp32.shape:
        raise ValueError("split, stock and reference logits differ in shape")
    split_error, stock_error = _rms(split - fp32), _rms(stock - fp32)
    if stock_error == 0:
        ratio = 1.0 if split_error == 0 else math.inf
    else:
        ratio = split_error / stock_error
    return {
        "bit_identical": bool(np.array_equal(split, stock)),
        "rms_split_minus_stock": _rms(split - stock),
        "max_abs_split_minus_stock": float(np.abs(split - stock).max()),
        "rms_split_vs_fp32": split_error,
        "rms_stock_vs_fp32": stock_error,
        "error_ratio": ratio,
    }


def equivalence_verdict(cases: list[dict], max_ratio: float = MAX_ERROR_RATIO) -> dict:
    """Every case needs identical tokens and an FP32 error ratio within ``max_ratio``."""
    if not cases:
        raise ValueError("split-vs-stock equivalence needs at least one case")
    tokens = all(case["tokens_identical"] for case in cases)
    worst = max(case["error_ratio"] for case in cases)
    return {
        "passed": bool(tokens and worst <= max_ratio),
        "criterion": (
            "identical generated tokens for every case and "
            f"RMS(split-FP32)/RMS(stock-FP32) <= {max_ratio} for every case"
        ),
        "cases": len(cases),
        "bit_identical_cases": sum(case["bit_identical"] for case in cases),
        "tokens_identical_cases": sum(case["tokens_identical"] for case in cases),
        "worst_error_ratio": worst,
        "per_case": cases,
    }


def compare_split_to_stock(split_root: Path, stock_root: Path) -> dict:
    import numpy as np

    from profiling.runners.neuron.vllm_forward_reference import (
        reference_contract,
        reference_path,
    )

    if (split_root / "cases.json").read_bytes() != (stock_root / "cases.json").read_bytes():
        raise ValueError("split and stock engines ran different canonical cases")
    plan = json.loads((stock_root / "plan.json").read_text())
    split_rows = {
        row["case_id"]: row
        for row in json.loads((split_root / "accuracy-outputs.json").read_text())
    }
    cases = []
    for row in json.loads((stock_root / "accuracy-outputs.json").read_text()):
        case_id = row["case_id"]
        split_row = split_rows[case_id]
        if split_row["prompt_ids"] != row["prompt_ids"]:
            raise ValueError(f"split and stock prompts differ for {case_id}")
        # The stock history defines the FP32 reference positions for both engines.
        contract = reference_contract(
            plan["identity"], "fp32", row["prompt_ids"] + row["token_ids"][:-1]
        )
        fp32 = np.load(reference_path(plan["reference_cache"], contract))
        comparison = case_equivalence(
            np.load(split_root / f"native-{case_id}.npy"),
            np.load(stock_root / f"native-{case_id}.npy"),
            fp32,
        )
        cases.append(
            {
                "case_id": case_id,
                "tokens_identical": split_row["token_ids"] == row["token_ids"],
                **comparison,
            }
        )
    return equivalence_verdict(cases)


def timing_verdict(region_times: dict, stock_times: dict, shapes, tolerance=TIMING_TOLERANCE):
    """Per shape: sum of region medians against the same-group stock whole median."""
    verdict = {}
    for phase, bucket in sorted(shapes):
        model = statistics.median(region_times[("model", phase, bucket)])
        head = statistics.median(region_times[("head", phase, bucket)])
        stock = statistics.median(stock_times[(phase, bucket)])
        error = (model + head) / stock - 1
        verdict[f"{phase}:{bucket}"] = {
            "model_median_ms": model,
            "head_median_ms": head,
            "sum_region_medians_ms": model + head,
            "stock_whole_median_ms": stock,
            "error_pct": 100 * error,
            "forwards": len(region_times[("model", phase, bucket)]),
            "stock_forwards": len(stock_times[(phase, bucket)]),
            "passed": abs(error) <= tolerance,
        }
    return verdict


def compiled_shapes(specs, context) -> set[tuple[str, int]]:
    """Graphs the engine compiles for ``cases_for_plan``: prefill plus every decode bucket."""
    buckets = {spec["token_bucket"] for spec in specs if spec["phase"] == "decode"} or {16}
    if any(spec["phase"] == "prefill" for spec in specs):
        buckets.add(16)
    return {("prefill", context)} | {("decode", bucket) for bucket in {1, *buckets}}


def _require_precision(root: Path, shapes: set[str], engine: str) -> None:
    precision = json.loads((root / "precision.json").read_text())
    failed = sorted(shape for shape in shapes if not precision["by_shape"][shape]["passed"])
    if failed:
        raise RuntimeError(f"{engine} full-logit vendor check failed for {failed}")


def _require_partition_receipts(root: Path, stage: str, expected) -> None:
    receipts = [
        json.loads(path.read_text())
        for path in (root / f"partition-{stage}").glob("rank*-*/receipt.json")
    ]
    if not receipts or not all(receipt["passed"] for receipt in receipts):
        raise RuntimeError(f"structural partition receipts missing or failed in {stage}")
    observed = {(receipt["phase"], receipt["token_bucket"]) for receipt in receipts}
    if not set(expected) <= observed:
        raise RuntimeError(f"{stage} did not partition every compiled shape: {observed}")


def _require_stable_region_binaries(split_root: Path, compile_cache: Path) -> None:
    from profiling.runners.neuron.vllm_identity import compile_cache_snapshot

    bound = json.loads((split_root / "region-graph-binding.json").read_text())
    accuracy = json.loads((split_root / "accuracy-binaries-after.json").read_text())
    current = compile_cache_snapshot(compile_cache)
    for graph in bound:
        if graph not in accuracy or current.get(graph) != accuracy[graph]:
            raise RuntimeError(f"region NEFF {graph} changed after numerical validation")


def _measure_group(root: Path, cache: Path, context: int, specs: list[dict]) -> dict:
    """Run every stage and gate for one context group; return region times."""
    from profiling.runners.neuron.vllm_forward_trace import export_and_measure
    from profiling.runners.neuron.vllm_identity import seal_accuracy_binaries
    from profiling.runners.neuron.vllm_region_trace import export_and_measure_regions

    stock, split = root / "stock", root / "split"
    compile_cache = cache / "cache/neuron/compile_cache"
    shapes = {(spec["phase"], spec["token_bucket"]) for spec in specs}
    compiled = compiled_shapes(specs, context)
    run_stage(stock, "accuracy")
    run_stage(split, "accuracy", engine=REGION_ENGINE)
    _require_partition_receipts(split, "accuracy", compiled)
    run_stage(stock, "reference")
    run_stage(split, "reference")
    precision_shapes = {precision_shape(spec, context) for spec in specs}
    _require_precision(stock, precision_shapes, "stock")
    _require_precision(split, precision_shapes, "split")
    seal_accuracy_binaries(stock)
    equivalence = compare_split_to_stock(split, stock)
    (root / "split-equivalence.json").write_text(json.dumps(equivalence, indent=2))
    if not equivalence["passed"]:
        raise RuntimeError(f"split-vs-stock equivalence failed; see {root}")
    run_stage(stock, "profile")
    stock_times = export_and_measure(stock, compile_cache)
    run_stage(split, "profile", engine=REGION_ENGINE)
    _require_partition_receipts(split, "profile", compiled)
    region_times = export_and_measure_regions(split, compile_cache, compiled)
    _require_stable_region_binaries(split, compile_cache)
    timing = timing_verdict(region_times, stock_times, shapes)
    (root / "timing-gate.json").write_text(json.dumps(timing, indent=2))
    return {"region_times": region_times, "timing": timing, "equivalence": equivalence}


def profile_region_batch(kwargs_list):
    results = [None] * len(kwargs_list)
    groups = {}
    for index, spec in enumerate(kwargs_list):
        try:
            validate_region_shape(**spec)
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
    for context, indices in groups.items():
        root = cache / "region-runs" / uuid.uuid4().hex
        forward_specs = []
        for index in indices:
            spec = {name: v for name, v in kwargs_list[index].items() if name != "region"}
            if spec not in forward_specs:
                forward_specs.append(spec)
        for engine in ("stock", "split"):
            (root / engine).mkdir(parents=True)
            write_plan(root / engine, forward_specs, context, model, identity, cache)
        try:
            group = _measure_group(root, cache, context, forward_specs)
        except Exception as error:
            for index in indices:
                results[index] = RunnerResult(error=f"{error}; artifacts={root}")
            continue
        from profiling.runners.neuron.vllm_forward_work import estimate_region_work

        for index in indices:
            spec = kwargs_list[index]
            shape = (spec["phase"], spec["token_bucket"])
            gate = group["timing"][f"{shape[0]}:{shape[1]}"]
            if not gate["passed"]:
                results[index] = RunnerResult(
                    error=(
                        f"region timing composition off by {gate['error_pct']:.2f}% "
                        f"(limit {100 * TIMING_TOLERANCE:.0f}%); see {root / 'timing-gate.json'}"
                    )
                )
                continue
            times = group["region_times"][(spec["region"], *shape)]
            duration = statistics.median(times)
            work = estimate_region_work(spec["region"], *shape, context)
            results[index] = RunnerResult(
                metrics=ComputeMetrics(
                    duration,
                    work["contraction_flops_per_rank"] / (duration * 1e9),
                    work["persistent_operand_bytes_per_rank"] / (duration * 1e6),
                )
            )
            equivalence = group["equivalence"]
            _PROVENANCE[key(spec)] = (
                f"stock_forward_fx_regions; region={spec['region']}; artifacts={root}; "
                f"samples={len(times)}; timing_gate_error_pct={gate['error_pct']:.3f}; "
                f"bit_identical_cases={equivalence['bit_identical_cases']}/"
                f"{equivalence['cases']}; "
                f"worst_error_ratio={equivalence['worst_error_ratio']:.4f}; "
                "work=per_rank_contractions_and_persistent_operands_estimates_not_counters"
            )
    return results


def row_provenance(**spec):
    return _PROVENANCE.get(key(spec))
