"""Stock vLLM Neuron forward with prefill segmentation (worker only).

The same stages as ``vllm_forward`` (accuracy, CPU reference, profile) with a
plan that carries ``segment_size``; the shared engine then serves the context
with S-token prefill segments. One run covers one (context, segment size).
Prefill rows are keyed by their cached prior; see ``measure_segmented_trace``.
"""

from __future__ import annotations

import json
import os
import statistics
import uuid
from pathlib import Path

from profiling.db.args import DType
from profiling.runners.metrics import ComputeMetrics, RunnerResult
from profiling.runners.neuron.stock_contract import (
    BLOCK_SIZE,
    KV_BLOCKS,
    TP_SIZE,
    largest_decode_bucket,
)
from profiling.runners.neuron.vllm_forward import (
    key,
    precision_shape,
    run_stage,
    verify_identity,
    write_plan,
)
from profiling.runners.neuron.vllm_identity import seal_accuracy_binaries

# 8192 cross-checks the unsegmented whole forward; the others exceed it.
SEGMENTED_CONTEXTS = (8192, 16384, 32768)
SEGMENT_SIZES = (512, 1024, 2048, 4096)

_PROVENANCE: dict[str, str] = {}


def validate_segmented_shape(
    phase,
    token_bucket,
    prior_tokens,
    segment_size,
    max_model_len,
    kv_blocks,
    block_size,
    tp_size,
    dtype,
):
    if DType.from_value(dtype) != DType.BF16:
        raise ValueError("stock Llama forward requires BF16 weights, compute and KV")
    if (kv_blocks, block_size, tp_size) != (KV_BLOCKS, BLOCK_SIZE, TP_SIZE):
        raise ValueError(
            f"verified stock configuration requires {KV_BLOCKS} KV blocks, "
            f"page{BLOCK_SIZE} and TP{TP_SIZE}"
        )
    if max_model_len not in SEGMENTED_CONTEXTS:
        raise ValueError(f"segmented max_model_len must be one of {SEGMENTED_CONTEXTS}")
    if segment_size not in SEGMENT_SIZES or segment_size >= max_model_len:
        raise ValueError(f"segment_size must be one of {SEGMENT_SIZES} and below the context")
    if phase == "prefill":
        if token_bucket != segment_size:
            raise ValueError("a prefill segment runs the segment-size token bucket")
        if prior_tokens < 0 or prior_tokens % segment_size:
            raise ValueError("prefill prior_tokens must be a whole number of segments")
        if prior_tokens + 1 > max_model_len - 8:
            raise ValueError("no canonical (context-8)-token prompt has a segment at this prior")
    elif phase == "decode":
        if prior_tokens != 0:
            raise ValueError("decode rows carry prior_tokens 0")
        if token_bucket < 1 or token_bucket & (token_bucket - 1):
            raise ValueError("decode buckets are powers of two")
        if token_bucket > largest_decode_bucket(max_model_len):
            raise ValueError("decode bucket exceeds the context's KV-resident capacity")
    else:
        raise ValueError("phase must be prefill or decode")


def measurement_key(spec):
    return (spec["phase"], spec["token_bucket"], spec["prior_tokens"])


def profile_segmented_forward_batch(kwargs_list):
    results = [None] * len(kwargs_list)
    groups = {}
    for index, spec in enumerate(kwargs_list):
        try:
            validate_segmented_shape(**spec)
            groups.setdefault((spec["max_model_len"], spec["segment_size"]), []).append(index)
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
    from profiling.runners.neuron.vllm_forward_trace import export_and_measure
    from profiling.runners.neuron.vllm_forward_work import estimate_segmented_work

    for (context, segment), indices in groups.items():
        root = cache / "segmented-forward-runs" / uuid.uuid4().hex
        root.mkdir(parents=True)
        specs = [kwargs_list[i] for i in indices]
        write_plan(root, specs, context, model, identity, cache, segment_size=segment)
        try:
            run_stage(root, "accuracy")
            run_stage(root, "reference")
            acceptance = json.loads((root / "precision.json").read_text())
            seal_accuracy_binaries(root)
            accepted_indices = []
            for index in indices:
                shape = precision_shape(kwargs_list[index], context)
                if acceptance["by_shape"][shape]["passed"]:
                    accepted_indices.append(index)
                else:
                    results[index] = RunnerResult(
                        error=(
                            f"independent full-logit numerical check failed for {shape}; "
                            f"see {root / 'precision.json'}"
                        )
                    )
            if not accepted_indices:
                continue
            run_stage(root, "profile")
            measurements = export_and_measure(root, cache / "cache/neuron/compile_cache")
            for index in accepted_indices:
                spec = kwargs_list[index]
                times = measurements.get(measurement_key(spec))
                if not times:
                    results[index] = RunnerResult(
                        error=f"no native execution at {measurement_key(spec)}; artifacts={root}"
                    )
                    continue
                work = estimate_segmented_work(
                    spec["phase"], spec["token_bucket"], spec["prior_tokens"], context
                )
                duration = statistics.median(times)
                results[index] = RunnerResult(
                    metrics=ComputeMetrics(
                        duration,
                        work["contraction_flops_per_rank"] / (duration * 1e9),
                        work["persistent_operand_bytes_per_rank"] / (duration * 1e6),
                    )
                )
                _PROVENANCE[key(spec)] = (
                    f"stock_segmented_forward; artifacts={root}; samples={len(times)}; "
                    "prefill=batch1_timing_calls_kth_segment_prior_k_times_S; "
                    "work=per_rank_contractions_and_persistent_operands_estimates_not_counters; "
                    "corpus=eight_museum_subjects_each_bucket; "
                    "single_prompt_precision_may_fail_despite_corpus_aggregate_acceptance"
                )
        except Exception as error:
            for index in indices:
                if results[index] is None:
                    results[index] = RunnerResult(error=f"{error}; artifacts={root}")
    return results


def row_provenance(**spec):
    return _PROVENANCE.get(key(spec))
