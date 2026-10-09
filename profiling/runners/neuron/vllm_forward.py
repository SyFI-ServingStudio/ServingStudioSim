"""Stock vLLM Neuron full-forward profiling coordinator (worker only).

Each context group uses one accuracy engine and one production timing engine.
The public serving API owns rank launch and KV allocation; this runner never
chooses cores or changes device visibility. Artifact paths survive worker exit.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import uuid
from pathlib import Path

from profiling.db.args import DType
from profiling.runners.metrics import ComputeMetrics, RunnerResult
from profiling.runners.neuron.vllm_identity import (
    MODEL_HASHES as MODEL_HASHES,
)
from profiling.runners.neuron.vllm_identity import (
    seal_accuracy_binaries,
    verify_checkpoint,
)
from profiling.runners.neuron.vllm_identity import (
    sha256 as sha256,
)

VERSIONS = {
    "vllm": "0.24.0",
    "vllm-neuron": "0.24.0.1.1.0",
    "torch": "2.11.0",
    "transformers": "5.15.0",
    "libtorch-neuronx-lite": "2.11.0.1.0.1284+f49d8626",
    "neuronx-cc": "2.27.5334.0+f702b353",
    "nki": "0.6.0+31049202112.g85070674",
}

_PROVENANCE: dict[str, str] = {}


def key(spec):
    return json.dumps(spec, sort_keys=True, default=str)


def validate_shape(phase, token_bucket, max_model_len, kv_blocks, block_size, tp_size, dtype):
    if DType.from_value(dtype) != DType.BF16:
        raise ValueError("stock Llama forward requires BF16 weights, compute and KV")
    if (kv_blocks, block_size, tp_size) != (6782, 32, 4):
        raise ValueError("verified stock configuration requires 6782 KV blocks, page32 and TP4")
    if max_model_len not in (128, 512, 2048):
        raise ValueError("verified max_model_len buckets are 128, 512 and 2048")
    if phase == "prefill":
        if token_bucket != max_model_len:
            raise ValueError("initial prefill measurements require the full context bucket")
    elif phase == "decode":
        if token_bucket not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            raise ValueError("initial decode buckets are powers of two from1 to512")
        if token_bucket > {128: 128, 512: 512, 2048: 128}[max_model_len]:
            raise ValueError("decode bucket exceeds verified runtime/capacity scope")
    else:
        raise ValueError("phase must be prefill or decode")


def verify_identity(model):
    if os.environ.get("NEURON_VISIBLE_DEVICES") != "0-3":
        raise RuntimeError("initial stock forward capture supports the first TP4/LNC2 chip only")
    observed = {name: importlib.metadata.version(name) for name in VERSIONS}
    if observed != VERSIONS:
        raise RuntimeError(f"stock forward runtime identity mismatch: {observed}")
    hashes = verify_checkpoint(model)
    return {"versions": observed, "model_sha256": hashes}


def run_stage(root, stage, engine="profiling.runners.neuron.vllm_forward_engine"):
    """Run one isolated stage; ``engine`` selects the serving-process entry module."""
    module = "profiling.runners.neuron.vllm_forward_reference" if stage == "reference" else engine
    with (root / f"{stage}.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", module, str(root), stage], stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"{stage} failed ({result.returncode}); see {root / (stage + '.log')}")
    if "events were dropped" in (root / f"{stage}.log").read_text():
        raise RuntimeError(f"native trace dropped events; see {root / (stage + '.log')}")


def write_plan(root, specs, context, model, identity, cache):
    plan = {
        "specs": specs,
        "context": context,
        "model": str(model),
        "identity": identity,
        "reference_cache": str(cache / "forward-reference-cache"),
        "profile_repeats": 3,
    }
    (root / "plan.json").write_text(json.dumps(plan, indent=2, default=str))
    return plan


def precision_shape(spec, context):
    """The validation workload shape whose full logits accept this row."""
    batch = 16 if spec["phase"] == "prefill" else min(spec["token_bucket"], 6782 * 32 // context)
    return f"b{batch}-s{context - 8}"


def profile_forward_batch(kwargs_list):
    results = [None] * len(kwargs_list)
    groups = {}
    for index, spec in enumerate(kwargs_list):
        try:
            validate_shape(**spec)
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
        root = cache / "forward-runs" / uuid.uuid4().hex
        root.mkdir(parents=True)
        write_plan(root, [kwargs_list[i] for i in indices], context, model, identity, cache)
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
            from profiling.runners.neuron.vllm_forward_trace import export_and_measure

            measurements = export_and_measure(root, cache / "cache/neuron/compile_cache")
            for index in accepted_indices:
                spec = kwargs_list[index]
                times = measurements[(spec["phase"], spec["token_bucket"])]
                from profiling.runners.neuron.vllm_forward_work import estimate_work

                work = estimate_work(spec["phase"], spec["token_bucket"], context)
                duration = statistics.median(times)
                metrics = ComputeMetrics(
                    duration,
                    work["contraction_flops_per_rank"] / (duration * 1e9),
                    work["persistent_operand_bytes_per_rank"] / (duration * 1e6),
                )
                results[index] = RunnerResult(metrics=metrics)
                _PROVENANCE[key(spec)] = (
                    f"stock_full_forward; artifacts={root}; samples={len(times)}; "
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
