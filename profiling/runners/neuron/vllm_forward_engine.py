"""Isolated stock public serving process; startup and correctness are untimed."""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

from profiling.runners.neuron.stock_contract import (
    KV_BLOCKS,
    VALIDATION_OUTPUT_TOKENS,
    prefill_validation_bucket,
    prompt_length,
    resident_capacity,
)


def save(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2, default=str))


def validate_greedy_tokens(logits, token_ids):
    """Require selected tokens to be exact maxima of complete finite stored rows.

    BF16 logits can have tied maxima. The engine may select any tied index;
    NumPy's first-index argmax is not the sampler's tie-breaking contract.
    Compare without casting, tolerance or shifting the stored logits.
    """
    import numpy as np

    logits = np.asarray(logits)
    tokens = np.asarray(token_ids)
    if (
        logits.ndim != 2
        or 0 in logits.shape
        or logits.dtype.kind != "f"
        or not np.isfinite(logits).all()
    ):
        raise RuntimeError("native logits must be nonempty complete finite floating rows")
    if (
        tokens.ndim != 1
        or tokens.size != logits.shape[0]
        or tokens.dtype.kind not in "iu"
        or np.any(tokens < 0)
        or np.any(tokens >= logits.shape[1])
    ):
        raise RuntimeError("sampled tokens must provide one integral in-range index per logit row")
    selected = logits[np.arange(logits.shape[0]), tokens]
    if not np.array_equal(selected, logits.max(axis=1)):
        raise RuntimeError("sampled token logit differs from the exact stored row maximum")


def prefill_buckets_of(plan):
    """The prefill inventory the plan's engine compiles: the requested prefill
    buckets plus the context, which vllm-neuron requires as the last entry. A
    segmented plan compiles its one S-token graph."""
    if "segment_size" in plan:
        return [plan["segment_size"]]
    requested = {s["token_bucket"] for s in plan.get("specs", ()) if s["phase"] == "prefill"}
    return sorted(requested | {plan["context"]})


def cases_for_plan(plan, tokenizer):
    context = plan["context"]
    validation = prefill_validation_bucket(context)
    decode = {s["token_bucket"] for s in plan.get("specs", ()) if s["phase"] == "decode"}
    buckets = decode or {validation}
    prefill = {s["token_bucket"] for s in plan.get("specs", ()) if s["phase"] == "prefill"}
    # Prefill graphs below the context get their own shorter prompts; a
    # segmented plan's prefill rows keep the canonical prompts.
    own_prompts = set() if "segment_size" in plan else {t for t in prefill if t < context}
    if prefill - own_prompts:
        buckets.add(validation)  # Frozen full-logit validation workload for the prefill row.
    subjects = (
        "paintings",
        "maps",
        "books",
        "sculptures",
        "photographs",
        "coins",
        "clocks",
        "instruments",
    )
    texts = [
        f"The museum contains {subject}. Visitors read the descriptions "
        "and discuss the history of each object. "
        for subject in subjects
    ]
    # Truncated prefixes are unchanged by the repeat count; 1000 copies fill C8192
    # and context // 8 copies (about 19 tokens each) fill the longer contexts.
    encoded = [tokenizer.encode(text * max(1000, context // 8)) for text in texts]
    capacity = resident_capacity(context)
    # Each workload is (prompt length, decode bucket, prefill_only). Decode
    # buckets run the canonical C-8 prompts. A prefill bucket T < C runs T-8
    # prompts: they pad to T, because the buckets are 128 apart. Those requests
    # time the prefill graph only, because their decodes hold fewer KV tokens
    # than a decode row.
    workloads = [(prompt_length(context), bucket, False) for bucket in sorted(buckets)]
    workloads += [(prompt_length(t), validation, True) for t in sorted(own_prompts)]
    cases = []
    for length, bucket, prefill_only in workloads:
        prompts = [ids[:length] for ids in encoded]
        if any(len(ids) != length for ids in prompts):
            raise ValueError("prompt generator did not reach the declared prompt length")
        batch = min(bucket, capacity)
        shape = f"b{batch}-s{length}"
        # Every bucket covers the same prespecified eight subjects. Small
        # batches require several public calls, aggregated by configuration.
        for start in range(0, max(8, batch), batch):
            cases.append(
                {
                    "id": f"{shape}-subject{start}",
                    "shape_id": shape,
                    "bucket": bucket,
                    "prompts": [prompts[i % len(prompts)] for i in range(start, start + batch)],
                }
                | ({"prefill_only": True} if prefill_only else {})
            )
    return cases


def seqs_buckets(plan, cases):
    """Compiled decode graphs: the cases' buckets, plus the stock bucket 1 unless the
    requested decode inventory omits it (C4096 serves without its unaccepted B1 graph,
    which would otherwise run for interleaved single decodes)."""
    buckets = {case["bucket"] for case in cases}
    decode = {s["token_bucket"] for s in plan.get("specs", ()) if s["phase"] == "decode"}
    if not decode or 1 in decode:
        buckets.add(1)
    return sorted(buckets)


def segment_timing_cases(plan, cases):
    """Batch-1 prefill timing calls of a segmented plan, one per validation subject.

    A lone request is the only sequence the scheduler can prefill, so the k-th
    prefill execution of the call is its k-th segment (prior = k x segment).
    Each call reuses the subject's prompt from the prefill validation shape and
    must reproduce that accepted request's tokens. Whole-forward plans get none.
    """
    if not plan.get("segment_size") or not any(s["phase"] == "prefill" for s in plan["specs"]):
        return []
    bucket = prefill_validation_bucket(plan["context"])
    validation = [case for case in cases if case["bucket"] == bucket]
    slots = [(case, slot) for case in validation for slot in range(len(case["prompts"]))]
    smallest = min(seqs_buckets(plan, cases))
    timing = []
    for subject in range(8):
        case, slot = slots[subject]
        timing.append(
            {
                "id": f"segment-timing-subject{subject}",
                "shape_id": case["shape_id"],
                # A batch-1 decode pads to the smallest compiled graph.
                "bucket": smallest,
                "prompts": [case["prompts"][slot]],
                "expected_case_id": f"{case['id']}-slot{slot}",
                "segment_timing": True,
            }
        )
    return timing


def engine_config(plan, cases, root, mode):
    """Public ``LLM`` arguments; ``cases`` is the full (unfiltered) canonical workload.

    The prefill inventory is ``prefill_buckets_of(plan)``. A plan with
    ``segment_size`` S serves the same context with the stock prefill
    segmentation: S-token budgets, S-token batched-token and KV-segment buckets.
    """
    tokens = plan.get("segment_size", plan["context"])
    nc = {
        "num_batched_tokens_buckets": prefill_buckets_of(plan),
        "num_seqs_buckets": seqs_buckets(plan, cases),
    }
    if "segment_size" in plan:
        nc["kv_segment_size_buckets"] = [tokens]
    extra = {"neuron_config": nc}
    if mode == "accuracy":
        nc["debug_logits_dir"] = str(root / "raw-logits")
    else:
        extra["neuron_profiler"] = {
            "output_dir": str(root / "profiles"),
            "activities": ["system_profile"],
            "neuron_cores": [0, 1, 2, 3],
            "sys_trace_max_events_per_nc": 4000000,
        }
    config = dict(
        model=plan["model"],
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_num_seqs=max(nc["num_seqs_buckets"]),
        max_model_len=plan["context"],
        max_num_batched_tokens=tokens,
        enable_prefix_caching=False,
        seed=0,
        additional_config=extra,
        num_gpu_blocks_override=KV_BLOCKS,
    )
    if mode == "profile":
        config["profiler_config"] = {"profiler": "cuda"}
    return config


def profile_cases(plan, cases, timing, precision):
    """Numerically accepted calls for the timing stage.

    Segmented plans time prefill only from the batch-1 ``timing`` calls, so the
    canonical calls run only when decode rows need them.
    """
    if timing and not any(s["phase"] == "decode" for s in plan["specs"]):
        cases = []
    selected = [
        case for case in cases + timing if precision["by_shape"][case["shape_id"]]["passed"]
    ]
    if not selected:
        raise ValueError("no numerically accepted cases to profile")
    return selected


def main(root, mode):
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    plan = json.loads((root / "plan.json").read_text())
    cases = cases_for_plan(
        plan, AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
    )
    save(root, "cases.json", cases)
    timing = segment_timing_cases(plan, cases)
    if timing:
        save(root, "timing-cases.json", timing)
    config = engine_config(plan, cases, root, mode)
    if mode == "profile":
        precision = json.loads((root / "precision.json").read_text())
        cases = profile_cases(plan, cases, timing, precision)
    save(root, f"{mode}-config.json", config)
    from profiling.runners.neuron.vllm_identity import (
        accuracy_source_identity,
        compile_cache_snapshot,
        stock_compile_cache_dir,
    )

    if mode == "accuracy":
        cache = stock_compile_cache_dir()
        save(root, "accuracy-source-before.json", accuracy_source_identity())
        save(root, "accuracy-binaries-before-load.json", compile_cache_snapshot(cache))
    llm = LLM(**config)
    if mode == "accuracy":
        save(root, "accuracy-binaries-before.json", compile_cache_snapshot(cache))

    def generate(case):
        start = time.time_ns()
        outputs = llm.generate(
            [{"prompt_token_ids": ids} for ids in case["prompts"]],
            SamplingParams(temperature=0, max_tokens=VALIDATION_OUTPUT_TOKENS, ignore_eos=True),
            use_tqdm=False,
        )
        result = {
            "shape_id": case["shape_id"],
            "batch": len(outputs),
            "bucket": case["bucket"],
            "start_epoch_ns": start,
            "stop_epoch_ns": time.time_ns(),
            "requests": [],
        }
        for flag in ("segment_timing", "prefill_only"):
            if case.get(flag):
                result[flag] = True
        for slot, output in enumerate(outputs):
            tokens = output.outputs[0].token_ids
            if (
                output.prompt_token_ids != case["prompts"][slot]
                or len(tokens) != VALIDATION_OUTPUT_TOKENS
            ):
                raise RuntimeError("public engine output violates requested history")
            result["requests"].append(
                {
                    "case_id": f"{case['id']}-slot{slot}",
                    "shape_id": case["shape_id"],
                    "request_id": output.request_id,
                    "prompt_ids": output.prompt_token_ids,
                    "token_ids": tokens,
                }
            )
        return result

    if mode == "accuracy":
        rows = []
        for case in cases:
            raw = root / "raw-logits"
            before = set(raw.glob("*.bin"))
            batch = generate(case)
            # This is an acquisition receipt, including a case that later fails
            # logit validation. The failed stage prevents reference/profile work.
            rows.extend(batch["requests"])
            save(root, "accuracy-outputs.json", rows)
            by_request = {}
            for path in set(raw.glob("*.bin")) - before:
                data = path.read_bytes()
                count, vocab = struct.unpack_from("qq", data)
                if vocab != 128256:
                    raise RuntimeError("unexpected native vocabulary")
                offset = 16
                for _ in range(count):
                    request, position = struct.unpack_from("qq", data, offset)
                    offset += 16
                    logits = np.frombuffer(data, dtype="<f4", count=vocab, offset=offset).copy()
                    offset += vocab * 4
                    by_request.setdefault(request, []).append((position, logits))
                if offset != len(data):
                    raise RuntimeError("native logit file length mismatch")
            if len(by_request) != len(case["prompts"]):
                raise RuntimeError("native logit request coverage mismatch")
            for row in batch["requests"]:
                items = sorted(by_request[int(row["request_id"])], key=lambda item: item[0])
                length = len(row["prompt_ids"])
                if [item[0] for item in items] != list(range(length - 1, length + 7)):
                    raise RuntimeError(
                        "native logits do not cover the eight exact output positions"
                    )
                logits = np.stack([item[1] for item in items])
                np.save(root / f"native-{row['case_id']}.npy", logits)
                validate_greedy_tokens(logits, row["token_ids"])
        save(root, "accuracy-binaries-after.json", compile_cache_snapshot(cache))
        save(root, "accuracy-source-after.json", accuracy_source_identity())
    else:
        expected = {
            row["case_id"]: row["token_ids"]
            for row in json.loads((root / "accuracy-outputs.json").read_text())
        }

        def checked(case):
            row = generate(case)
            if any(
                request["token_ids"] != expected[case.get("expected_case_id", request["case_id"])]
                for request in row["requests"]
            ):
                raise RuntimeError("production sampling differs from numerical validation")
            return row

        # Warm every requested shape; compiler warmup alone is insufficient.
        save(root, "warmup.json", [checked(case) for case in cases])
        captures = []
        llm.start_profile()
        try:
            for case in cases:
                for _ in range(plan["profile_repeats"]):
                    captures.append(checked(case))
                    save(root, "profiled-outputs.json", captures)
        finally:
            llm.stop_profile()
    print("FORWARD_STAGE_COMPLETE", mode, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
