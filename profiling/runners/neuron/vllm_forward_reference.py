"""Independent CPU eager FP32/BF16 references and unmodified vendor acceptance."""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path


def main(root):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from vllm_neuron.accuracy.constants import (
        DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
        DEFAULT_TOLERANCE_MAP,
    )
    from vllm_neuron.accuracy.logit_validation import (
        DEFAULT_AGGREGATE_CONFIG,
        _compute_aggregate_metrics,
        _validate_single_token_logits,
    )

    torch.set_num_threads(4)
    plan = json.loads((root / "plan.json").read_text())
    rows = json.loads((root / "accuracy-outputs.json").read_text())
    cache = Path(plan["reference_cache"])
    cache.mkdir(exist_ok=True)
    references = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model = None
        for row in rows:
            ids = row["prompt_ids"] + row["token_ids"][:-1]
            contract = {
                "identity": plan["identity"],
                "attention": "eager",
                "dtype": name,
                "ids": ids,
                "positions": 8,
            }
            key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            path, receipt = cache / f"{key}.npy", cache / f"{key}.json"
            if path.exists() and receipt.exists():
                expected = json.loads(receipt.read_text())
                if (
                    expected["contract"] != contract
                    or expected["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise RuntimeError("cached independent reference identity/checksum mismatch")
            else:
                if model is None:
                    model = AutoModelForCausalLM.from_pretrained(
                        plan["model"],
                        dtype=dtype,
                        attn_implementation="eager",
                        local_files_only=True,
                    ).eval()
                with torch.inference_mode():
                    logits = (
                        model(torch.tensor([ids]), use_cache=False, logits_to_keep=8)
                        .logits[0]
                        .float()
                    )
                if logits.shape != (8, 128256) or not torch.isfinite(logits).all():
                    raise RuntimeError("independent reference invalid")
                np.save(path, logits.numpy())
                receipt.write_text(
                    json.dumps(
                        {
                            "contract": contract,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                )
            references[(name, row["case_id"])] = str(path)
        del model
        gc.collect()
    by_shape = {}
    comparison_cache = {}
    for row in rows:
        native_array = np.load(root / f"native-{row['case_id']}.npy")
        key = (
            tuple(row["prompt_ids"] + row["token_ids"]),
            hashlib.sha256(native_array.tobytes()).hexdigest(),
        )
        if key not in comparison_cache:
            native = torch.from_numpy(native_array)
            reference = torch.from_numpy(np.load(references[("bf16", row["case_id"])]))
            fp32 = torch.from_numpy(np.load(references[("fp32", row["case_id"])]))
            if any(
                value.shape != (8, 128256) or not torch.isfinite(value).all()
                for value in (native, reference, fp32)
            ):
                raise RuntimeError("invalid native/reference logit shape or nonfinite values")
            details = []
            for position in range(8):
                _, detail = _validate_single_token_logits(
                    reference[position],
                    native[position],
                    DEFAULT_TOLERANCE_MAP,
                    DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
                    False,
                    actual_token_id=row["token_ids"][position],
                    baseline_logits=fp32[position],
                )
                details.append(detail)
            comparison_cache[key] = (all(detail["passed"] for detail in details), [details])
        by_shape.setdefault(row["shape_id"], []).append(comparison_cache[key])
    summary = {}
    for shape, selected in by_shape.items():
        aggregate = _compute_aggregate_metrics(selected, DEFAULT_AGGREGATE_CONFIG)
        summary[shape] = {
            "passed": bool(
                all(item[0] for item in selected)
                or aggregate["agg_sigma_ratio"] <= 1.0
                or aggregate["agg_bc"] == "PASS"
            ),
            "aggregate": aggregate,
        }
    result = {
        "passed": all(item["passed"] for item in summary.values()),
        "by_shape": summary,
        "criterion": (
            "Unmodified vendor defaults: all static checks OR aggregate RMS ratio<=1 "
            "OR every per-position BC>=0.99; no logit shifts."
        ),
        "reference_cache": str(cache),
        "unique_comparisons": len(comparison_cache),
    }
    (root / "precision.json").write_text(
        json.dumps(result, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
