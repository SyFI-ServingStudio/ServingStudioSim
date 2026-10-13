"""Write acceptance_by_position.json: MTP acceptance by output position, from each case's workload pass.

Same estimator as the GLM-5.3 DFlash2 pack (glm53_nvfp4_b200_dflash2, `position_accept.buckets_from_metrics`): per
bucket of `output_tokens_before` at EDGES, the per-verify-round distribution of accepted draft tokens 0..K over every
`decode_request_progress` row of the warm workload pass; a bucket under MIN_ROUNDS rounds merges into the previous
one. make_pack.py reads the JSON into each case's `speculative_acceptance_by_output_position`.

Run from the repo root:
  uv run python presets/alignment/qwen38_flash_next_nvfp4_b200_tp1/acceptance_by_position.py <cases dir>
where <cases dir> holds `<NN>_<name>/profile_workload/vllm/*_metrics.jsonl` (wt-qf-capture/logs/20261013_2_qf_cases).
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = 3
EDGES = [1, 32, 128, 512, 1024, 2048, 4096, 8192]  # bucket lower bounds
MIN_ROUNDS = 50
EVIDENCE_ROOT = "logs/20261013_2_qf_cases"


def buckets_from_metrics(path: Path, K: int) -> list[tuple[int, int | None, list[float]]]:
    counts = [Counter() for _ in EDGES]
    for line in open(path):
        for p in json.loads(line)["decode_request_progress"]:
            pos, a = p.get("output_tokens_before"), p.get("accepted_draft_tokens")
            if pos is None or a is None:
                continue
            b = max(i for i, lo in enumerate(EDGES) if pos >= lo) if pos >= 1 else 0
            counts[b][min(a, K)] += 1
    # Merge sparse buckets into the previous one; drop empty tails.
    merged: list[list] = []
    for i, c in enumerate(counts):
        n = sum(c.values())
        if n == 0:
            continue
        if merged and n < MIN_ROUNDS:
            merged[-1][1] += c
            continue
        merged.append([EDGES[i], Counter(c)])
    if len(merged) > 1 and sum(merged[0][1].values()) < MIN_ROUNDS:
        merged[1][1] += merged[0][1]
        merged[1][0] = merged[0][0]
        merged.pop(0)
    out = []
    for j, (lo, c) in enumerate(merged):
        hi = merged[j + 1][0] if j + 1 < len(merged) else None
        n = sum(c.values())
        out.append((lo, hi, [c[a] / n for a in range(K + 1)]))
    return out


def main() -> None:
    cases_dir = Path(sys.argv[1])
    result = {}
    for case_dir in sorted(p for p in cases_dir.iterdir() if p.is_dir() and p.name[:2].isdigit()):
        files = sorted((case_dir / "profile_workload/vllm").glob("*_workload_metrics.jsonl"))
        assert len(files) == 1, (case_dir, files)
        buckets = buckets_from_metrics(files[0], DRAFT)
        result[str(int(case_dir.name[:2]))] = {
            "buckets": [{"from": lo, "p": p} for lo, _hi, p in buckets],
            "evidence": f"{EVIDENCE_ROOT}/{case_dir.name}/profile_workload/vllm/{files[0].name}",
        }
        print(case_dir.name, " ".join(
            f"{lo}:{sum((a + 1) * x for a, x in enumerate(p)):.2f}" for lo, _hi, p in buckets))
    (HERE / "acceptance_by_position.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
