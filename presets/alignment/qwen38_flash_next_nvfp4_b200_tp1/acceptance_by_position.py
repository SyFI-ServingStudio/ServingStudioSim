"""Write acceptance_by_position.json: MTP acceptance by output position, from each case's workload pass.

The estimator is `launcher.alignment_campaign.calibration.buckets_from_metrics`, the one a case's
`speculative_acceptance_by_output_position: {status: from_capture}` uses: per bucket of `output_tokens_before` at
its EDGES, the per-verify-round distribution of accepted draft tokens 0..K over every `decode_request_progress` row
of the warm workload pass; a bucket under MIN_ROUNDS rounds merges into the previous one. make_pack.py reads the
JSON into each case's `speculative_acceptance_by_output_position`.

Run from the repo root:
  uv run python presets/alignment/qwen38_flash_next_nvfp4_b200_tp1/acceptance_by_position.py <cases dir>
where <cases dir> holds `<NN>_<name>/profile_workload/vllm/*_metrics.jsonl` (wt-qf-capture/logs/20261013_2_qf_cases).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # the repo root, for `launcher`
from launcher.alignment_campaign.calibration import buckets_from_metrics  # noqa: E402

DRAFT = 3
EVIDENCE_ROOT = "logs/20261013_2_qf_cases"


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
