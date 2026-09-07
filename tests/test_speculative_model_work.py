"""Spec5 floors count executed candidates, including subsequently rejected ones."""

import json
from pathlib import Path

import pytest

from model.work import floors
from model.work.registry import load_model
from model.work.speculative import aggregate_workload

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "model/config/glm52_nvfp4.json"


def totals(*, depth=5, prefill=((0, 2),), decode=((106, 6), (2053, 6)), copies=1):
    geometry = {"draft_tokens": depth, "max_model_len": 8192, "prefill": prefill, "decode": decode}
    return {
        "matmul_tokens": (sum(q for _, q in prefill) + sum(q for _, q in decode)) * copies,
        "prefill_tokens": sum(q for _, q in prefill) * copies,
        "prefill_requests": len(prefill) * copies,
        "prefill_stateful_requests": sum(p > 0 for p, _ in prefill) * copies,
        "prefill_pairs": sum(q * p + q * (q + 1) / 2 for p, q in prefill) * copies,
        "prefill_cached": sum(p for p, _ in prefill) * copies,
        "decode_passes": len(decode) * copies,
        "decode_kv": sum(kv - q for kv, q in decode) * copies,
        "speculative_geometry": {json.dumps(geometry): copies},
    }


def write_params(tmp_path, *, mode="index_share", depth=5):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/params.json").write_text(
        json.dumps(
            {
                "pools": {
                    "main": {
                        "groups": [
                            {
                                "gpu": "NVIDIA B200",
                                "arch": {
                                    "type": "glm52_vllm_nvfp4_dsa_moe_speculative",
                                    "model_config": str(CONFIG),
                                    "draft_tokens": depth,
                                    "mtp_mode": mode,
                                },
                            }
                        ]
                    }
                }
            }
        )
    )


def test_q6_and_recurrent_attention_have_independent_causal_goldens():
    workload = aggregate_workload(totals())
    assert workload.matmul_tokens == 14
    assert workload.head_positions == 13
    assert workload.num_attention_steps == 3
    assert workload.stages["mtp_first"].head_positions == 3
    assert workload.stages["mtp_recurrent"].matmul_tokens == 12
    segments = {s.name: s for s in load_model(CONFIG).label(workload).segments}
    # Query contexts are 101..106 and 2048..2053. The second request
    # contributes six capped 2048-key sparse rows, not min(2048, sum(context)).
    assert segments["dense_full_index.indexer.decode"].flops_total == 3 * 8192 * 12924
    assert segments["dense_full_index.attn.decode"].flops_total == 3 * 2 * 64 * 1088 * 12909
    # Draft endpoint contexts: 3..6, 107..110, 2054..2057.
    assert segments["mtp_recurrent.attn.decode"].flops_total == 2 * 64 * 1088 * 8644
    assert segments["mtp_first.lm_head"].flops_total == 2 * 3 * 6144 * 154880
    assert segments["mtp_recurrent.lm_head"].flops_total == 2 * 12 * 6144 * 154880
    # All target q6 rows remain necessary even when their candidates are rejected.
    assert segments["lm_head"].flops_total == 2 * 13 * 6144 * 154880


@pytest.mark.parametrize(
    "mode,depth,map_name",
    [
        ("index_share", 5, "index_share"),
        ("full_index", 5, "full_index"),
        ("index_share", 1, "single_draft"),
    ],
)
@pytest.mark.parametrize("phase", ["prefill", "decode", "mixed"])
def test_floor_handoff_and_exact_semantic_coverage(tmp_path, mode, depth, map_name, phase):
    write_params(tmp_path, mode=mode, depth=depth)
    shape = totals(
        depth=depth,
        prefill=() if phase == "decode" else ((0, 2),),
        decode=() if phase == "prefill" else ((106, depth + 1),),
    )
    result = floors.compute_floors(tmp_path, {"main/0": shape})["main/0"]
    assert "error" not in result, result
    assert result["segmented"] >= result["necessary"] > 0
    mapping = json.loads(
        (ROOT / f"model/work/location_maps/glm52_speculative_{map_name}.json").read_text()
    )
    mapped = [s for row in mapping["locations"] for s in row["semantics"]]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == {s["name"] for s in result["segments"]}
    locked = floors.compute_locked_compositions(
        tmp_path, {"main/0": [{"occurrences": 3, "totals": shape}]}
    )["main/0"]
    assert "error" not in locked, locked
    assert locked["necessary"] == pytest.approx(3 * result["necessary"])
    assert locked["segmented"] == pytest.approx(3 * result["segmented"])


def test_histogram_replication_preserves_sparse_saturation_and_stages():
    model = load_model(CONFIG)
    single = model.label(aggregate_workload(totals()))
    repeated = model.label(aggregate_workload(totals(copies=1000)))
    assert repeated.flops_total == pytest.approx(1000 * single.flops_total)
    assert repeated.bytes["kv"] == pytest.approx(1000 * single.bytes["kv"])


def test_geometry_cannot_silently_replace_missing_or_different_work():
    broken = totals()
    broken["matmul_tokens"] -= 1
    with pytest.raises(ValueError, match="disagrees with matmul_tokens"):
        aggregate_workload(broken)
    with pytest.raises(ValueError, match="decode query"):
        aggregate_workload(totals(decode=((100, 1),)))


def test_logged_depth_must_match_deployment(tmp_path):
    write_params(tmp_path, depth=3)
    result = floors.compute_floors(tmp_path, {"main": totals()})["main"]
    assert "draft depth disagrees" in result["error"]
