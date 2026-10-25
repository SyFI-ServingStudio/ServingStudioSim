"""tools/trn2-config-search: enumeration, traces, preset rendering, metrics.

CPU tier, except the one test that expands a rendered preset through the
launcher's own validator (needs the built schema).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "trn2-config-search"))

from trn2_search import report as rp  # noqa: E402
from trn2_search.collect import goodput, percentile  # noqa: E402
from trn2_search.presets import member_name, render, split_by_keyset  # noqa: E402
from trn2_search.space import (  # noqa: E402
    enumerate_layouts,
    expand_stage,
    parse_layout,
    structural_errors,
)
from trn2_search.workloads import Workload, from_spec, trace_path  # noqa: E402


def test_layouts_follow_the_arch_rules():
    lays = enumerate_layouts((32,))
    assert "dp4-tp8-g8-ep32-mt1" in lays
    assert "dp2-tp4-g16-ep16-mt2" in lays
    for lay in lays:
        p = parse_layout(lay)
        g = max(p["attn_tp"], p["gdn_tp"])
        assert p["attn_dp"] * g == 32 == p["moe_ep"] * p["moe_tp"]
        assert 24 % p["attn_tp"] == 0  # no tp16 QSA
        assert structural_errors({"family": "unified", **p}) == []


def test_structural_errors_catch_bad_layouts():
    assert structural_errors({"attn_dp": 4, "attn_tp": 8, "moe_ep": 16})  # ep x tp != ranks
    assert structural_errors({"attn_dp": 2, "attn_tp": 16, "moe_ep": 32})  # 24 heads / 16
    assert structural_errors(
        {"attn_dp": 4, "attn_tp": 8, "moe_ep": 16, "moe_tp": 2, "moe_dispatch": "all_to_all"}
    )
    assert structural_errors({"attn_dp": 4, "attn_tp": 8, "moe_ep": 32, "replicas": 3})  # 96 cores
    assert structural_errors({"attn_dp": 4, "attn_tp": 8, "moe_ep": 32, "draft_tokens": 2})
    pd = {
        "family": "pd",
        "prefill.attn_dp": 4,
        "prefill.attn_tp": 8,
        "prefill.moe_ep": 32,
        "decode.attn_dp": 4,
        "decode.attn_tp": 8,
        "decode.moe_ep": 32,
    }
    assert structural_errors(pd) == []
    assert structural_errors({**pd, "decode.replicas": 2})  # 96 cores


def test_expand_stage_crosses_grid_varies_oat_and_dedupes():
    stage = {
        "family": "unified",
        "base": {"layout": "dp4-tp8-g8-ep32-mt1", "max_batch_tokens": 2048},
        "grid": {"tp_collective": ["all_reduce", "reduce_scatter_all_gather"]},
        # 2048 repeats the base point: deduped.
        "oat": {"max_batch_tokens": [1024, 2048]},
        "scenarios": {"base": {}, "half": {"launch_coll_ms": 0.625}},
    }
    cfgs = expand_stage(stage)
    assert len(cfgs) == 2 * 2 * 2  # grid x (base, 1024) x scenarios
    assert all(c["replicas"] == 2 and c["attn_dp"] == 4 for c in cfgs)
    assert len({c["label"] for c in cfgs}) == len(cfgs)
    # A config and its what-if share a core id.
    base = {c["core_id"] for c in cfgs if c["scenario"] == "base"}
    half = {c["core_id"] for c in cfgs if c["scenario"] == "half"}
    assert base == half and len(base) == 4


def test_workloads_are_seeded_and_in_range(tmp_path):
    chat = Workload(
        "c", "chat", 400, {"input_len": [1000, 2000], "output_len": [200, 400], "rate": 2.0}
    )
    a, b = chat.rows(), chat.rows()
    assert a == b
    assert all(1000 <= i <= 2000 and 200 <= o <= 400 for i, o, _ in a)
    span_s = a[-1][2] / 1000.0
    assert 399 / span_s == pytest.approx(2.0, rel=0.2)
    ag = from_spec(
        {
            "name": "ag",
            "kind": "agentic",
            "requests": 60,
            "input_len": [16000, 32000],
            "output_len": [200, 500],
            "burst_rate": 0.05,
            "burst_mean": 6,
        }
    )
    rows = ag.rows()
    assert len(rows) == 60 and [r[2] for r in rows] == sorted(r[2] for r in rows)
    gaps = [rows[i + 1][2] - rows[i][2] for i in range(59)]
    assert sum(g < 1000 for g in gaps) > 30  # bursty: most requests arrive with a burst
    path = trace_path(ag, tmp_path, accept_rate=0.7)
    head, first = path.read_text().splitlines()[:2]
    assert head == "id,input_len,output_len,arrival_time,accept_rate" and first.endswith(",0.7")


def test_render_puts_varying_knobs_in_the_compound_group():
    cfgs = expand_stage(
        {
            "family": "mtp",
            "base": {"layout": "dp4-tp8-g8-ep32-mt1"},
            "grid": {"draft_tokens": [1, 3]},
        }
    )
    ((fam, group),) = split_by_keyset(cfgs)
    p = render(fam, group, [("w", "t.csv")], "logs/x")
    arch = p["pools"]["main"]["groups"][0]["arch"]
    worker = p["pools"]["main"]["groups"][0]["worker"]
    assert arch["type"] == "qwen4_exp_neuron_bf16_speculative"
    assert arch["draft_tokens"] == worker["draft_tokens"] == f"${{{member_name('draft_tokens')}}}"
    assert arch["attn_dp"] == 4  # constant -> literal
    assert p["workload"]["request_rate"] == 1.0
    assert p["workload"]["input_file_tags"] == ["speculative"]
    assert sorted(p["compound"]["config"]) == sorted(c["label"] for c in cfgs)
    assert p["io"]["log_dir"] == "logs/x/{config}/{workload}"


def test_render_pd_splits_knobs_by_pool():
    cfg = {
        "family": "pd",
        "prefill.attn_dp": 4,
        "prefill.attn_tp": 8,
        "prefill.moe_ep": 32,
        "decode.attn_dp": 8,
        "decode.attn_tp": 4,
        "decode.moe_ep": 32,
        "label": "x",
        "scenario": "base",
        "core_id": "c",
    }
    p = render("pd", [cfg], [("w", "t.csv")], "logs/x")
    assert p["deployment"] == "pd"
    assert p["pools"]["decode"]["groups"][0]["worker"]["type"] == "pd_decode"
    assert p["compound"]["config"]["x"][member_name("decode.attn_tp")] == 4


@pytest.mark.needs_binary
def test_rendered_preset_passes_the_launcher_validator():
    from launcher.__main__ import expand_preset
    from launcher.schema.loader import load_schema

    try:
        schema = load_schema("release")
    except Exception:  # noqa: BLE001 - no built schema on this host
        pytest.skip("no release deployment schema")
    cfgs = expand_stage(
        {
            "family": "unified",
            "base": {"layout": "dp4-tp8-g8-ep32-mt1"},
            "grid": {
                "region_granularity": ["mixer", "layer"],
                "kind_time_scale": [[], ["moe_block@256=0.5"]],
            },
        }
    )
    ((fam, group),) = split_by_keyset(cfgs)
    preset = render(
        fam,
        group,
        [
            ("w1", "presets/trn2/trn2_fixed_2k_512_c64.csv"),
            ("w2", "presets/trn2/trn2_fixed_2k_512_c64.csv"),
        ],
        "logs/x",
    )
    runs = expand_preset(preset, schema, "x.yaml")
    assert len(runs) == 4 * 2
    assert len({r["io"]["log_dir"] for r in runs}) == 8


def test_percentile_and_goodput():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    reqs = [
        {"completed": True, "ttft_ms": 100.0, "tpot_mean_ms": 50.0, "num_output_tokens": 10},
        {"completed": True, "ttft_ms": 900.0, "tpot_mean_ms": 50.0, "num_output_tokens": 10},
        {"completed": True, "ttft_ms": 100.0, "tpot_mean_ms": None, "num_output_tokens": 1},
        {"completed": False, "ttft_ms": None, "tpot_mean_ms": None, "num_output_tokens": 0},
    ]
    g = goodput(reqs, 2000.0, ttft_ms=500.0, tpot_ms=60.0)
    assert g == {"goodput_req_s": 1.0, "goodput_tok_s": 5.5, "slo_attain": 0.5}


def test_pareto_sensitivity_and_rank_agreement():
    rows = [
        {"x": 10, "y": 5, "cfg.k": "a"},
        {"x": 8, "y": 3, "cfg.k": "b"},
        {"x": 7, "y": 4, "cfg.k": "b"},  # dominated by (8, 3)
        {"x": 12, "y": 9, "cfg.k": "a"},
    ]
    assert [(r["x"], r["y"]) for r in rp.pareto(rows, "x", "y")] == [(8, 3), (10, 5), (12, 9)]
    (s,) = rp.sensitivity(rows, "x")
    assert s["knob"] == "k" and s["best_value"] == "'a'" and s["spread"] == pytest.approx(4 / 12)
    assert rp.kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert rp.kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0


def test_seeds_replace_the_bare_base_and_pd_scenarios_reach_both_pools():
    seeds = [{"attn_dp": 4, "attn_tp": 8, "moe_ep": 32}]
    cfgs = expand_stage({"family": "unified", "seeds": seeds})
    assert len(cfgs) == 1 and cfgs[0]["attn_dp"] == 4  # no layout-less base point
    pd = expand_stage(
        {
            "family": "pd",
            "grid": {
                "prefill.layout": ["dp4-tp8-g8-ep32-mt1"],
                "decode.layout": ["dp8-tp4-g4-ep32-mt1"],
            },
            "scenarios": {"half": {"launch_coll_ms": 0.625}},
        }
    )
    (c,) = pd
    assert c["prefill.launch_coll_ms"] == c["decode.launch_coll_ms"] == 0.625
    assert "launch_coll_ms" not in c
