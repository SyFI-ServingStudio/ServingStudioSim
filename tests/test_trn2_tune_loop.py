"""tools/trn2-tune-loop: knob translation, proposal, prediction bookkeeping, the
engine run (a stub `slice_run.sh` stands in for the device) and the comparison.
CPU tier: nothing simulates and nothing touches a Neuron device."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "trn2-tune-loop"))

import tune_loop as tl  # noqa: E402

BEST = {
    "tp_collective": "reduce_scatter_all_gather",
    "moe_dispatch": "all_gather",
    "region_granularity": "block",
    "attn_dp": 8,
    "attn_tp": 4,
    "gdn_tp": 4,
    "moe_ep": 32,
    "moe_tp": 1,
    "draft_tokens": 1,
    "max_batch_tokens": 512,
    "replicas": 2,
}


def _row(label, tok_s, workload="fixed_2k_512-a0.67", scenario="base", **cfg):
    r = {f"cfg.{k}": v for k, v in cfg.items()}
    r.update(
        {
            "label": label,
            "workload": workload,
            "workload_base": workload.split("-a")[0],
            "cfg.scenario": scenario,
            "cfg.family": "mtp",
            "cfg.accept_rate": 0.67,
            "total_tok_s": tok_s,
            "ttft_p50": 1.0,
            "tpot_p50": 2.0,
        }
    )
    return r


@pytest.fixture
def search_exp(tmp_path):
    exp = tmp_path / "s3_mtp"
    exp.mkdir()
    rows = [
        # Best, but the engine cannot run it (GDN TP != QSA TP).
        _row("x_g16", 4000, **dict(BEST, attn_dp=2, gdn_tp=16)),
        _row("x_best", 3000, **BEST),
        _row("x_whatif", 9999, scenario="no_pad", **BEST),
        _row("x_second", 2500, **dict(BEST, tp_collective="all_reduce")),
        _row("x_other_wl", 5000, workload="chat_r1-a0.67", **BEST),
    ]
    (exp / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return exp


def test_engine_knobs_fill_the_engines_defaults():
    k = tl.engine_knobs(dict(BEST, family="mtp", label="x", scenario="base"))
    assert "family" not in k and "label" not in k
    assert k["prefill_buckets"] == [128, 512]
    assert k["decode_buckets"] == [1, 2, 4, 8, 16] and k["context_buckets"] == [2048, 4096]
    assert tl.engine_knobs(dict(BEST, max_batch_tokens=4096))["prefill_buckets"] == [128, 512, 2048]
    assert tl.engine_blockers(k) == []


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"spmd_padding": False}, "spmd_padding"),
        ({"step_forwards": "fused"}, "step_forwards"),
        ({"kind_time_scale": ["moe_block@256=0.33"]}, "kind_time_scale"),
        ({"max_prefill_chunks_per_partition": 16}, "max_prefill_chunks"),
        ({"gdn_tp": 16, "attn_dp": 2}, "attn_tp must equal gdn_tp"),
        ({"moe_ep": 16, "moe_tp": 2}, "EP32"),
        ({"moe_routed_impls": ["nki"]}, "moe_routed_impls"),
        ({"attn_dp": 4}, "32 cores"),
    ],
)
def test_sim_only_and_unsupported_knobs_are_refused(over, needle):
    why = tl.engine_blockers(tl.engine_knobs(dict(BEST, **over)))
    assert any(needle in w for w in why), why


def test_chunk_budget_above_the_bucket_is_a_sim_mismatch():
    wl = tl.WORKLOADS["w64"]
    assert tl.sim_mismatches(tl.engine_knobs(dict(BEST, max_batch_tokens=1024)), wl)
    # Budget above the largest bucket but every prompt fits one bucket: modeled.
    assert tl.sim_mismatches(tl.engine_knobs(dict(BEST, max_batch_tokens=4096)), wl) == []
    assert tl.sim_mismatches(tl.engine_knobs(BEST), wl) == []


def test_parse_override_keeps_lists_whole():
    label, k = tl.parse_override("b:max_batch_tokens=4096,prefill_buckets=[128,512],moe_dispatch=x")
    assert label == "b"
    assert k == {"max_batch_tokens": 4096, "prefill_buckets": [128, 512], "moe_dispatch": "x"}


def test_propose_picks_runnable_top_variants_and_contrast(tmp_path, search_exp):
    loop = tl.Loop("t", root=tmp_path / "tune")
    cands = tl.propose(
        loop,
        [search_exp],
        "fixed_2k_512-a0.67",
        top=2,
        variants=["c4096:max_batch_tokens=4096"],
        contrasts=["a2a:moe_dispatch=all_to_all"],
        explicit=[],
    )
    assert [c["label"] for c in cands] == ["top1", "top2", "c4096", "a2a"]
    assert cands[0]["search"]["label"] == "x_best"  # what-if and blocked rows skipped
    assert cands[1]["knobs"]["tp_collective"] == "all_reduce"
    assert cands[2]["knobs"]["prefill_buckets"] == [128, 512, 2048]
    assert cands[3]["role"] == "contrast" and cands[3]["knobs"]["moe_dispatch"] == "all_to_all"
    assert all(c["accept_rate"] == 0.67 and c["family"] == "mtp" for c in cands)
    prop = loop.read("proposal.json")
    assert prop["skipped"][0]["label"] == "x_g16"
    assert loop.done("propose")
    with pytest.raises(ValueError, match="Sim does not cap"):
        tl.propose(loop, [search_exp], "fixed_2k_512-a0.67", 1, ["b:max_batch_tokens=1024"], [], [])


def test_render_preset_pins_the_engine_knobs(tmp_path, search_exp):
    loop = tl.Loop("t", root=tmp_path / "tune")
    (cand,) = tl.propose(loop, [search_exp], "fixed_2k_512-a0.67", 1, [], [], [])
    preset = json.loads(tl.render_preset(cand, tmp_path / "p").read_text())
    g = preset["pools"]["main"]["groups"][0]
    assert g["replicas"] == 2 and preset["pools"]["main"]["placement"] == "round-robin"
    assert g["arch"]["type"] == "qwen4_exp_neuron_bf16_speculative"
    assert g["arch"]["attn_tp"] == 4 and g["arch"]["prefill_buckets"] == [128, 512]
    assert g["worker"]["max_batch_tokens"] == 512 and g["worker"]["draft_tokens"] == 1
    assert g["worker"]["max_prefill_chunks_per_partition"] == 1
    assert preset["workload"]["request_rate"] == 1.0
    trace = (tmp_path / "p" / "trace.csv").read_text().splitlines()
    assert trace[0].endswith("accept_rate") and len(trace) == 65
    assert trace[2] == "1,2048,512,50.000000,0.67"


def _fake_sim_run(run_dir: Path, duration_s: float, ttft_ms: float, tpot_ms: float):
    import pyarrow as pa
    import pyarrow.parquet as pq

    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"sim_ms": duration_s * 1e3}))
    rows = [
        dict(completed=True, num_output_tokens=512, ttft_ms=ttft_ms, tpot_mean_ms=tpot_ms)
        for _ in range(64)
    ]
    pq.write_table(pa.Table.from_pylist(rows), run_dir / "raw" / "request_slo.parquet")


PRED = {"top1": (50.0, 15000.0, 70.0), "a2a": (60.0, 20000.0, 80.0)}


def _loop_with_predictions(tmp_path, search_exp):
    loop = tl.Loop("t", root=tmp_path / "tune")
    tl.propose(
        loop,
        [search_exp],
        "fixed_2k_512-a0.67",
        1,
        [],
        ["a2a:moe_dispatch=all_to_all"],
        [],
    )
    calls = []

    def simulate(preset: Path, analyze: bool):
        label = preset.parent.name
        calls.append(label)
        _fake_sim_run(preset.parent / "run", *PRED[label])

    preds = tl.predict(loop, analyze=False, simulate=simulate)
    return loop, preds, calls


def test_predict_writes_and_resumes(tmp_path, search_exp):
    loop, preds, calls = _loop_with_predictions(tmp_path, search_exp)
    assert calls == ["top1", "a2a"]
    p = preds["candidates"]["top1"]["predicted"]
    assert p["duration_s"] == 50.0 and p["out_tok_s"] == pytest.approx(64 * 512 / 50.0)
    assert p["ttft_p50_s"] == 15.0 and p["tpot_p50_ms"] == 70.0
    assert any("acceptance" in u for u in preds["inputs"]["uncertain"])
    again = []
    tl.predict(loop, analyze=False, simulate=lambda pr, a: again.append(pr))
    assert again == []  # resumed: nothing re-simulated


STUB = r"""#!/bin/bash
# stand-in for scripts/slice_run.sh: writes the serve.py timeline the real run would
port=$1; log=$2; shift 2
name=$(echo "$@" | sed -E 's/.*--name ([^ ]+).*/\1/')
echo "$port $*" >> calls.txt
python3 - "$name" <<'EOF'
import json, os, sys
name = sys.argv[1]
spec = json.loads(os.environ["STUB_TIMES"])[name.split("_r")[0].rsplit("_", 1)[1]]
dur, ttft, tpot = spec
rows = [{"type": "header", "attn_dp": 8, "draft_tokens": 1}]
for i in range(32):
    itl = [tpot] + [0.0, 2 * tpot] * 255 + [tpot]
    rows.append(dict(type="request", rid=i, group=i % 8, arrival_s=i * 0.1, prompt_len=2048,
                     output_len=512, finish_s=dur, ttft_ms=ttft, tpot_ms=tpot, itl_ms=itl))
rows.append({"type": "summary"})
with open(f"logs/engine/serve_{name}.jsonl", "w") as f:
    f.write("\n".join(json.dumps(r) for r in rows) + "\n")
EOF
"""


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    eng = tmp_path / "engine"
    (eng / "scripts").mkdir(parents=True)
    (eng / "logs" / "engine").mkdir(parents=True)
    stub = eng / "scripts" / "slice_run.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    monkeypatch.setattr(tl, "ENGINE_ROOT", eng)
    return eng


def test_run_refuses_before_predict(tmp_path, search_exp, fake_engine):
    loop = tl.Loop("t", root=tmp_path / "tune")
    tl.propose(loop, [search_exp], "fixed_2k_512-a0.67", 1, [], [], [])
    with pytest.raises(RuntimeError, match="predict first"):
        tl.run(loop, execute=False, logs=fake_engine / "logs" / "engine")


def test_engine_script_matches_the_knobs(tmp_path, search_exp, fake_engine):
    loop, _, _ = _loop_with_predictions(tmp_path, search_exp)
    (top,) = [c for c in loop.read("proposal.json")["candidates"] if c["label"] == "top1"]
    s = tl.engine_script(loop, top)
    lines = [x for x in s.splitlines() if "slice_run.sh" in x]
    assert len(lines) == 2
    assert lines[0].startswith("scripts/slice_run.sh 61000") and "--cores 0-31" in lines[0]
    assert lines[1].startswith("scripts/slice_run.sh 63000") and "--cores 32-63" in lines[1]
    for flag in (
        "--tp 4",
        "--ep 32",
        "--granularity block",
        "--tp-collective reduce_scatter_all_gather",
        "--moe-dispatch all_gather",
        "--mtp-k 1",
        "--max-batch-tokens 512",
        "--prefill-buckets 128,512",
        "--requests 64 --arrival fixed:0.05 --input fixed:2048 --output fixed:512",
        "--replicas 2",
        "prefill:512:4096",
        "decode:16:4096",
    ):
        assert flag in lines[0], flag


def test_mocked_engine_run_then_compare(tmp_path, search_exp, fake_engine, monkeypatch):
    loop, _, _ = _loop_with_predictions(tmp_path, search_exp)
    logs = fake_engine / "logs" / "engine"
    # Measured: top1 10% slower than predicted, a2a as predicted; same ranking.
    monkeypatch.setenv(
        "STUB_TIMES", json.dumps({"top1": [55.0, 16000, 75.0], "a2a": [60.0, 20000, 80.0]})
    )
    monkeypatch.chdir(fake_engine)
    out = tl.run(loop, execute=True, logs=logs, check_device=lambda: [])
    assert [r["status"] for r in out] == ["finished", "finished"]
    assert len((fake_engine / "calls.txt").read_text().splitlines()) == 4  # 2 cands x 2 replicas
    assert loop.done("run")
    # Resumable: a second execute runs nothing.
    tl.run(loop, execute=True, logs=logs, check_device=lambda: [])
    assert len((fake_engine / "calls.txt").read_text().splitlines()) == 4

    res = tl.compare(loop, threshold=15.0, logs=logs)
    top = next(r for r in res["rows"] if r["label"] == "top1")
    assert top["err"]["duration_s"] == pytest.approx(100 * (50 / 55 - 1))
    assert top["accept_meas"] == pytest.approx(1.0, abs=0.01)  # verify steps emit 2 tokens
    assert res["ranking"]["duration_s"]["measured"] == ["top1", "a2a"]
    assert res["ranking"]["duration_s"]["kendall_tau"] == 1.0
    assert res["verdict"] == "trust"
    assert "verdict: **trust**" in loop.path("comparison.md").read_text()

    # A tighter threshold flags the 9% duration error: re-align before trusting.
    assert tl.compare(loop, threshold=5.0, logs=logs)["verdict"] == "realign"


def test_device_busy_blocks_execution(tmp_path, search_exp, fake_engine):
    loop, _, _ = _loop_with_predictions(tmp_path, search_exp)
    with pytest.raises(RuntimeError, match="containers running"):
        tl.run(
            loop,
            execute=True,
            logs=fake_engine / "logs" / "engine",
            check_device=lambda: ["slice-61000"],
        )


def test_ranking_flip_beyond_noise_is_flagged():
    labels = ["a", "b", "c"]
    meas = [100.0, 110.0, 101.0]  # a < c < b; a vs c is a tie (1%)
    pred = [102.0, 100.0, 101.0]  # b predicted fastest: a/b and b/c flip
    assert tl.discordant_pairs(labels, meas, pred) == [("a", "b"), ("b", "c")]
    assert tl.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert tl.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0


def test_headroom_runs_each_what_if_on_the_candidates_knobs(tmp_path, search_exp):
    loop, _, _ = _loop_with_predictions(tmp_path, search_exp)
    seen = {}

    def simulate(preset: Path, analyze: bool):
        arch = json.loads(preset.read_text())["pools"]["main"]["groups"][0]["arch"]
        seen[preset.parent.name] = arch
        _fake_sim_run(preset.parent / "run", 40.0 if "kind_time_scale" in arch else 50.0, 1e4, 60)

    sc = {k: tl.SCENARIOS[k] for k in ("moe_prefill_3x", "no_pad")}
    h = tl.headroom(loop, "top1", sc, simulate=simulate)
    assert seen["no_pad"]["spmd_padding"] is False and seen["no_pad"]["attn_tp"] == 4
    assert h["scenarios"]["moe_prefill_3x"]["change_pct"]["duration_s"] == pytest.approx(-20.0)
    assert h["scenarios"]["no_pad"]["change_pct"]["duration_s"] == pytest.approx(0.0)
    md = loop.path("headroom_top1.md").read_text()
    assert md.index("moe_prefill_3x") < md.index("no_pad")  # ranked by gain
    tl.headroom(loop, "top1", sc, simulate=lambda p, a: pytest.fail("resumed runs re-simulated"))
