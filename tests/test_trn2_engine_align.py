"""tools/trn2-engine-align: reading the flashnext engine's serving logs (CPU tier)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "trn2-engine-align"))

import engine_align as ea  # noqa: E402


def test_percentile_is_the_engines_nearest_rank():
    assert ea.pct([4, 1, 3, 2], 0.5) == 3  # index int(0.5 * 3 + 0.5) = 2
    assert ea.pct([1.0], 0.9) == 1.0
    assert ea.pct([], 0.5) is None


def test_acceptance_reads_verify_steps_off_the_itls():
    # Prefill token, then steps of 2, 1, 2, 3 tokens, then a clipped 1-token tail:
    # ITL 0.0 joins a token to the previous one's step.
    itl = [5.0, 0.0, 9.0, 8.0, 0.0, 7.0, 0.0, 0.0, 6.0]
    req = {"itl_ms": itl}
    # Steps (prefill and tail dropped): [2, 1, 2, 3].
    assert ea.acceptance(req, 1) == [0.75]
    p1, p2 = ea.acceptance(req, 2)
    assert p1 == 0.75 and abs(p2 - 1 / 3) < 1e-9


def _rows(k=0):
    header = {"type": "header", "attn_dp": 2, "draft_tokens": k}
    reqs = [
        {"type": "request", "rid": 0, "group": 0, "admit_s": 0.0, "prompt_len": 3000},
        {"type": "request", "rid": 1, "group": 1, "admit_s": 0.1, "prompt_len": 100},
        {"type": "request", "rid": 2, "group": 1, "admit_s": 0.2, "prompt_len": 50},
    ]

    def step(i, dec, pre, ctx=4096):
        fw = []
        if any(dec):
            fw.append({"phase": "decode", "C": ctx, "wall_ms": 100.0})
        if any(pre):
            fw.append({"phase": "prefill", "C": 4096, "wall_ms": 900.0})
        return {
            "type": "step",
            "step": i,
            "kind": "mixed",
            "wall_ms": 1000.0,
            "decode_reqs": dec,
            "prefill_tokens": pre,
            "decode_B": max(dec),
            "prefill_T": 2048,
            "forwards": fw,
        }

    steps = [step(0, [0, 0], [2048, 100]), step(1, [0, 1], [952, 50]), step(2, [1, 2], [0, 0])]
    return [header, *steps, *reqs]


def test_step_cases_rebuild_each_groups_chunk_start():
    cases = ea.step_cases(_rows())
    g = [c["groups"] for c, _ in cases]
    assert g[0] == [{"prefill_chunk_pairs": [[0, 2048]]}, {"prefill_chunk_pairs": [[0, 100]]}]
    # Group 0 continues its prompt; group 1 starts its next one.
    assert g[1][0] == {"prefill_chunk_pairs": [[2048, 952]]}
    assert g[1][1] == {"decode_kv_lens": [4096], "prefill_chunk_pairs": [[0, 50]]}
    assert g[2] == [{"decode_kv_lens": [4096]}, {"decode_kv_lens": [4096, 4096]}]
    meas = cases[1][1]
    assert (meas["dec_wall"], meas["pre_wall"]) == (100.0, 900.0)


def test_speculative_step_cases_carry_the_verify_width():
    cases = ea.step_cases(_rows(k=2))
    assert cases[2][0]["groups"][1]["decode_requests"] == [[4096, 3], [4096, 3]]
    assert cases[0][0]["groups"][0]["decode_requests"] == []
