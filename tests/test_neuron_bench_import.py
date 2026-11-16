"""CPU tests for the GLM-5.3-Flash Trainium2 L1 importer, the InferenceX trace and the search
prefilter.

Covers ``tools/neuron-bench-import`` (bench row mapping, the derived grid fill, the profile.db
round trip), ``trace/generate_inferencex.py`` and ``tools/trn2-config-search`` (HBM prefilter,
decode buckets, InferenceX metrics on a synthetic run).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORT_DIR = ROOT / "tools" / "neuron-bench-import"
SEARCH_DIR = ROOT / "tools" / "trn2-config-search"
for p in (str(IMPORT_DIR), str(SEARCH_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import_bench = importlib.import_module("import_bench")
glm53 = importlib.import_module("families.glm53")
grid = importlib.import_module("grid")
G = importlib.import_module("geometry")
analytic = importlib.import_module("analytic")


def _bench(tmp_path: Path, rows: list[dict], name: str = "moe.jsonl") -> Path:
    d = tmp_path / "bench"
    d.mkdir(exist_ok=True)
    with (d / name).open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return d


def _row(kind, impl, tokens, batch=None, context=0, tp=1, ep=1, ms=1.0, **kw):
    return dict(
        kind=kind,
        impl=impl,
        phase="prefill",
        tokens=tokens,
        batch=batch or tokens,
        context=context,
        tp=tp,
        ep=ep,
        dtype="bf16",
        device_ms=ms,
        latency_ms=ms,
        pipelined_ms=ms,
        env={},
        **kw,
    )


def test_bench_rows_map_to_per_rank_args(tmp_path):
    """Catches a bench field landing on the wrong per-rank arg (heads by tp, experts by ep)."""
    d = _bench(
        tmp_path,
        [
            _row("kda_prefill", "torch", 2048, tp=8),
            _row("dsa_decode", "radix", 16, batch=16, context=4096, tp=4),
            _row("moe_partial", "cte_hybrid", 8192, ep=32),
            _row("router", "fp32", 64),
            _row("lm_head", "torch", 8, tp=16),
            _row("moe_partial", "no_such_impl", 8, ep=32),
            _row("mhc_pre", "torch_bf16", 4096, ms=None, error="boom"),
        ],
    )
    rows = {r.kind: r for r in glm53.rows(d)}
    assert rows["kda_prefill"].args["num_heads"] == 8
    assert rows["kda_prefill"].backend == "neuron_torch"
    assert (rows["dsa_decode"].args["batch_size"], rows["dsa_decode"].args["context_len"]) == (
        16,
        4096,
    )
    assert rows["dsa_decode"].args["num_heads"] == 16
    assert rows["moe_partial"].args["local_experts"] == 9
    assert rows["moe_partial"].args["shared_intermediate_size"] == 64
    assert rows["moe_partial"].args["weight_dtype"] == "fp8_e4m3"
    assert rows["moe_router"].backend == "neuron_fp32"
    assert rows["lm_head"].args["vocab_size"] == 9680
    assert "mhc_pre" not in rows, "an errored row is not imported"
    assert all(r.source.startswith("imported:moe.jsonl:") for r in rows.values())


def test_fill_grid_interpolates_ratios_and_never_overwrites_measured(tmp_path):
    """Catches the fill replacing a measured cell, or deriving a cell off the measured trend."""
    measured = []
    for t, ratio in [(512, 2.0), (2048, 4.0)]:
        args = dict(num_tokens=t, **G.kda(8))
        measured.append(
            import_bench.ImportRow(
                "kda_prefill",
                "neuron_torch",
                args,
                analytic.estimate("kda_prefill", args)[0] * ratio,
                "imported:kda.jsonl:1",
            )
        )
    coll = list(
        importlib.import_module("families.collectives").rows(
            IMPORT_DIR / "data" / "trn2_flashnext_collectives.jsonl"
        )
    )
    derived = grid.fill_grid(measured + coll, [8], 32)
    kda8 = {
        r.args["num_tokens"]: r
        for r in derived
        if r.kind == "kda_prefill" and r.args["num_heads"] == 8
    }
    assert 512 not in kda8 and 2048 not in kda8, "measured cells stay measured"
    est = lambda t: analytic.estimate("kda_prefill", dict(num_tokens=t, **G.kda(8)))[0]  # noqa: E731
    assert kda8[1024].time_ms == pytest.approx(est(1024) * (2.0 * 4.0) ** 0.5)  # log-midpoint
    assert kda8[8192].time_ms == pytest.approx(est(8192) * 4.0), "endpoint ratio held"
    assert kda8[128].source.startswith("derived:interp")
    # A kind with no measured row anywhere is pure analytic.
    mhc = [r for r in derived if r.kind == "mhc_post"]
    assert mhc and all(r.source == "derived:analytic" for r in mhc)


def test_fill_grid_covers_every_cell_and_collective(tmp_path):
    """Catches a grid cell the Rust arch asks for that the fill leaves empty (a build miss)."""
    coll = list(
        importlib.import_module("families.collectives").rows(
            IMPORT_DIR / "data" / "trn2_flashnext_collectives.jsonl"
        )
    )
    derived = grid.fill_grid(coll, [2, 4, 8, 16], 32)
    keys = {r.key for r in coll + derived}
    for kind, names, cfg, cells in G.series([2, 4, 8, 16], 32):
        impls = G.DEFAULT_IMPL[kind]
        for impl in (impls,) if isinstance(impls, str) else impls:
            for cell in cells:
                args = dict(zip(names, cell), **cfg)
                assert (kind, f"neuron_{impl}", json.dumps(args, sort_keys=True)) in keys
    for op in ("all_gather", "reduce_scatter"):
        for n, fabric in G.comm_groups([2, 4, 8, 16], 32):
            for b in G.COMM_BYTES:
                args = dict(num_gpus=n, message_size_bytes=b, dtype="bf16", fabric=fabric)
                assert (op, "neuron_cc", json.dumps(args, sort_keys=True)) in keys


def test_rows_round_trip_through_profile_db(tmp_path):
    """Catches an args schema the registry rejects, or provenance lost on insert."""
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import Table

    d = _bench(tmp_path, [_row("mhc_post", "torch", 64, ms=0.25)], "mhc_mhc_post.jsonl")
    db = tmp_path / "profile.db"
    rows = list(glm53.rows(d))
    counts = import_bench.insert(rows, db, dry_run=False)
    assert counts[("mhc_post", "neuron_torch", "measured")] == 1
    spec = find_kernel_profiler_spec("mhc_post", "neuron_torch")
    got = Table(spec, db).query(
        [spec.args_schema(**rows[0].args)], backend="neuron_torch", gpu_name=import_bench.GPU_NAME
    )
    assert got[0].time_ms == pytest.approx(0.25)


def test_inferencex_trace_shape(tmp_path):
    """Catches the 8k1k trace drifting from InferenceX: 12C requests, ISL in
    [0.8 ISL, ISL], OSL fixed."""
    spec = importlib.util.spec_from_file_location(
        "generate_inferencex", ROOT / "trace" / "generate_inferencex.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tmp_path / "t.csv"
    assert gen.write_inferencex_trace(out, concurrency=4, accept_rate=0.8) == 48
    lines = out.read_text().splitlines()
    assert lines[0] == "id,input_len,output_len,arrival_time,accept_rate"
    rows = [ln.split(",") for ln in lines[1:]]
    assert len(rows) == 48
    assert all(6553 <= int(r[1]) <= 8192 and r[2] == "1024" and r[3] == "0.0" for r in rows)


def test_search_prefilter_and_buckets():
    """Catches the HBM prefilter disagreeing with the arch: TP1 never fits, TP8 x 64 does."""
    search = importlib.import_module("search")
    assert not search.memory_gb(search.Layout(attn_tp=1, dmax=8))["fits"]
    assert search.memory_gb(search.Layout(attn_tp=8, dmax=64))["fits"]
    assert search.memory_gb(search.Layout(attn_tp=1))["weights_gb"] == pytest.approx(
        24.47, abs=0.01
    )
    assert search.Layout(attn_tp=16, dmax=64).decode_buckets() == [16, 32, 64]
    assert search.Layout(attn_tp=2, dmax=16).decode_buckets() == [8, 16]
    assert not search.Layout(attn_tp=16, dmax=8).valid()


def test_inferencex_point_drops_warmup(tmp_path):
    """Catches warm-up requests counted in the measured window or a wrong TPOT definition."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    inferencex = importlib.import_module("inferencex")
    run = tmp_path / "run"
    (run / "raw").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"num_gpus": 64}))
    reqs = []
    for i in range(6):  # C = 1 x (2 warm-up + 4 measured)
        start = i * 1000.0
        reqs.append(
            dict(
                request_id=i,
                completed=True,
                arrival_time_ms=start,
                ttft_ms=100.0,
                num_output_tokens=11,
                finish_decode_time_ms=start + 1100.0,
                prefill_processed=1000,
                fresh_prompt_tokens=1000,
            )
        )
    pq.write_table(pa.Table.from_pylist(reqs), run / "raw" / "request_slo.parquet")
    p = inferencex.point(run, conc=1, warmup=2)
    assert p["requests"] == 4
    assert p["median_tpot_ms"] == pytest.approx(100.0)
    assert p["intvty"] == pytest.approx(10.0)
    assert p["window_s"] == pytest.approx((5 * 1000 + 1100 - 2000) / 1e3)
    assert p["tok_s_per_chip"] == pytest.approx(4 * 1011 / p["window_s"] / 16)


def test_engine_align_parses_a_timeline_and_drops_first_use(tmp_path):
    """Catches a compile stall kept in a measured median, or a decode step put in the wrong
    context bucket (positions rebuilt from the request rows)."""
    engine_align = importlib.import_module("engine_align")
    eng = {
        "world": 32,
        "tp": 8,
        "dp": 4,
        "slots": 17,
        "cmax": 9216,
        "prefill_buckets": [2048],
        "decode_buckets": [8, 16],
        "context_buckets": [2048, 4096, 8192, 9216],
    }
    rows = [{"kind": "header", "engine": eng}]
    t = 0.0
    for i, ms in enumerate([50000.0] + [100.0] * 9):  # the first decode compiles
        rows.append(
            {
                "kind": "step",
                "step": i,
                "t": t,
                "decode": {"rows": [4] * 4, "ms": ms},
                "wall_ms": ms,
            }
        )
        t += ms / 1e3 + 0.0001
    rows.append(
        {
            "kind": "step",
            "step": 10,
            "t": t,
            "prefill": {"rows": [2048, 0, 0, 0], "ms": 1200.0, "starts": [2048, 0, 0, 0]},
            "wall_ms": 1200.0,
        }
    )
    rows.append(
        {
            "kind": "request",
            "rid": 0,
            "isl": 7000,
            "osl": 1024,
            "arrival": 0.0,
            "ttft": 0.0,
            "e2e": 1000.0,
            "tpot": 0.1,
        }
    )
    path = tmp_path / "serve.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    parsed = engine_align.parse(path)
    by_key = {tuple(c["key"]): c for c in parsed["cases"]}
    dec = by_key[("decode", 8, 8192)]
    assert (dec["n"], dec["dropped"], dec["measured_ms"]) == (9, 1, 100.0)
    assert by_key[("prefill", 2048, 4096, 2048)]["measured_ms"] == 1200.0
    assert parsed["step_gap_ms"] == pytest.approx(0.1, abs=1e-6)
    # The pipelined forward: device-bound regions, then host-bound ones.
    assert engine_align.forward_ms([(10.0, 2)], 1.0, 0.1, 0.0, 1.0) == pytest.approx(20.2)
    assert engine_align.forward_ms([(0.1, 2)], 1.0, 0.1, 5.0, 1.0) == pytest.approx(7.0)
