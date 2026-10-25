"""Neuron GDN / QSA kinds and their bench importer, on the CPU (no device, no bench files).

The kinds' args are the wire contract with the Rust KernelSpecs (doc/pr/trn2-flashnext/
L1_CONTRACT.md): a reordered or renamed field silently misses every row. The importer maps a bench
row's tp onto per-rank head counts and MTP verify rows onto query_len; a wrong mapping files real
measurements under the wrong shape.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from profiling.db.registry import find_kernel_profiler_spec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "neuron-bench-import"))

from families import gdn as gdn_family  # noqa: E402
from families import qsa as qsa_family  # noqa: E402

GDN_COMMON = [
    "hidden_size",
    "num_key_heads",
    "num_value_heads",
    "head_dim",
    "conv_kernel_size",
    "dtype",
]
QSA_COMMON = [
    "hidden_size",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "index_heads",
    "index_head_dim",
    "index_topk_tokens",
    "compress_ratio",
    "dtype",
]
ROWS, CTX = ["mode", "num_rows"], ["mode", "num_rows", "context_len"]
# kind -> (args in contract order, backends)
CONTRACT = {
    "gdn_prefill_mixer": (["num_tokens", *GDN_COMMON], ["neuron_torch", "neuron_nki"]),
    "gdn_decode_mixer": (
        ["batch_size", "query_len", *GDN_COMMON],
        ["neuron_torch", "neuron_nki"],
    ),
    "gdn_in_proj": (["num_tokens", *GDN_COMMON], ["neuron_torch"]),
    "gdn_conv1d_prefill": (["num_tokens", *GDN_COMMON], ["neuron_torch", "neuron_nki"]),
    "gdn_conv1d_decode": (
        ["batch_size", "query_len", *GDN_COMMON],
        ["neuron_torch", "neuron_nki"],
    ),
    "gdn_split_gating": (["num_tokens", *GDN_COMMON], ["neuron_torch"]),
    "gdn_prefill_core": (["num_tokens", *GDN_COMMON], ["neuron_torch", "neuron_nki"]),
    "gdn_decode_core": (
        ["batch_size", "query_len", *GDN_COMMON],
        ["neuron_torch", "neuron_nki"],
    ),
    "gdn_output": (["num_tokens", *GDN_COMMON], ["neuron_torch"]),
    "gdn_state_io": (["batch_size", *GDN_COMMON], ["neuron_torch"]),
    "qsa_mixer2": ([*CTX, *QSA_COMMON], ["neuron_torch_sparse", "neuron_nki"]),
    "qsa_project": ([*ROWS, *QSA_COMMON], ["neuron_torch"]),
    "qsa_indexer_project": ([*ROWS, *QSA_COMMON], ["neuron_torch"]),
    "qsa_cache_write": ([*ROWS, *QSA_COMMON], ["neuron_torch"]),
    "qsa_indexer_compress": ([*ROWS, *QSA_COMMON], ["neuron_torch"]),
    "qsa_indexer_scores": ([*CTX, *QSA_COMMON], ["neuron_torch", "neuron_torch_bf16"]),
    "qsa_indexer_select": ([*CTX, *QSA_COMMON], ["neuron_torch", "neuron_nki"]),
    "qsa_attention": (
        [*CTX, *QSA_COMMON],
        ["neuron_torch_dense", "neuron_torch_sparse", "neuron_nki"],
    ),
    "qsa_output": ([*ROWS, *QSA_COMMON], ["neuron_torch"]),
}


@pytest.mark.parametrize("kind", sorted(CONTRACT))
def test_args_and_backends_follow_the_contract(kind: str) -> None:
    order, backends = CONTRACT[kind]
    for backend in backends:
        spec = find_kernel_profiler_spec(kind, backend)
        assert [f.name for f in fields(spec.args_schema)] == order
        # The generic runner splits profile_<kind>__<impl> back into the adapter's (kind, impl).
        assert spec.runner_ref.function_name == f"profile_{kind}__{backend.removeprefix('neuron_')}"


def _bench(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_gdn_verify_rows_are_decode_mixer_rows_with_query_len_spec_tokens(tmp_path: Path) -> None:
    bench = _bench(
        tmp_path,
        "gdn.jsonl",
        [
            dict(kind="gdn_verify_mixer", impl="torch", phase="decode", tokens=4, batch=8, tp=16,
                 spec_tokens=4, device_ms=0.5),
            dict(kind="gdn_decode_mixer", impl="nki", phase="decode", tokens=1, batch=8, tp=8,
                 device_ms=0.3),
            dict(kind="gdn_decode_mixer", impl="nki_fused", phase="decode", tokens=1, batch=8, tp=1,
                 lnc=1, device_ms=0.3),
            dict(kind="gdn_prefill_mixer", impl="nki", phase="prefill", tokens=128, batch=1, tp=1,
                 device_ms=None, device_error="x"),
        ],
    )  # fmt: skip
    got = {
        (r.backend, r.args["batch_size"], r.args["query_len"]): r for r in gdn_family.rows(bench)
    }
    assert set(got) == {("neuron_torch", 8, 4), ("neuron_nki", 8, 1)}
    verify = got[("neuron_torch", 8, 4)]
    assert verify.kind == "gdn_decode_mixer"
    assert (verify.args["num_key_heads"], verify.args["num_value_heads"]) == (1, 3)
    assert (
        got[("neuron_nki", 8, 1)].args["num_key_heads"],
        got[("neuron_nki", 8, 1)].args["num_value_heads"],
    ) == (2, 6)


def test_gdn_token_kinds_prefer_the_prefill_row(tmp_path: Path) -> None:
    from import_bench import latest

    bench = _bench(
        tmp_path,
        "gdn.jsonl",
        [
            dict(kind="gdn_in_proj", impl="torch", phase="prefill", tokens=16, batch=1, tp=1,
                 device_ms=1.0),
            dict(kind="gdn_in_proj", impl="torch", phase="decode", tokens=1, batch=16, tp=1,
                 device_ms=2.0),
            dict(kind="gdn_in_proj", impl="torch", phase="decode", tokens=1, batch=2, tp=1,
                 device_ms=3.0),
        ],
    )  # fmt: skip
    rows = {r.args["num_tokens"]: r.time_ms for r in latest(gdn_family.rows(bench))}
    assert rows == {16: 1.0, 2: 3.0}


def test_qsa_rows_map_tp_to_local_heads_and_keep_only_summed_nki_select_mixer2(
    tmp_path: Path,
) -> None:
    common = dict(device_ms=1.0, batch=1, regions=2, device_neffs=2)
    bench = _bench(
        tmp_path,
        "qsa.jsonl",
        [
            dict(kind="qsa_mixer2_prefill", impl="nki", phase="prefill", tokens=2048, context=34816,
                 tp=8, select_impl="nki", **common),
            dict(kind="qsa_mixer2_prefill", impl="nki", phase="prefill", tokens=512, context=512,
                 tp=8, select_impl="torch", **common),
            dict(kind="qsa_mixer2_decode", impl="torch_sparse", phase="decode", tokens=64,
                 context=32768, tp=1, select_impl="nki", **common),
            # device_neffs 1: both regions lumped under one trace model; the time is no sum.
            dict(kind="qsa_mixer2_decode", impl="nki", phase="decode", tokens=8, context=4096,
                 tp=8, select_impl="nki", device_ms=0.3, batch=8, regions=2, device_neffs=1),
            dict(kind="qsa_mixer_decode", impl="nki", phase="decode", tokens=64, context=32768,
                 tp=1, select_impl="nki", **common),
            dict(kind="qsa_attention", impl="nki", phase="prefill", tokens=128, context=128, tp=1,
                 error="RuntimeError: x", device_ms=1.0),
        ],
    )  # fmt: skip
    rows = list(qsa_family.rows(bench))
    assert [(r.kind, r.backend) for r in rows] == [
        ("qsa_mixer2", "neuron_nki"),
        ("qsa_mixer2", "neuron_torch_sparse"),
    ]
    prefill, decode = (r.args for r in rows)
    assert (prefill["mode"], prefill["num_rows"], prefill["context_len"]) == (
        "prefill",
        2048,
        34816,
    )
    assert (prefill["num_q_heads"], prefill["num_kv_heads"]) == (3, 1)
    assert (decode["mode"], decode["num_rows"], decode["context_len"]) == ("decode", 64, 32768)
    assert (decode["num_q_heads"], decode["num_kv_heads"]) == (24, 2)
    assert [qsa_family.geometry(tp)["num_q_heads"] for tp in (2, 4)] == [12, 6]
    assert {qsa_family.geometry(tp)["num_kv_heads"] for tp in (2, 4, 8)} == {1}


def test_qsa_row_only_kinds_keep_only_the_profiled_context(tmp_path: Path) -> None:
    # The profiler builds row-only kinds at prefill prior 0 and decode context 4096; a bench row at
    # another context is a different region and must not take the key.
    bench = _bench(
        tmp_path,
        "qsa.jsonl",
        [
            dict(kind="qsa_project", impl="torch", phase="prefill", tokens=384, context=2084, tp=1,
                 device_ms=10.5),
            dict(kind="qsa_project", impl="torch", phase="prefill", tokens=512, context=512, tp=1,
                 device_ms=1.5),
            dict(kind="qsa_output", impl="torch", phase="decode", tokens=8, context=32768, tp=1,
                 device_ms=0.1),
            dict(kind="qsa_output", impl="torch", phase="decode", tokens=8, context=4096, tp=2,
                 device_ms=0.08),
        ],
    )  # fmt: skip
    got = [(r.kind, r.args["mode"], r.args["num_rows"], r.time_ms) for r in qsa_family.rows(bench)]
    assert got == [("qsa_project", "prefill", 512, 1.5), ("qsa_output", "decode", 8, 0.08)]
    assert all("context_len" not in r.args for r in qsa_family.rows(bench))


def _grid_args(kind: str) -> list[dict]:
    gdn = gdn_family.geometry(8)
    qsa = qsa_family.geometry(8)
    if kind.startswith("gdn_"):
        shape = CONTRACT[kind][0][: -len(GDN_COMMON)]
        point = dict(num_tokens=8192, batch_size=256, query_len=3)
        return [{k: point[k] for k in shape} | gdn, {k: 1 for k in shape} | gdn]
    out = []
    for mode, n, ctx in (("prefill", 128, 128), ("prefill", 8192, 40960), ("decode", 1, 4096)):
        a = dict(mode=mode, num_rows=n) | qsa
        if "context_len" in CONTRACT[kind][0]:
            a["context_len"] = ctx
        out.append(a)
    return out


@pytest.mark.parametrize("kind", sorted(CONTRACT))
def test_every_row_gets_nonzero_flops_or_bytes(kind: str) -> None:
    # A row with neither looks free to the headroom analyzer.
    family = gdn_family if kind.startswith("gdn_") else qsa_family
    for args in _grid_args(kind):
        flops, nbytes = family.cost(kind, args)
        assert flops >= 0 and nbytes > 0, (kind, args)


def test_qsa_attention_counts_each_rows_causal_context_up_to_the_selection() -> None:
    a = dict(mode="prefill", num_rows=4, context_len=4) | qsa_family.geometry(8)
    assert qsa_family.attended(a) == 1 + 2 + 3 + 4
    long = dict(a, num_rows=128, context_len=32896)
    assert qsa_family.attended(long) == 128 * 2052
    assert qsa_family.attended(dict(long, mode="decode", context_len=1000)) == 128 * 1000
