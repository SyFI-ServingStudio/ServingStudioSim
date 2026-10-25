"""Neuron MoE / HC / PLE / head / collective / launch kinds and their bench importers, on the CPU.

The kinds' args are the wire contract with the Rust KernelSpecs (doc/pr/trn2-flashnext/
L1_CONTRACT.md): a reordered or renamed field silently misses every row. The importers map a
bench row's (ep, tp) onto per-rank expert / intermediate / vocab / table sizes, the bench impl
onto the backend, and core lists onto a fabric; a wrong mapping files real measurements under
the wrong shape, and a noisy collective row (negative slope) must never become a time.
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from profiling.db.registry import find_kernel_profiler_spec, known_backends

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "neuron-bench-import"))

from families import collectives as cc_family  # noqa: E402
from families import head as head_family  # noqa: E402
from families import launch as launch_family  # noqa: E402
from families import moe as moe_family  # noqa: E402
from families import ple as ple_family  # noqa: E402

from profiling.runners.neuron import collectives as cc_runner  # noqa: E402

MIX = ["num_tokens", "hidden_size", "hc_mult", "lowrank", "dtype"]
CC = ["num_gpus", "message_size_bytes", "dtype", "fabric"]
ROUTED = [
    "neuron_auto", "neuron_nki", "neuron_nki_tkg", "neuron_nki_tkg_all", "neuron_nki_cte",
    "neuron_nki_cte_cf2", "neuron_torch_gather_w", "neuron_torch_dense", "neuron_torch_gather",
    "neuron_torch_onehot",
]  # fmt: skip
CONTRACT = {
    "moe_router": (
        ["num_tokens", "hidden_size", "num_experts", "top_k", "dtype"],
        ["neuron_torch", "neuron_torch_topk_first", "neuron_torch_bf16", "neuron_nki_topk"],
    ),
    "moe_routed_experts": (
        [
            "num_tokens",
            "hidden_size",
            "num_experts",
            "top_k",
            "local_experts",
            "intermediate_size",
            "dtype",
        ],
        ROUTED,
    ),
    "moe_shared_expert": (
        ["num_tokens", "hidden_size", "intermediate_size", "dtype"],
        ["neuron_torch", "neuron_nki"],
    ),
    "moe_block": (
        [
            "num_tokens",
            "hidden_size",
            "num_experts",
            "top_k",
            "local_experts",
            "intermediate_size",
            "shared_intermediate_size",
            "dtype",
        ],
        ["neuron_auto", "neuron_nki"],
    ),  # fmt: skip
    "hc_mix": (MIX, ["neuron_torch", "neuron_torch_cat"]),
    "hc_mix_final": (MIX, ["neuron_torch"]),
    "hc_combine_mix": (MIX, ["neuron_torch", "neuron_torch_cat"]),
    "hc_combine": (["num_tokens", "hidden_size", "hc_mult", "hidden_dtype"], ["neuron_torch"]),
    "ple_project": (
        ["num_tokens", "embed_dim", "hidden_size", "hc_mult", "dtype"],
        ["neuron_torch"],
    ),
    "ple_residual": (["num_tokens", "hidden_size", "hc_mult", "dtype"], ["neuron_torch"]),
    "ple_post": (
        [
            "num_tokens",
            "embed_dim",
            "hidden_size",
            "hc_mult",
            "conv_kernel_size",
            "dilation",
            "dtype",
        ],
        ["neuron_torch"],
    ),
    "ple_block": (
        ["num_tokens", "num_heads", "head_dim", "table_rows", "hidden_size", "hc_mult", "dtype"],
        ["neuron_torch"],
    ),
    "ple_ngram_ids": (
        ["num_tokens", "num_reqs", "ngram_context_len", "heads_per_ngram"],
        ["neuron_torch"],
    ),
    "ple_gate": (["num_tokens", "hidden_size", "hc_mult", "dtype"], ["neuron_torch"]),
    "ple_short_conv": (
        ["batch_size", "query_len", "channels", "kernel_size", "dilation", "mode", "dtype"],
        ["neuron_torch"],
    ),
    "embedding_row_gather": (
        ["num_tokens", "num_heads", "head_dim", "table_rows", "residency", "weight_dtype"],
        ["neuron_torch"],
    ),
    "gemm_fp32_output": (["m", "n", "k", "input_dtype"], ["neuron_torch"]),
    "single_gemm": (["m", "n", "k", "dtype"], ["neuron_torch"]),
    "vocab_argmax": (["num_tokens", "vocab_size", "dtype"], ["neuron_torch", "neuron_nki"]),
    "lm_head_argmax": (
        ["num_tokens", "hidden_size", "vocab_size", "dtype"],
        ["neuron_torch", "neuron_nki"],
    ),
    "all_reduce": (CC, ["neuron_cc"]),
    "all_gather": (CC, ["neuron_cc"]),
    "reduce_scatter": (CC, ["neuron_cc"]),
    "all_to_all": (CC, ["neuron_cc"]),
    "region_launch_overhead": (["num_regions", "dtype"], ["neuron_torch"]),
}
COMM_KINDS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


@pytest.mark.parametrize("kind", sorted(CONTRACT))
def test_neuron_kind_matches_contract(kind: str) -> None:
    order, backends = CONTRACT[kind]
    neuron = sorted(b for b in known_backends(kind) if b.startswith("neuron_"))
    assert neuron == sorted(backends)
    for backend in backends:
        spec = find_kernel_profiler_spec(kind, backend)
        assert [f.name for f in fields(spec.args_schema)] == order
        assert spec.supports.device_family == "neuron"


@pytest.mark.parametrize("kind", COMM_KINDS)
def test_neuron_collectives_are_list_native_comm_on_one_chip(kind: str) -> None:
    # One chunk = one chip: the runner spawns ranks over the 4-core span itself, so a
    # gpu_count_fn would make the pool look for several chips.
    spec = find_kernel_profiler_spec(kind, "neuron_cc")
    assert spec.metric_family.name == "COMM"
    assert spec.list_native and spec.neuron_logical_cores == 4 and spec.gpu_count_fn is None
    assert spec.runner_ref.module_name == "profiling.runners.neuron.collectives"


def _moe(kind, impl, ep=1, tp=1, routing="real", tokens=64):
    return dict(kind=kind, impl=impl, tokens=tokens, ep=ep, tp=tp, routing=routing, device_ms=1.0)


def test_moe_block_maps_routed_impl_and_splits_shared_over_all_ranks() -> None:
    table, backend, args = moe_family.map_row(
        _moe("moe_block", "torch_topk_first/nki/torch", ep=4, tp=2)
    )
    assert (table, backend) == ("moe_block", "neuron_nki")
    assert args["local_experts"] == 128 and args["intermediate_size"] == 320
    assert args["shared_intermediate_size"] == 640 // 8
    assert (
        moe_family.map_row(_moe("moe_block", "torch_topk_first/auto/torch", routing="uniform"))
        is None
    )
    assert moe_family.map_row(_moe("moe_block", "torch/auto/torch")) is None


def test_moe_shared_expert_is_640_over_split_and_routed_needs_real_routing() -> None:
    table, backend, args = moe_family.map_row(_moe("shared_expert", "nki", tp=16, routing="none"))
    assert (table, backend, args["intermediate_size"]) == ("moe_shared_expert", "neuron_nki", 40)
    table, backend, args = moe_family.map_row(_moe("routed_experts", "nki_cte", ep=16))
    assert (table, backend) == ("moe_routed_experts", "neuron_nki_cte")
    assert (args["local_experts"], args["intermediate_size"]) == (32, 640)
    assert moe_family.map_row(_moe("routed_experts", "nki_cte", ep=16, routing="uniform")) is None
    # an impl the contract does not register files nothing
    assert moe_family.map_row(_moe("routed_experts", "nki_cte_noskip")) is None


def test_ple_layout_rule_and_short_conv_mode() -> None:
    def row(kind, phase, tokens, batch, **kw):
        base = dict(kind=kind, impl="torch", phase=phase, tokens=tokens, batch=batch, tp=1)
        return {**base, **kw}

    assert ple_family.map_row(row("ple_post", "decode", 128, 128))[1]["num_tokens"] == 128
    assert ple_family.map_row(row("ple_post", "prefill", 128, 1)) is None
    assert ple_family.map_row(row("ple_post", "prefill", 512, 1))[1]["num_tokens"] == 512
    _, conv = ple_family.map_row(row("ple_short_conv", "prefill", 512, 1))
    assert (conv["batch_size"], conv["query_len"], conv["mode"], conv["channels"]) == (
        1,
        512,
        "prefill",
        10240,
    )
    _, conv = ple_family.map_row(row("ple_short_conv", "decode", 64, 64))
    assert (conv["batch_size"], conv["query_len"], conv["mode"]) == (64, 1, "decode")
    _, ids = ple_family.map_row(row("ple_ngram_ids", "decode", 64, 64))
    assert (ids["num_tokens"], ids["num_reqs"]) == (64, 64)
    table, gather = ple_family.map_row(row("ple_gather", "decode", 8, 8, tp=16))
    assert table == "embedding_row_gather"
    assert (gather["num_heads"], gather["head_dim"], gather["table_rows"]) == (
        16,
        160,
        320001536 // 16,
    )


def test_head_rows_use_the_rank_vocab_shard() -> None:
    def row(kind, impl, tp):
        return dict(kind=kind, impl=impl, tokens=4, tp=tp, phase="decode")

    table, _, args = head_family.map_row(row("embed", "torch", 8))
    assert (
        table == "embedding_row_gather" and args["table_rows"] == 31040 and args["num_heads"] == 1
    )
    table, _, args = head_family.map_row(row("lm_head", "torch", 16))
    assert (table, args["n"], args["k"]) == ("gemm_fp32_output", 15520, 2560)
    table, backend, args = head_family.map_row(row("argmax_local", "nki", 2))
    assert (table, backend, args["vocab_size"]) == ("vocab_argmax", "neuron_nki", 124160)


def test_launch_cases_map_to_region_counts() -> None:
    assert launch_family.map_case(dict(case="trivial_add", tokens=1)) == 1
    assert launch_family.map_case(dict(case="chain_k_regions", tokens=1, k=8)) == 8
    assert launch_family.map_case(dict(case="chain_k_regions", tokens=128, k=8)) is None
    assert launch_family.map_case(dict(case="chain_one_region", tokens=1, k=8)) is None


@pytest.mark.parametrize(
    ("cores", "fabric"),
    [
        ([32, 33], "neuron_c2"),
        ([32, 33, 34, 35], "neuron_c4"),
        (list(range(32, 40)), "neuron_c4"),
        ([32, 36, 40, 44], "neuron_c1"),
        ([32, 33, 36], None),
    ],
)
def test_collective_fabric_from_cores(cores, fabric) -> None:
    assert cc_family.fabric(cores) == fabric


def test_collective_rows_drop_noise_and_prefer_v2(tmp_path: Path) -> None:
    base = dict(world=2, cores=[32, 36], bytes=16384)
    v1 = [
        dict(op="all_reduce", per_cc_device_ms=0.02, **base),
        dict(op="all_gather", per_cc_device_ms=-0.01, **base),
        dict(op="all_to_all", error="RuntimeError: Invalid", **base),
        dict(op="reduce_scatter", per_cc_device_ms=0.03, **base),
    ]
    v2 = [
        dict(op="all_reduce", per_cc_device_ms=0.025, **base),
        dict(op="reduce_scatter", error="RuntimeError: Invalid", **base),
    ]
    for name, rows in zip(cc_family.FILES, (v1, v2)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    from import_bench import latest

    got = {(r.kind, r.args["fabric"]): r.time_ms for r in latest(cc_family.rows(tmp_path))}
    assert got == {("all_reduce", "neuron_c1"): 0.025, ("reduce_scatter", "neuron_c1"): 0.03}


def test_live_collective_runner_refuses_multi_chip_groups() -> None:
    ok = dict(num_gpus=2, message_size_bytes=16384, dtype="bf16", fabric="neuron_c2")
    assert cc_runner.check_spec(ok) is None
    assert cc_runner.check_spec(dict(ok, num_gpus=4, fabric="neuron_c4")) is None
    for bad in (dict(ok, fabric="neuron_c1"), dict(ok, num_gpus=8, fabric="neuron_c4")):
        assert "sim_l1.collectives" in cc_runner.check_spec(bad)
    assert cc_runner.bus_factor("all_gather", 4) == (4.0, 3.0)
