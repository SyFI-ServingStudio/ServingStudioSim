"""CPU tests: tree_verify_attention reference and the tensorfold_triton runner's input gate."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.attention import tree_verify_attention_reference as ref
from profiling.runners.attention.tree_verify_attention_tensorfold import validate_args, work
from profiling.runners.exceptions import ProfilerNotImplemented


def test_path_mask_is_ancestors_and_self() -> None:
    # Catches a causal (lower-triangular) mask standing in for the tree mask:
    # row 3's parent is 1, so it must not see its uncle row 2.
    mask = ref.path_mask(torch, [-1, 0, 0, 1, -1])
    assert mask[3].tolist() == [True, True, False, True, False]
    assert mask[2].tolist() == [True, False, True, False, False]
    assert mask[4].tolist() == [False, False, False, False, True]  # a second root


def test_path_mask_rejects_a_parent_after_its_child() -> None:
    with pytest.raises(ValueError):
        ref.path_mask(torch, [-1, 2, 0])


def test_reference_matches_per_row_softmax_across_streams() -> None:
    # Catches a stream reading another stream's cache or window rows, or GQA
    # heads mapped interleaved instead of grouped.
    torch.manual_seed(0)
    h, hk, d = 4, 2, 8
    parents = [[-1, 0, 0], [-1]]
    committed = [5, 0]
    width = 4
    q = torch.randn(width, h, d)
    kn, vn = torch.randn(width, hk, d), torch.randn(width, hk, d)
    caches = [(torch.randn(6, hk, d), torch.randn(6, hk, d)), (torch.randn(1, hk, d),) * 2]
    out = ref.tree_verify_attention_reference(torch, q, kn, vn, caches, parents, committed, 0.3)
    visible = {0: [0], 1: [0, 1], 2: [0, 2], 3: [3]}  # global window rows each row sees
    stream = {0: 0, 1: 0, 2: 0, 3: 1}
    for row in range(width):
        s = stream[row]
        for head in range(h):
            kv = head // (h // hk)
            keys = torch.cat([caches[s][0][: committed[s], kv], kn[visible[row], kv]])
            values = torch.cat([caches[s][1][: committed[s], kv], vn[visible[row], kv]])
            weights = torch.softmax(keys @ q[row, head] * 0.3, dim=0)
            torch.testing.assert_close(out[row, head], weights @ values)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (dict(q_dtype="fp8_e4m3"), ProfilerNotImplemented),
        (dict(head_dim=64), ProfilerNotImplemented),
        (dict(num_qo_heads=25), ValueError),
        (dict(query_context_pairs=()), ValueError),
        (dict(query_context_pairs=((129, 10),)), ValueError),  # attention.MAX_NODES
        (dict(query_context_pairs=((0, 10),)), ValueError),
        (dict(query_context_pairs=((4, -1),)), ValueError),
        (dict(query_context_pairs=((4.0, 10),)), TypeError),
    ],
)
def test_runner_refuses_what_attention_does_not_run(kwargs, error) -> None:
    args = dict(
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim=256,
        q_dtype="bf16",
        kv_dtype="bf16",
        o_dtype="bf16",
        query_context_pairs=((16, 5000),),
    )
    args.update(kwargs)
    with pytest.raises(error):
        validate_args(**args)


def test_runner_accepts_json_pairs_and_counts_chain_work() -> None:
    # Pairs arrive as JSON lists; a chain row r sees committed + r + 1 keys.
    pairs = validate_args(24, 4, 256, "bf16", "bf16", "bf16", [[2, 10], [1, 0]])
    assert pairs == ((2, 10), (1, 0))
    flops, _ = work(pairs, num_qo_heads=24, num_kv_heads=4, head_dim=256)
    assert flops == 4 * 24 * 256 * ((10 + 1) + (10 + 2) + 1)
