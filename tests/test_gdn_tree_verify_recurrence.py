"""CPU tests: gdn_tree_verify_recurrence's slot-class rule, oracle and runner input gate."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

from profiling.runners.attention import gdn_tree_verify_recurrence_reference as ref
from profiling.runners.attention.gdn_chunk_delta_rule_tensorfold import reference as chain_reference
from profiling.runners.attention.gdn_tree_verify_recurrence_tensorfold import (
    launch_trees,
    stream_rows,
    validate_args,
)
from profiling.runners.exceptions import ProfilerNotImplemented

_TENSORFOLD_SRC = Path(__file__).resolve().parents[1] / "alignment/profiler/tensorfold/src"


def _tensorfold_schedule():
    if str(_TENSORFOLD_SRC) not in sys.path:
        sys.path.insert(0, str(_TENSORFOLD_SRC))
    return pytest.importorskip("tensorfold.cuda.kernels.gdn").schedule


def _best_first(rows: int, seed: int, top_k: int = 3) -> list[int]:
    """A DFlash2-like draft tree: expand the most probable frontier node, top_k children each."""
    rng = random.Random(seed)
    parents, frontier = [-1], [(1.0, 0)]
    while len(parents) < rows:
        frontier.sort(reverse=True)
        score, node = frontier.pop(0)
        probs = sorted((rng.random() for _ in range(top_k)), reverse=True)
        total = sum(probs)
        for p in probs:
            if len(parents) == rows:
                break
            parents.append(node)
            frontier.append((score * p / total, len(parents) - 1))
    return parents


_TREES = {
    "root": [-1],
    "chain16": [-1] + list(range(15)),
    "chain128": [-1] + list(range(127)),
    "star16": [-1] + [0] * 15,
    "binary16": [-1] + [(i - 1) // 2 for i in range(1, 16)],
    "binary24": [-1] + [(i - 1) // 2 for i in range(1, 24)],
    "ternary16": [-1] + [(i - 1) // 3 for i in range(1, 16)],
    "caterpillar16": [-1, 0, 1, 2, 2, 3, 5, 5, 6, 8, 8, 9, 11, 11, 12, 14],
    "two_chains16": [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14],
    "five_slots16": [-1, 0, 1, 0, 2, 4, 4, 2, 2, 8, 6, 5, 1, 4, 7, 6],
    "binary256": [-1] + [(i - 1) // 2 for i in range(1, 256)],
    **{f"best_first16_{s}": _best_first(16, s) for s in range(6)},
    **{f"best_first24_{s}": _best_first(24, s, top_k=4) for s in range(3)},
}


@pytest.mark.parametrize("name", sorted(_TREES))
def test_slot_count_matches_tensorfold_schedule(name: str) -> None:
    # Catches the arch-side rule drifting from the scheduler that picks the variant.
    parents = _TREES[name]
    assert ref.slot_count_for_parents(parents) == _tensorfold_schedule()(parents)[1]


def test_slot_count_matches_tensorfold_on_random_trees() -> None:
    schedule = _tensorfold_schedule()
    rng = random.Random(0)
    for _ in range(400):
        rows = rng.randint(1, 40)
        parents = [-1] + [rng.randrange(i) for i in range(1, rows)]
        assert ref.slot_count_for_parents(parents) == schedule(parents)[1], parents


def test_launch_class_is_the_largest_stream_class() -> None:
    # Catches a launch keyed by its first or typical stream: one bushy stream sets it.
    chain, binary = _TREES["chain16"], _TREES["binary16"]
    assert ref.slot_class_for_parents([chain] * 7) == "chain"
    assert ref.slot_class_for_parents([chain] * 7 + [_TREES["star16"]]) == "slots_1_2"
    assert ref.slot_class_for_parents([chain] * 7 + [binary]) == "slots_3_4"
    assert ref.slot_class_for_parents([binary, _TREES["five_slots16"]]) == "slots_5_8"
    assert [ref.slot_class_of_count(n) for n in (0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 32)] == [
        "chain",
        "slots_1_2",
        "slots_1_2",
        "slots_3_4",
        "slots_3_4",
        "slots_5_8",
        "slots_5_8",
        "slots_9_16",
        "slots_9_16",
        "slots_17_32",
        "slots_17_32",
    ]


@pytest.mark.parametrize("slot_class", ["chain", "slots_1_2", "slots_3_4", "slots_5_8"])
def test_canonical_trees_stay_in_their_class(slot_class: str) -> None:
    # Catches a chain tail that changes the variant the runner claims to time.
    schedule = _tensorfold_schedule()
    for rows in range(ref.min_rows_for_class(slot_class), 129):
        tree = ref.canonical_tree(slot_class, rows)
        assert len(tree) == rows
        assert ref.slot_class_of_count(schedule(tree)[1]) == slot_class


def test_reference_follows_each_rows_path_from_its_streams_state() -> None:
    # Catches a row stepping from the previous row instead of its parent, a stream
    # starting from another stream's state, or grouped heads mapped interleaved.
    torch.manual_seed(0)
    hk, hv, d = 2, 4, 128
    trees = [[-1, 0, 0, 1], [-1, 0]]
    rows = 6
    q = torch.randn(rows, hk, d)
    k = torch.nn.functional.normalize(torch.randn(rows, hk, d), dim=-1)
    v = torch.randn(rows, hv, d)
    g = torch.rand(rows, hv) * 0.5 + 0.5
    beta = torch.rand(rows, hv)
    states = [0.1 * torch.randn(hv, d, d) for _ in trees]
    out = ref.gdn_tree_verify_recurrence_reference(torch, q, k, v, g, beta, states, trees)
    base = 0
    for state, tree in zip(states, trees):
        for row, path in enumerate(ref.path_rows(tree)):
            idx = [base + r for r in path]
            expected, _ = chain_reference(torch, q[idx], k[idx], v[idx], g[idx], beta[idx], state)
            torch.testing.assert_close(out[base + row], expected[-1], rtol=1e-5, atol=1e-5)
        base += len(tree)


_ARGS = dict(
    batch_size=2,
    num_tokens=32,
    num_qk_heads=16,
    num_value_heads=48,
    key_head_dim=128,
    value_head_dim=128,
    dtype="bf16",
    state_dtype="fp32",
    slot_class="slots_3_4",
)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        (dict(dtype="fp16"), ProfilerNotImplemented),
        (dict(state_dtype="bf16"), ProfilerNotImplemented),
        (dict(key_head_dim=64), ProfilerNotImplemented),
        (dict(slot_class="slots_9_16"), ProfilerNotImplemented),
        (dict(slot_class="bushy"), ValueError),
        (dict(num_value_heads=40), ValueError),
        (dict(num_tokens=1), ValueError),
        (dict(num_tokens=8), ValueError),  # an 8-row seed leaves the other stream none
        (dict(slot_class="slots_5_8", num_tokens=16), ValueError),
        (dict(slot_class="slots_1_2", batch_size=1, num_tokens=1025), ValueError),
        (dict(num_tokens=32.0), ValueError),
    ],
)
def test_runner_refuses_what_it_cannot_run(kwargs, error) -> None:
    args = dict(_ARGS)
    args.update(kwargs)
    with pytest.raises(error):
        validate_args(**args)


def test_rows_split_evenly_and_short_streams_run_chains() -> None:
    # Catches a ragged split that loses rows, or a stream too short for the class
    # given a tree the class cannot hold.
    assert stream_rows(37, 4) == [10, 9, 9, 9]
    # Too few rows for an even split to hold the 16-row seed: one stream takes it.
    assert stream_rows(24, 8, 16) == [16, 2, 1, 1, 1, 1, 1, 1]
    args = validate_args(**{**_ARGS, "batch_size": 8, "num_tokens": 24, "slot_class": "slots_5_8"})
    assert ref.slot_class_for_parents(launch_trees(args)) == "slots_5_8"
    args = validate_args(**{**_ARGS, "batch_size": 2, "num_tokens": 15})
    trees = launch_trees(args)
    assert [len(t) for t in trees] == [8, 7]
    assert ref.slot_count_for_parents(trees[0]) == 3
    assert trees[1] == list(range(-1, 6))


def test_runner_accepts_long_single_stream_chains() -> None:
    args = validate_args(**{**_ARGS, "batch_size": 1, "num_tokens": 2048, "slot_class": "chain"})
    assert launch_trees(args) == [list(range(-1, 2047))]
