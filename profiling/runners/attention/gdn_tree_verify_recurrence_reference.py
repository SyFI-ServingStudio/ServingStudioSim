"""References for ``gdn_tree_verify_recurrence``: the slot-class rule and a Torch oracle.

Slot class (pure Python, no Torch)
----------------------------------
TensorFold (perf-study 6469730a, ``tensorfold/cuda/kernels/gdn.py`` ``schedule`` /
``_order_slots``) visits each stream's tree in one order and keeps a parent's state
in a shared-memory *slot* while some child of it does not run right after it. A
launch compiles one ``tree_kernel`` variant from the largest slot count of any stream
in it (``gdn.cu`` ``dispatch_tree``). :func:`slot_count_for_parents` restates that
rule; :func:`slot_class_for_parents` maps a launch's trees to the kind's
``slot_class`` value. An arch reproduces the key from its draft trees with these two
functions alone.

The rule, for one stream's ``parents`` (row 0 the root ``-1``, every parent before
its child):

1. Two visiting orders: depth-first preorder (children in row order) and level order
   (by depth, then row).
2. For an order, walk it; a node's state goes to a new slot when any of its children
   is not the very next visited node. A slot is released when the last-visited child
   of its node runs, and the lowest released slot is reused before a new one opens.
   The order's count is the most slots ever open.
3. The stream needs the smaller of the two counts (``schedule`` refuses a tree that
   needs more than 32).

So a chain needs 0, a star 1, a 16-row binary heap 3; a chain tail added after a
tree's last row adds nothing.

Oracle
------
:func:`gdn_tree_verify_recurrence_reference` is the operation itself, independent of
TensorFold: for every window row, the fp32 gated delta rule stepped along the row's
root-to-row path from its stream's committed state, then the read-out. With ``S`` the
``(value, key)`` state of head ``h`` and key head ``h // (Hv / Hk)``, a row does
``S <- g S; S <- S + beta (v - S k) k^T; o = S q``. The committed states are not
changed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: ``slot_class`` values, in kernel-variant order. Each covers the slot counts that
#: compile one ``tree_kernel`` variant on a GPU outside ``dispatch_tree``'s
#: many-SM sm_120 branch (``gdn.cu``): template ``SLOTS`` 0, 2, 4, 8, 16, 32.
SLOT_CLASSES: tuple[str, ...] = (
    "chain",
    "slots_1_2",
    "slots_3_4",
    "slots_5_8",
    "slots_9_16",
    "slots_17_32",
)
_CLASS_UPPER = (0, 2, 4, 8, 16, 32)
MAX_SLOTS = 32


def _check_parents(parents: Sequence[int]) -> list[int]:
    out = [int(p) for p in parents]
    if (
        not out
        or out[0] != -1
        or any(not -1 <= p < i or (p == -1) != (i == 0) for i, p in enumerate(out))
    ):
        raise ValueError("a tree needs one root at row zero and parents before their children")
    return out


def _order_slot_count(parents: list[int], order: list[int]) -> int:
    position = {node: i for i, node in enumerate(order)}
    children: dict[int, list[int]] = {}
    for node, parent in enumerate(parents):
        if parent >= 0:
            children.setdefault(parent, []).append(node)
    last_child = {node: max(position[child] for child in kids) for node, kids in children.items()}
    open_slots: set[int] = set()  # nodes whose state sits in a slot
    used = 0
    for i, node in enumerate(order):
        parent = parents[node]
        if parent in open_slots and last_child[parent] == i:
            open_slots.discard(parent)
        if any(position[child] != i + 1 for child in children.get(node, ())):
            open_slots.add(node)
            used = max(used, len(open_slots))
    return used


def slot_count_for_parents(parents: Sequence[int]) -> int:
    """Shared-memory states TensorFold's ``schedule`` needs for one stream's tree."""
    tree = _check_parents(parents)
    children: dict[int, list[int]] = {}
    depth = [0] * len(tree)
    for node, parent in enumerate(tree[1:], start=1):
        children.setdefault(parent, []).append(node)
        depth[node] = depth[parent] + 1
    preorder, stack = [], [0]
    while stack:
        node = stack.pop()
        preorder.append(node)
        stack.extend(reversed(children.get(node, ())))
    level = sorted(range(len(tree)), key=lambda n: (depth[n], n))
    need = min(_order_slot_count(tree, preorder), _order_slot_count(tree, level))
    if need > MAX_SLOTS:
        raise ValueError(f"this tree needs {need} live states; TensorFold runs at most {MAX_SLOTS}")
    return need


def slot_class_of_count(slots: int) -> str:
    """The ``slot_class`` value covering ``slots`` live states."""
    if not 0 <= slots <= MAX_SLOTS:
        raise ValueError(f"slot count must lie in 0..{MAX_SLOTS}, got {slots}")
    return SLOT_CLASSES[next(i for i, upper in enumerate(_CLASS_UPPER) if slots <= upper)]


def slot_class_for_parents(parents_lists: Sequence[Sequence[int]]) -> str:
    """A launch's ``slot_class``: the class of the largest slot count over its streams."""
    if not parents_lists:
        raise ValueError("a launch has at least one stream")
    return slot_class_of_count(max(slot_count_for_parents(p) for p in parents_lists))


# Smallest trees (found by search over all trees of up to 16 rows) that need the
# lowest slot count of each class reachable within 16 rows. A 16-row tree needs at
# most 5 slots; classes from slots_9_16 up need far larger trees.
_CLASS_SEEDS: dict[str, tuple[int, ...]] = {
    "chain": (-1,),
    "slots_1_2": (-1, 0, 0),
    "slots_3_4": (-1, 0, 0, 1, 1, 2, 3, 3),
    "slots_5_8": (-1, 0, 1, 0, 2, 4, 4, 2, 2, 8, 6, 5, 1, 4, 7, 6),
}


def min_rows_for_class(slot_class: str) -> int:
    """Fewest rows of a canonical tree in ``slot_class``; a class without a seed is unsupported."""
    if slot_class not in _CLASS_SEEDS:
        raise ValueError(f"no canonical tree for slot_class {slot_class!r}")
    return len(_CLASS_SEEDS[slot_class])


def canonical_tree(slot_class: str, rows: int) -> list[int]:
    """The runner's tree: the class's seed, then a chain hanging off its last row.

    The chain tail adds no slot, so every row count from the seed's up stays in the
    class; the tail's rows read their parent from registers, as most rows of a
    best-first draft tree do.
    """
    seed = list(_CLASS_SEEDS.get(slot_class, ()))
    if not seed:
        raise ValueError(f"no canonical tree for slot_class {slot_class!r}")
    if rows < len(seed):
        raise ValueError(f"slot_class {slot_class} needs at least {len(seed)} rows, got {rows}")
    return seed + list(range(len(seed) - 1, rows - 1))


def path_rows(parents: Sequence[int]) -> list[list[int]]:
    """Each row's root-to-row path (window-local rows)."""
    paths: list[list[int]] = []
    for row, parent in enumerate(_check_parents(parents)):
        paths.append((paths[parent] if parent >= 0 else []) + [row])
    return paths


def gdn_tree_verify_recurrence_reference(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    g: Any,
    beta: Any,
    states: Sequence[Any],
    parents_lists: Sequence[Sequence[int]],
) -> Any:
    """fp32 ``(W, Hv, Dv)`` outputs for streams back to back in the row tensors.

    ``q``/``k`` ``(W, Hk, Dk)``, ``v`` ``(W, Hv, Dv)``, ``g``/``beta`` ``(W, Hv)``;
    ``states[s]`` is stream ``s``'s ``(Hv, Dv, Dk)`` fp32 committed state (read only).
    """
    group = v.shape[1] // q.shape[1]
    qf = q.float().repeat_interleave(group, dim=1)
    kf = k.float().repeat_interleave(group, dim=1)
    vf, gf, bf = v.float(), g.float(), beta.float()
    out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    base = 0
    for state, parents in zip(states, parents_lists, strict=True):
        node_state: list[Any] = []
        for row, parent in enumerate(_check_parents(parents)):
            t = base + row
            s = (node_state[parent] if parent >= 0 else state.float()) * gf[t][:, None, None]
            memory = torch.einsum("hvk,hk->hv", s, kf[t])
            s = s + ((vf[t] - memory) * bf[t][:, None])[:, :, None] * kf[t][:, None, :]
            out[t] = torch.einsum("hvk,hk->hv", s, qf[t])
            node_state.append(s)
        base += len(parents)
    if base != q.shape[0]:
        raise ValueError(f"trees hold {base} rows, the inputs {q.shape[0]}")
    return out
