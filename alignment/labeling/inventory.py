"""Read the two artifacts every labeling decision is made against.

The labeled kernel inventory (`kernel_sequences_labeled.json`) is a *folded*
program: a phase holds unique sequences, a sequence holds segments, and a
segment is either a flat kernel list or a repeat body. Nothing downstream wants
that nesting — every question ("what ran right before this kernel", "which
kernels are still unmapped", "does this name mean two different things") is
asked in program order — so `walk_kernels` flattens it once, keeping the folded
coordinates on each position so a finding can be pointed back at the file.

The other artifact is the timing-predict cost manifest, which names every
simulated slot in compile order. That order is the strongest evidence available
when deciding what an unfamiliar kernel is: a decoder layer's measured kernels
and its slots are the same list twice.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelPosition:
    """One kernel in program order, with the folded coordinates to find it again."""

    phase: str
    sequence_id: str
    segment_index: int
    offset: int
    """Index inside the segment body — NOT the expanded iteration position."""
    repeat: int
    """How many times the enclosing segment body runs (1 for a flat segment)."""
    name: str
    label: dict
    """The live label dict. Mutating it edits the document being walked."""
    previous_name: str | None = None
    """The kernel name immediately before this one in the same segment body.
    Weaker evidence than the predecessor's operation but always present, which
    is what separates the quant kernel before `o_proj` from the one before
    `gate_up` when neither of their neighbours is labeled yet."""
    previous_operation: str | None = None
    """The operation of the nearest mapped kernel before this one in the same
    segment body, which is the `ordered neighbours` evidence: one CUTLASS tile
    serves several roles and what ran just before it is what tells them apart.
    None at a body's first kernel and after any unmapped kernel — an unmapped
    neighbour carries no evidence and must not be silently skipped over."""
    next_name: str | None = None
    """The immediately following kernel in the same body. This distinguishes
    an implementation-identical repeated layer boundary from a one-off model
    boundary such as the final norm."""

    @property
    def operation(self) -> str | None:
        return self.label.get("operation") if self.label.get("status") == "mapped" else None

    @property
    def coordinate(self) -> str:
        return f"{self.phase}/{self.sequence_id[:12]}/seg{self.segment_index}[{self.offset}]"


def load_inventory(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_inventory(path: Path, document: dict) -> None:
    Path(path).write_text(json.dumps(document, indent=2))


def segment_body(segment: dict) -> list[dict]:
    """The kernel list of a flat segment or of a repeat segment's body."""
    if "kernels" in segment:
        return segment["kernels"]
    return segment["repeat"]["body"]["kernels"]


def segment_repeat(segment: dict) -> int:
    return 1 if "kernels" in segment else int(segment["repeat"]["count"])


def unfold_inventory(document: dict) -> None:
    """Expand every repeat into one literal program node, losslessly in place.

    Folding remains the compact default. Full-model boundary labeling is the
    exception: the last occurrence of an identical layer kernel can own a
    different CostTree slot and therefore must remain individually addressable.
    """
    for phase in document["phases"].values():
        for sequence in phase["unique_sequences"]:
            expanded: list[dict] = []
            for segment in sequence["program"]:
                body = segment_body(segment)
                for _ in range(segment_repeat(segment)):
                    expanded.extend(copy.deepcopy(body))
            expected_count = int(sequence["expanded_kernel_count"])
            if len(expanded) != expected_count:
                raise ValueError(
                    f"sequence {sequence['sequence_id']!r} unfolded to {len(expanded)} "
                    f"kernels, expected {expected_count}"
                )
            sequence["program"] = [{"kernels": expanded}]
    document["encoding"] = "literal-v1"
    document["folding_policy"] = {"kind": "none", "source": "label-initialize-unfold"}


def walk_kernels(document: dict) -> Iterator[KernelPosition]:
    """Every kernel in program order, phase by phase and sequence by sequence.

    Order matters: the `after` evidence (what ran immediately before) is only
    meaningful within one segment body, and consumers rely on this walk not
    stitching two bodies together.

    `previous_operation` is read back from the label dict after each position is
    consumed, so a caller that labels a kernel in place is the predecessor of
    the next one — which is what lets a chain (`router_gemm` then its split-K
    reduce) be labeled in a single pass.
    """
    for phase, block in document["phases"].items():
        for sequence in block["unique_sequences"]:
            for segment_index, segment in enumerate(sequence["program"]):
                repeat = segment_repeat(segment)
                previous_name: str | None = None
                previous_operation: str | None = None
                body = segment_body(segment)
                for offset, kernel in enumerate(body):
                    label = kernel.setdefault("label", {})
                    yield KernelPosition(
                        phase=phase,
                        sequence_id=sequence["sequence_id"],
                        segment_index=segment_index,
                        offset=offset,
                        repeat=repeat,
                        name=kernel["name"],
                        label=label,
                        previous_name=previous_name,
                        previous_operation=previous_operation,
                        next_name=body[offset + 1]["name"] if offset + 1 < len(body) else None,
                    )
                    previous_operation = (
                        label.get("operation") if label.get("status") == "mapped" else None
                    )
                    previous_name = kernel["name"]


def load_slots(manifest_path: Path) -> list[tuple[str, str]]:
    """`(slot name, kernel kind)` in compile order, de-duplicated.

    A slot name repeats once per fan-out child (per DP group, per EP rank); the
    labeling side only ever cares about the distinct names and their order.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    seen: set[str] = set()
    slots: list[tuple[str, str]] = []
    for section in manifest["sections"]:
        for slot in section["slots"]:
            if slot["name"] in seen:
                continue
            seen.add(slot["name"])
            slots.append((slot["name"], slot["kind"]))
    return slots


def slots_ending(slots: list[tuple[str, str]], *suffixes: str) -> list[str]:
    """Every layer variant's copy of one logical slot.

    A decoder's `attention.q_absorb` exists once per layer kind (dense,
    initial-shared, cycle-full-index, cycle-shared). One measured kernel name
    covers all of them, so a label lists them all.
    """
    found = sorted(
        name for name, _ in slots if any(name.endswith("." + suffix) for suffix in suffixes)
    )
    if not found:
        raise ValueError(f"no simulated slot ends with any of {suffixes}")
    return found


def slots_with_prefix(slots: list[tuple[str, str]], prefix: str) -> list[tuple[str, str]]:
    """One layer's slots in compile order, with the shared prefix stripped."""
    return [
        (name[len(prefix) :].lstrip("."), kind) for name, kind in slots if name.startswith(prefix)
    ]
