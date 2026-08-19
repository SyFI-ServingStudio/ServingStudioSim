"""Move a labeling pass onto a freshly parsed inventory of the same capture.

Re-parsing a capture — after a parser fix, or with a different window — rebuilds
the inventory from the same kernels, so the labels a person spent hours writing
are still correct. What changes is the bookkeeping around them: which device and
which step each sequence occurred at.

Transfer is therefore allowed only when the two documents are the *same program*:
identical phases, identical sequence ids, identical segment structure, and
identical kernel names and categories at every position. That is a strict
equality check and not a similarity score — a fuzzy transfer silently charges one
kernel's label to another, which is exactly the failure mode labeling exists to
prevent. When the structures differ the transfer refuses and names the first
position that disagrees.

Occurrences, provenance and folding policy are the destination's own. Only the
`label` dict moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .inventory import load_inventory, save_inventory, segment_body, segment_repeat


@dataclass(frozen=True)
class TransferReport:
    sequences: int
    positions: int
    transferred: int
    """Positions that carried a label in the source and now carry it here."""
    unlabeled: int
    """Positions the source had no label for. Not an error: a fresh inventory
    out of `alignment parse` looks like this, and coverage is where it shows."""

    def format(self) -> str:
        return (
            f"{self.sequences} sequences, {self.positions} positions\n"
            f"  transferred  {self.transferred}\n"
            f"  unlabeled    {self.unlabeled}"
        )


def _program_shape(document: dict) -> list[tuple]:
    """Everything about the document that a label's meaning depends on."""
    shape: list[tuple] = []
    for phase in sorted(document["phases"]):
        for sequence in document["phases"][phase]["unique_sequences"]:
            for track in sequence["tracks"]:
                track_index = int(track["track_index"])
                for segment_index, segment in enumerate(track["program"]):
                    for offset, kernel in enumerate(segment_body(segment)):
                        shape.append(
                            (
                                phase,
                                sequence["sequence_id"],
                                track_index,
                                segment_index,
                                segment_repeat(segment),
                                offset,
                                kernel["name"],
                                kernel.get("suggested_category"),
                            )
                        )
    return sorted(shape)


def _describe(position: tuple) -> str:
    phase, sequence_id, track_index, segment_index, repeat, offset, name, category = position
    return (
        f"{phase}/{sequence_id[:12]}/track{track_index}"
        f"/seg{segment_index}(x{repeat})[{offset}] {name} [{category}]"
    )


def _assert_same_program(source: dict, destination: dict) -> None:
    source_shape = _program_shape(source)
    destination_shape = _program_shape(destination)
    for left, right in zip(source_shape, destination_shape):
        if left != right:
            raise ValueError(
                "the two inventories are not the same program, so labels cannot be "
                f"transferred:\n  source      {_describe(left)}\n  destination {_describe(right)}"
            )
    if len(source_shape) != len(destination_shape):
        longer, shorter = (
            ("source", destination_shape)
            if len(source_shape) > len(destination_shape)
            else ("destination", source_shape)
        )
        extra = (source_shape if longer == "source" else destination_shape)[len(shorter)]
        raise ValueError(
            f"the {longer} inventory has {abs(len(source_shape) - len(destination_shape))} "
            f"positions the other does not, first one at {_describe(extra)}"
        )


def _labels_by_position(document: dict) -> dict[tuple[str, str, int, int, int], dict]:
    labels: dict[tuple[str, str, int, int, int], dict] = {}
    for phase, block in document["phases"].items():
        for sequence in block["unique_sequences"]:
            for track in sequence["tracks"]:
                track_index = int(track["track_index"])
                for segment_index, segment in enumerate(track["program"]):
                    for offset, kernel in enumerate(segment_body(segment)):
                        # An empty dict is what `walk_kernels` leaves behind on a
                        # position nobody labeled, so it carries no decision to move.
                        if kernel.get("label"):
                            key = (
                                phase,
                                sequence["sequence_id"],
                                track_index,
                                segment_index,
                                offset,
                            )
                            labels[key] = kernel["label"]
    return labels


def transfer_labels(source: dict, destination: dict) -> TransferReport:
    """Copy every label from `source` onto the identical position in `destination`."""
    _assert_same_program(source, destination)
    labels = _labels_by_position(source)
    sequences = 0
    positions = 0
    transferred = 0
    for phase, block in destination["phases"].items():
        for sequence in block["unique_sequences"]:
            sequences += 1
            for track in sequence["tracks"]:
                track_index = int(track["track_index"])
                for segment_index, segment in enumerate(track["program"]):
                    for offset, kernel in enumerate(segment_body(segment)):
                        positions += 1
                        label = labels.get(
                            (
                                phase,
                                sequence["sequence_id"],
                                track_index,
                                segment_index,
                                offset,
                            )
                        )
                        if label is None:
                            kernel.pop("label", None)
                            continue
                        kernel["label"] = label
                        transferred += 1
    return TransferReport(
        sequences=sequences,
        positions=positions,
        transferred=transferred,
        unlabeled=positions - transferred,
    )


def transfer_label_file(
    source_path: Path, destination_path: Path, output_path: Path, *, write: bool = True
) -> TransferReport:
    source = load_inventory(source_path)
    destination = load_inventory(destination_path)
    report = transfer_labels(source, destination)
    if write:
        save_inventory(output_path, destination)
    return report
