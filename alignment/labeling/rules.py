"""Apply a checked-in rule file to an inventory, so a labeling pass is a diff.

A labeling pass used to be a throwaway script: a list of "this kernel name is
that operation" decisions, run once against one log directory, and afterwards
recoverable only by reading the labeled JSON back. That is the wrong artifact to
keep. The decisions are the reviewable part — a rule states which evidence it
rests on (`name`, `after`, `phase`) and which simulated slots it claims — while
the 40 MB of labeled inventory it produces is derived.

The evidence keys are exactly the ones `operate-run-alignment` allows a
name to be disambiguated by, and no others:

  * `name` — a fragment of the measured kernel name, always required;
  * `after` — the operation of the nearest mapped kernel before it in the same
    segment body. One CUTLASS tile is the router projection in one layer kind
    and the indexer's wk projection in another; the predecessor is what tells
    them apart. A rule with `after` never fires across a body boundary;
  * `after_name` — a fragment of the immediately preceding kernel's name, for
    the same job where the neighbour is not labeled: every projection is
    preceded by the same quant kernel, and what separates those quant kernels
    from each other is what ran before *them*;
  * `before_name` — a fragment of the immediately following kernel's name. It
    distinguishes identical kernels at repeated layer and one-off model
    boundaries;
  * `before` — the immediately following kernel's mapped operation, populated
    by an earlier fixed-point pass;
  * `phase` — `forward` / `postprocess` / …, which is what separates the
    q_absorb batched GEMM from the lm_head that reuses the same tile.
  * `stream_role` — `primary` / `concurrent`, for kernels reused on streams with
    different execution roles.

A rule claims simulated slots by suffix rather than by full name because one
logical slot exists once per layer variant (dense, initial-shared, cycle-full-
index, cycle-shared) and a measured kernel covers all of them. The suffix is
resolved against the run's own cost manifest, so a rule that names a slot the
model no longer emits fails loudly instead of labeling kernels against nothing.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path

from .inventory import load_slots, slots_ending, walk_kernels

CROSS_RANK_VALUES = ("independent", "synchronizing")
STREAM_ROLE_VALUES = ("primary", "concurrent")


@dataclass(frozen=True)
class Rule:
    """One labeling decision, stated in terms of the evidence it rests on."""

    name: str
    """Fragment that must appear in the measured kernel name."""
    operation: str
    type: str
    role: str
    slot_suffixes: tuple[str, ...]
    """Suffixes of the simulated slots this operation is compared against."""
    after: str | None = None
    after_name: str | None = None
    before: str | None = None
    before_name: str | None = None
    phase: str | None = None
    stream_role: str | None = None
    cross_rank: str = "independent"
    overwrite: bool = False
    """Replace an already-mapped label. Off by default: a pass that silently
    overrode earlier decisions would make the order of the rule file matter."""
    note: str = ""
    """Why this rule is the reading of the evidence. Kept in the file, not in
    the labeled output, because it is for the next reader of the rules."""

    @staticmethod
    def from_mapping(record: dict) -> Rule:
        unknown = set(record) - {
            "name",
            "operation",
            "type",
            "role",
            "slot_suffixes",
            "after",
            "after_name",
            "before",
            "before_name",
            "phase",
            "stream_role",
            "cross_rank",
            "overwrite",
            "note",
        }
        if unknown:
            raise ValueError(f"rule has unknown keys: {sorted(unknown)}")
        for required in ("name", "operation", "type", "role", "slot_suffixes"):
            if not record.get(required):
                raise ValueError(f"rule for {record.get('name')!r} has no {required}")
        cross_rank = record.get("cross_rank", "independent")
        if cross_rank not in CROSS_RANK_VALUES:
            raise ValueError(f"cross_rank must be one of {CROSS_RANK_VALUES}, got {cross_rank!r}")
        stream_role = record.get("stream_role")
        if stream_role is not None and stream_role not in STREAM_ROLE_VALUES:
            raise ValueError(
                f"stream_role must be one of {STREAM_ROLE_VALUES}, got {stream_role!r}"
            )
        return Rule(
            name=record["name"],
            operation=record["operation"],
            type=record["type"],
            role=record["role"],
            slot_suffixes=tuple(record["slot_suffixes"]),
            after=record.get("after"),
            after_name=record.get("after_name"),
            before=record.get("before"),
            before_name=record.get("before_name"),
            phase=record.get("phase"),
            stream_role=stream_role,
            cross_rank=cross_rank,
            overwrite=bool(record.get("overwrite", False)),
            note=record.get("note", ""),
        )

    def matches(self, position) -> bool:
        if self.name not in position.name:
            return False
        if self.phase is not None and position.phase != self.phase:
            return False
        if self.stream_role is not None and position.stream_role != self.stream_role:
            return False
        if self.after is not None and position.previous_operation != self.after:
            return False
        if self.after_name is not None and (
            position.previous_name is None or self.after_name not in position.previous_name
        ):
            return False
        if self.before is not None and position.next_operation != self.before:
            return False
        if self.before_name is not None and (
            position.next_name is None or self.before_name not in position.next_name
        ):
            return False
        return True


def load_rules(path: Path) -> list[Rule]:
    document = json.loads(Path(path).read_text())
    records = document["rules"] if isinstance(document, dict) else document
    return [Rule.from_mapping(record) for record in records]


@dataclass
class ApplyReport:
    applied: collections.Counter = field(default_factory=collections.Counter)
    """Kernel positions labeled, per operation."""
    confirmed: collections.Counter = field(default_factory=collections.Counter)
    """Positions that already carried the operation the rule claims. Re-running
    a pass over its own output is all confirmations and no changes, which is how
    a rule file is checked against the inventory it produced."""
    unfired: list[str] = field(default_factory=list)
    """Rules that matched no position at all — almost always a name that
    changed. A rule whose positions are already labeled is confirmed, not
    unfired."""
    conflicts: list[str] = field(default_factory=list)
    """Positions a rule wanted but that already carry a different operation."""

    def format(self) -> str:
        lines = []
        for operation, count in self.applied.most_common():
            lines.append(f"{count:6d}  {operation}")
        if not self.applied:
            lines.append("no positions labeled")
        if self.confirmed:
            lines.append("")
            lines.append(
                f"{sum(self.confirmed.values())} positions across "
                f"{len(self.confirmed)} operations already carried their rule's label"
            )
        if self.unfired:
            lines.append("")
            lines.append("rules that matched nothing:")
            lines += [f"  {rule}" for rule in self.unfired]
        if self.conflicts:
            lines.append("")
            lines.append("positions already mapped to another operation (left alone):")
            lines += [f"  {conflict}" for conflict in self.conflicts]
        return "\n".join(lines)


def label_body(rule: Rule, slots: list[tuple[str, str]]) -> dict:
    """The label a rule writes, with its slot suffixes resolved against the run.

    Every kernel of one operation gets a byte-identical body — the labeled
    inventory loader rejects an operation whose kernels disagree, and it reports
    only the operation name, never which two labels differed.
    """
    resolved: list[str] = []
    for slot in slots_ending(slots, *rule.slot_suffixes):
        if slot not in resolved:
            resolved.append(slot)
    return {
        "status": "mapped",
        "operation": rule.operation,
        "type": rule.type,
        "role": rule.role,
        "simulated_slots": sorted(resolved),
        # Written explicitly even though the analyzer defaults an absent value
        # to independent: an omission here only surfaces as a schema error in a
        # downstream consumer, which is a long way from the decision that caused
        # it.
        "cross_rank": rule.cross_rank,
    }


def apply_rules(document: dict, rules: list[Rule], slots: list[tuple[str, str]]) -> ApplyReport:
    """Label the document in place. First matching rule wins, in file order."""
    bodies = {rule.operation: label_body(rule, slots) for rule in rules}
    report = ApplyReport()
    fired: set[int] = set()

    for position in walk_kernels(document):
        for index, rule in enumerate(rules):
            if not rule.matches(position):
                continue
            mapped_operation = (
                position.label.get("operation")
                if position.label.get("status") == "mapped"
                else None
            )
            fired.add(index)
            if mapped_operation is not None and not rule.overwrite:
                if mapped_operation == rule.operation:
                    report.confirmed[rule.operation] += 1
                else:
                    report.conflicts.append(
                        f"{position.coordinate}: {mapped_operation} kept, "
                        f"rule {rule.name!r} wanted {rule.operation}"
                    )
                break
            position.label.clear()
            position.label.update(bodies[rule.operation])
            report.applied[rule.operation] += 1
            break

    report.unfired = [
        f"{rule.operation} <- {rule.name!r}"
        for index, rule in enumerate(rules)
        if index not in fired
    ]
    return report


def apply_rule_file(
    inventory_path: Path, manifest_path: Path, rules_path: Path, write: bool
) -> ApplyReport:
    from .inventory import load_inventory, save_inventory

    document = load_inventory(inventory_path)
    slots = load_slots(manifest_path)
    report = apply_rules(document, load_rules(rules_path), slots)
    if write:
        save_inventory(inventory_path, document)
    return report
