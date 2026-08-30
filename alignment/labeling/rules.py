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

Any of the four neighbour keys may be the sentinel `"<none>"`, which matches
only where the slot is **empty** — the position is the first or last kernel of
its body, so there is no neighbour at all. Without it an absent neighbour is
unnameable: omitting `after` means "don't care", so a rule meant for a body's
first kernel also matches every later one, and the only way to keep it off them
was to let a second rule file overwrite the result. That is how order-dependence
gets in. The sentinel is deliberately *structural* — it asks whether a
neighbouring kernel exists, not whether it happens to be labeled yet — so its
truth does not change as a fixpoint fills the document in.

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

NO_NEIGHBOUR = "<none>"
"""Sentinel for `after`/`after_name`/`before`/`before_name`: the slot is empty.

Angle brackets appear in no measured kernel name and no operation name, so the
sentinel cannot collide with a value it is being compared against."""


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
    source_experiment: str = ""
    """The run whose review produced this rule. A rule set assembled from several
    alignment experiments used to carry that history in its filenames; keeping it
    per rule is what lets the files be merged without losing where a decision
    came from."""

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
            "source_experiment",
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
            source_experiment=record.get("source_experiment", ""),
        )

    def matches(self, position) -> bool:
        if self.name not in position.name:
            return False
        if self.phase is not None and position.phase != self.phase:
            return False
        if self.stream_role is not None and position.stream_role != self.stream_role:
            return False
        # `previous_name` / `next_name` are present for every neighbour that
        # exists, mapped or not, so they are what the `<none>` sentinel is tested
        # against — including when it is written on the operation key.
        if not _neighbour_matches(
            self.after, position.previous_operation, position.previous_name, exact=True
        ):
            return False
        if not _neighbour_matches(
            self.after_name, position.previous_name, position.previous_name, exact=False
        ):
            return False
        if not _neighbour_matches(
            self.before, position.next_operation, position.next_name, exact=True
        ):
            return False
        if not _neighbour_matches(
            self.before_name, position.next_name, position.next_name, exact=False
        ):
            return False
        return True


def _neighbour_matches(
    want: str | None, value: str | None, neighbour_name: str | None, *, exact: bool
) -> bool:
    """One neighbour key against one position.

    `want is None` is "don't care"; `NO_NEIGHBOUR` is "the slot is empty", which
    is decided by whether a neighbouring kernel exists at all rather than by
    whether it carries an operation — an unlabeled neighbour is still a
    neighbour, and a test that said otherwise would change its answer as the
    fixpoint progressed.
    """
    if want is None:
        return True
    if want == NO_NEIGHBOUR:
        return neighbour_name is None
    if value is None:
        return False
    return value == want if exact else want in value


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


def evidence_of(position) -> tuple:
    """Everything `Rule.matches` is allowed to look at, and nothing else.

    Two positions with the same tuple are indistinguishable to every rule, so a
    rule set can be checked against the distinct tuples of a capture instead of
    its hundreds of thousands of kernels.
    """
    return (
        position.name,
        position.phase,
        position.stream_role,
        position.previous_operation,
        position.previous_name,
        position.next_operation,
        position.next_name,
    )


def disagreements(document: dict, rules: list[Rule]) -> list[str]:
    """Positions where two rules match and claim different operations.

    This is the whole of what "first matching rule wins" decides. An empty result
    means no position has a contested reading, so which rule is tried first
    cannot change the outcome and the rules are a *set* rather than a sequence —
    the property that lets them be reordered, regrouped, or split across files
    without the labels moving.

    Run it against a document at the fixpoint: a rule resting on a neighbour's
    operation cannot be judged against a document where that neighbour has not
    been labeled yet.
    """
    seen: set[tuple] = set()
    findings: list[str] = []
    for position in walk_kernels(document):
        evidence = evidence_of(position)
        # `matches` reads nothing but the evidence, so one position per distinct
        # tuple settles it for every position sharing that tuple.
        if evidence in seen:
            continue
        seen.add(evidence)
        claims = {rule.operation for rule in rules if rule.matches(position)}
        if len(claims) > 1:
            findings.append(f"{position.coordinate}: {position.name} claimed by {sorted(claims)}")
    return findings


#: Evidence keys compared for equality, and keys where the rule's value must be
#: contained in the position's. `Rule.matches` splits them the same way.
EXACT_KEYS = ("phase", "stream_role", "after", "before")
SUBSTRING_KEYS = ("name", "after_name", "before_name")


def _subsumes(wide: Rule, narrow: Rule) -> bool:
    """True when every position `narrow` matches, `wide` matches as well.

    Purely syntactic, and deliberately one-directional: an omitted key is no
    constraint, an exact key must be the same value, and a substring key must be
    contained in the narrower rule's fragment — every kernel name containing
    `nvjet_sm100_tst_64x32_64x16` contains `nvjet_sm100_tst_`. The sentinel is
    structural, so it subsumes only itself.

    It never reports an overlap that cannot happen; it does miss some that can
    (two fragments may still co-occur in one name). That asymmetry is the point:
    a hit here is a proof, not a suspicion.
    """
    for key in EXACT_KEYS:
        want = getattr(wide, key)
        if want is not None and getattr(narrow, key) != want:
            return False
    for key in SUBSTRING_KEYS:
        want = getattr(wide, key)
        if want is None:
            continue
        have = getattr(narrow, key)
        if want == NO_NEIGHBOUR or have == NO_NEIGHBOUR:
            if want != have:
                return False
        elif have is None or want not in have:
            return False
    return True


def subsumptions(rules: list[Rule]) -> list[str]:
    """Rule pairs whose labels are decided by which one is tried first.

    When a wider rule claims a different operation than a narrower one it fully
    contains, every position the narrow rule was written for is also claimed by
    the wide one, so the outcome is whichever the applier reaches first. That is
    order-dependence provable from the rule file alone — no capture needed, which
    is what lets it be a pack check rather than something only a labeling run
    finds.

    `disagreements` is the empirical counterpart: it sees overlaps this cannot
    prove, but only for the evidence one capture happens to contain.
    """
    findings: list[str] = []
    for i, wide in enumerate(rules):
        for narrow in rules[i + 1:]:
            if wide.operation == narrow.operation:
                continue
            if _subsumes(wide, narrow):
                first, second = wide, narrow
            elif _subsumes(narrow, wide):
                first, second = narrow, wide
            else:
                continue
            findings.append(
                f"{first.operation} <- {first.name!r} claims everything "
                f"{second.operation} <- {second.name!r} does, so the label depends "
                "on which is tried first"
            )
    return findings


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
