"""`alignment-campaign label` — apply a pack's rules to a fixpoint.

Rules are position-sensitive: `Rule.matches` can test
`position.previous_operation` / `next_operation`, which only exist once an
earlier pass has written them. So one sweep is not enough, and the existing
practice was to hardcode a pass count per campaign (2 for one batch, 5 for
another, 2 for a third). Those numbers are guesses that happen to have been
large enough.

The termination condition is **the label state stopped changing**, not "no rule
fired". Those differ: `apply_rules` counts a position in `applied` whenever an
`overwrite` rule rewrites it, even when the new body is byte-identical, so a
driver waiting for zero runs to its sweep cap without ever being wrong about the
labels. Comparing the state instead terminates the GLM rules in 3-6 sweeps on
every one of the fifteen cases.

**Order-independence is enforced, not assumed.** `apply_rules` resolves a
contested position by "first matching rule wins", which makes position in the
file a silent priority — and the rule sets this drives used to depend on it, as
seven files whose later `overwrite` entries corrected earlier wider matchers.
That is a property of the *rules*, so this checks the rules: after the fixpoint,
`disagreements` looks for any position two rules claim with different
operations. None means the outcome cannot depend on the order they were tried
in, and the set can be reordered, regrouped or split without the labels moving.
Any hit is a hard failure — it is the one defect that reproduces as "the numbers
changed and the diff looks like nothing happened".

Two report fields are signals rather than noise:

- `conflicts` records a rule wanting a position that already carries a different
  operation. With no disagreements this can only be a rule confirming across
  sweeps, so it is reported, not fatal.
- `unfired` is a drift warning. A rule that matches nothing anywhere in the pack
  is almost always a kernel name that changed underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alignment.labeling.diagnose import check as check_inventory
from alignment.labeling.inventory import load_inventory, load_slots, save_inventory, walk_kernels
from alignment.labeling.rules import apply_rules, disagreements, load_rules

from .pack import Pack, PackError, Variant
from .render import TIMING_PREDICT_PHASE

LABELED_NAME = "kernel_sequences_labeled.json"
INVENTORY_NAME = "kernel_sequences.json"
COST_MANIFEST_DIR = Path("raw/cost_manifest")

#: Guard against a rule set that oscillates (rule A relabels what rule B just
#: labeled and back). Hitting this is a defect in the rules, not a tuning knob.
MAX_SWEEPS = 25


@dataclass
class LabelReport:
    case_slug: str
    sweeps: int = 0
    applied: dict[str, int] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    unfired: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    converged: bool = False

    @property
    def ok(self) -> bool:
        """Converged, order-independent, and no defect that reaches a number."""
        return self.converged and not self.findings and not self.disagreements

    def format(self) -> str:
        lines = [
            f"{self.case_slug}: {sum(self.applied.values())} positions labeled over "
            f"{self.sweeps} sweep(s)"
            + ("" if self.converged else f" — did NOT converge within {MAX_SWEEPS}")
        ]
        for operation, count in sorted(self.applied.items(), key=lambda item: -item[1])[:10]:
            lines.append(f"  {count:6d}  {operation}")
        if self.disagreements:
            lines.append(
                f"  ORDER-DEPENDENT: {len(self.disagreements)} position(s) are claimed by "
                "two rules with different operations, so the label depends on which rule "
                "is tried first. Narrow the matchers until each position has one reading:"
            )
            lines += [f"    {item}" for item in self.disagreements[:5]]
        if self.conflicts:
            lines.append(
                f"  {len(self.conflicts)} position(s) a rule wanted but an earlier "
                "decision kept:"
            )
            lines += [f"    {item}" for item in self.conflicts[:5]]
        if self.unfired:
            lines.append(f"  {len(self.unfired)} rule(s) matched nothing:")
            lines += [f"    {item}" for item in self.unfired[:10]]
        if self.findings:
            lines.append("  check findings:")
            lines += [f"    {item}" for item in self.findings[:10]]
        return "\n".join(lines)


def _label_state(document: dict) -> tuple:
    """The labeling decision, independent of how many times it was rewritten.

    Only `status` and `operation` are compared: the rest of a label body is
    derived from the rule that wrote it, so two rules producing the same
    operation produce the same body.
    """
    return tuple(
        (position.coordinate, position.label.get("status"), position.label.get("operation"))
        for position in walk_kernels(document)
    )


def cost_manifest_path(case_dir: Path) -> Path:
    """The timing-predict cost manifest whose slot names the rules resolve
    against. There is one per worker; the labeling side only reads slot names
    and order, which every worker's copy agrees on."""
    directory = case_dir / TIMING_PREDICT_PHASE / COST_MANIFEST_DIR
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        raise PackError(
            f"no cost manifest under {directory} — run the {TIMING_PREDICT_PHASE} phase first"
        )
    return candidates[0]


def rule_files(pack: Pack, variant: Variant) -> list[Path]:
    """The pack's rule files.

    Still a list, because a pack may want its rules split by subsystem once there
    are enough of them — but the split is now for readers only. `rule_files`
    replaced `ordered_rule_files` when the ordering stopped meaning anything;
    `schema_version` 1 manifests used the old key and a fixed order.
    """
    import json

    manifest_path = pack.root / variant.label_rules
    manifest = json.loads(manifest_path.read_text())
    if "ordered_rule_files" in manifest:
        raise PackError(
            f"{manifest_path} uses `ordered_rule_files`, which meant the labels depended on "
            "the order the files were applied in. Merge them into an order-independent set "
            "and declare it under `rule_files`."
        )
    return [manifest_path.parent / name for name in manifest["rule_files"]]


def label_case(
    pack: Pack,
    variant: Variant,
    case_dir: Path,
    kernel_pass_name: str,
    *,
    refresh: bool = False,
) -> LabelReport:
    """Initialize if needed, then apply the rules to a fixpoint and check."""
    case_dir = Path(case_dir)
    report = LabelReport(case_slug=case_dir.name)
    labeled_path = case_dir / LABELED_NAME
    source_path = case_dir / kernel_pass_name / INVENTORY_NAME

    if refresh or not labeled_path.is_file():
        if not source_path.is_file():
            raise PackError(
                f"{source_path} is absent — run the {kernel_pass_name} phase first"
            )
        document = load_inventory(source_path)
        positions = list(walk_kernels(document))
        already = [item.coordinate for item in positions if item.label]
        if already:
            raise PackError(
                f"{source_path} already carries labels at {already[:3]}; labeling must "
                "start from an unlabeled parse so the result is attributable to the rules"
            )
        for position in positions:
            position.label.update({"status": "unmapped", "cross_rank": "independent"})
        save_inventory(labeled_path, document)

    slots = load_slots(cost_manifest_path(case_dir))
    rules = [rule for path in rule_files(pack, variant) for rule in load_rules(path)]
    document = load_inventory(labeled_path)

    applied: dict[str, int] = {}
    previous = _label_state(document)
    for sweep in range(1, MAX_SWEEPS + 1):
        report.sweeps = sweep
        result = apply_rules(document, rules, slots)
        for operation, count in result.applied.items():
            applied[operation] = applied.get(operation, 0) + count
        current = _label_state(document)
        if current == previous:
            # The fixpoint: a whole sweep left every label where it was. Report
            # the LAST sweep's conflicts and unfired rules — the earlier sweeps
            # saw a partially labeled document, so their neighbour-dependent
            # rules had not had their chance yet.
            report.converged = True
            report.unfired = result.unfired
            report.conflicts = result.conflicts
            break
        previous = current

    # At the fixpoint and not before: a rule resting on a neighbour's operation
    # cannot be judged against a document where that neighbour is still unmapped.
    report.disagreements = disagreements(document, rules)
    report.applied = applied
    save_inventory(labeled_path, document)
    report.findings = [
        f"{finding.severity}: {finding.summary}"
        for finding in check_inventory(load_inventory(labeled_path))
        if finding.severity == "error"
    ]
    return report
