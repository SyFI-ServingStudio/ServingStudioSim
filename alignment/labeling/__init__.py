"""Tools for the one manual step of an alignment run: labeling the inventory.

Every other phase is a command. Labeling is a person deciding, for each measured
kernel, which modelled operation it is — and the quality of that decision sets
the quality of every number downstream, because an unlabeled kernel is time the
comparison never sees and a mislabeled one is time charged to the wrong
operation. Both failures are silent: coverage looks like a percentage nobody
reads, and a wrong label produces a deviation that looks like a cost-model bug.

So this package holds the small amount of machinery that makes those two
failures visible — `diagnose` for where the unmapped time is and which labels
contradict each other, `inventory` for reading a position and its neighbours,
`rules` for making a pass a reviewable file rather than a one-off script, and
`transfer` for carrying a finished pass onto a re-parse of the same capture.

A third failure is quieter still: a rule set whose labels depend on the order
the rules happen to be listed in. `subsumptions` and `disagreements` are the two
halves of ruling it out — the first proves an order-dependent pair from the rule
text, the second observes one against a real capture.
"""

from .diagnose import Coverage, Finding, check, format_coverage, format_findings, read_coverage
from .inventory import (
    KernelPosition,
    load_inventory,
    load_slots,
    save_inventory,
    slots_ending,
    slots_with_prefix,
    walk_kernels,
)
from .rules import (
    ApplyReport,
    Rule,
    apply_rule_file,
    apply_rules,
    disagreements,
    load_rules,
    subsumptions,
)
from .transfer import TransferReport, transfer_label_file, transfer_labels

__all__ = [
    "ApplyReport",
    "Coverage",
    "Finding",
    "KernelPosition",
    "Rule",
    "TransferReport",
    "apply_rule_file",
    "apply_rules",
    "check",
    "disagreements",
    "format_coverage",
    "format_findings",
    "load_inventory",
    "load_rules",
    "load_slots",
    "read_coverage",
    "save_inventory",
    "slots_ending",
    "slots_with_prefix",
    "subsumptions",
    "transfer_label_file",
    "transfer_labels",
    "walk_kernels",
]
