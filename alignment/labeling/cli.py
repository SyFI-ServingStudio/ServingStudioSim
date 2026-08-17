"""`python -m alignment label <command>` — make the manual alignment step checkable.

    initialize start a labeled copy with explicit decisions
    coverage   where is the unmapped time, on both sides
    walk       what ran around this kernel, in program order
    slots      which simulated leaves are available under one prefix
    check      does the inventory contain a defect that reaches a number
    apply      run a rule file over the inventory
    transfer   move reviewed labels onto an identical re-parse

`coverage` needs an analyzer report and so runs after `analyze kernel-align`;
the other commands read only labeling inputs and write only explicit requested
outputs, so they run before it. That is the loop: initialize, inspect positions
and slots, apply rules, check coverage, and re-analyze.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnose import check, format_coverage, format_findings, read_coverage, short_name
from .inventory import (
    load_inventory,
    load_slots,
    save_inventory,
    slots_with_prefix,
    unfold_inventory,
    walk_kernels,
)
from .rules import apply_rule_file
from .transfer import transfer_label_file


def _coverage(args: argparse.Namespace) -> int:
    print(format_coverage(read_coverage(args.report), limit=args.limit))
    return 0


def _walk(args: argparse.Namespace) -> int:
    document = load_inventory(args.inventory)
    shown = 0
    for position in walk_kernels(document):
        if args.phase is not None and position.phase != args.phase:
            continue
        if args.name is not None and args.name not in position.name:
            continue
        operation = position.operation
        if args.unmapped and operation is not None:
            continue
        if shown >= args.limit:
            print(f"… stopped at {args.limit} positions")
            break
        marker = "x" if position.repeat > 1 else " "
        print(
            f"{position.coordinate:>44}{marker}{position.repeat:<4} "
            f"{operation or '-':<38} after={position.previous_operation or '-':<32} "
            f"{short_name(position.name)[:70]} "
            f"before={short_name(position.next_name or '-')[:40]}"
        )
        shown += 1
    return 0


def _initialize(args: argparse.Namespace) -> int:
    document = load_inventory(args.source)
    if args.unfold:
        unfold_inventory(document)
    positions = list(walk_kernels(document))
    labeled = [position.coordinate for position in positions if position.label]
    if labeled:
        raise ValueError(
            "source inventory already contains labels; first labeled positions: "
            + ", ".join(labeled[:5])
        )
    for position in positions:
        position.label.update({"status": "unmapped", "cross_rank": "independent"})
    save_inventory(args.output, document)
    representation = "literal" if args.unfold else "folded"
    print(f"initialized {len(positions)} {representation} positions as explicit unmapped labels")
    return 0


def _slots(args: argparse.Namespace) -> int:
    """One layer's simulated slots in compile order — the other half of `walk`."""
    for name, kind in slots_with_prefix(load_slots(args.manifest), args.prefix):
        print(f"{kind:<24} {name}")
    return 0


def _check(args: argparse.Namespace) -> int:
    findings = check(load_inventory(args.inventory))
    print(format_findings(findings))
    return 1 if any(finding.severity == "error" for finding in findings) else 0


def _apply(args: argparse.Namespace) -> int:
    report = apply_rule_file(args.inventory, args.manifest, args.rules, write=not args.dry_run)
    print(report.format())
    if args.dry_run:
        print("\n(dry run — the inventory was not written)")
    return 0


def _transfer(args: argparse.Namespace) -> int:
    report = transfer_label_file(args.source, args.destination, args.output, write=not args.dry_run)
    print(report.format())
    if args.dry_run:
        print("\n(dry run — nothing was written)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alignment label", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    initialize = subcommands.add_parser(
        "initialize", help="copy a fresh inventory with explicit unmapped labels"
    )
    initialize.add_argument("source", type=Path)
    initialize.add_argument("output", type=Path)
    initialize.add_argument(
        "--unfold",
        action="store_true",
        help="expand repeats so occurrence-specific full-model boundaries can be labeled",
    )
    initialize.set_defaults(handler=_initialize)

    coverage = subcommands.add_parser("coverage", help="unmapped measured kernels and slots")
    coverage.add_argument(
        "report",
        type=Path,
        help="analysis_kernel/reports/alignment_iteration_report.json",
    )
    coverage.add_argument("--limit", type=int, default=20)
    coverage.set_defaults(handler=_coverage)

    walk = subcommands.add_parser("walk", help="kernels in program order with their neighbours")
    walk.add_argument("inventory", type=Path, help="kernel_sequences_labeled.json")
    walk.add_argument("--phase")
    walk.add_argument("--name", help="only kernels whose name contains this fragment")
    walk.add_argument("--unmapped", action="store_true")
    walk.add_argument("--limit", type=int, default=200)
    walk.set_defaults(handler=_walk)

    slots = subcommands.add_parser("slots", help="simulated slots under one prefix, in order")
    slots.add_argument("manifest", type=Path, help="timing_predict/raw/cost_manifest/*.json")
    slots.add_argument("prefix", help="e.g. unified.body.sparse_cycle_full_index")
    slots.set_defaults(handler=_slots)

    check_command = subcommands.add_parser("check", help="defects that reach a published number")
    check_command.add_argument("inventory", type=Path)
    check_command.set_defaults(handler=_check)

    apply_command = subcommands.add_parser("apply", help="run a rule file over the inventory")
    apply_command.add_argument("inventory", type=Path)
    apply_command.add_argument("manifest", type=Path)
    apply_command.add_argument("rules", type=Path)
    apply_command.add_argument("--dry-run", action="store_true")
    apply_command.set_defaults(handler=_apply)

    transfer = subcommands.add_parser(
        "transfer", help="move labels onto a re-parsed inventory of the same capture"
    )
    transfer.add_argument("source", type=Path, help="the labeled inventory")
    transfer.add_argument("destination", type=Path, help="the freshly parsed inventory")
    transfer.add_argument("output", type=Path)
    transfer.add_argument("--dry-run", action="store_true")
    transfer.set_defaults(handler=_transfer)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
