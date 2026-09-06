"""`python -m launcher alignment-campaign <verb>` — the campaign command surface.

A top-level command beside `alignment` / `timing-predict` / `kernel-profile`
rather than a subcommand of `alignment`: that parser describes itself as running
"one explicit stage", and a campaign is the layer that drives stages. Nesting
would also read as `alignment campaign run --phase simulation` invoking
`alignment sim` — the same prefix looping back on itself.

    check    validate a pack; pure CPU, this is what the pytest gate calls
    render   write each case's run directory from (pack, host)
    run      one phase across every ready case
    label    apply the pack's rule set to a fixpoint
    extract  completed reports -> one metrics document
    compare  judge that document against tolerances and the golden

`--phase` values are artifact directory names, so the phase, its config, its
output directory and its `.complete` marker all share one name.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import check as check_module
from . import compare as compare_module
from . import execute
from . import extract as extract_module
from . import label as label_module
from . import render as render_module
from .metrics import metric_names, metric_specs
from .pack import REPO_ROOT, Case, Pack, PackError, load_host, load_pack

PACK_ROOT = REPO_ROOT / "presets" / "alignment"
HOST_ROOT = PACK_ROOT / "hosts"


def _resolve_pack(value: str | None) -> Pack | None:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_dir():
        candidate = PACK_ROOT / value
    return load_pack(candidate)


def _resolve_host(value: str):
    candidate = Path(value)
    if not candidate.is_file():
        for suffix in (".yaml", ".yml", ".json"):
            probe = HOST_ROOT / f"{value}{suffix}"
            if probe.is_file():
                candidate = probe
                break
    return load_host(candidate)


def _selected_cases(pack: Pack, names: list[str] | None) -> list[Case]:
    if not names:
        return list(pack.cases)
    chosen: list[Case] = []
    for name in names:
        case = pack.case_named(name)
        if case is None:
            raise PackError(
                f"no case {name!r} in pack {pack.name}; "
                f"available: {[item.slug for item in pack.cases]}"
            )
        chosen.append(case)
    return chosen


def _acceptance_schemas(pack: Pack | None) -> dict[str, int] | None:
    if pack is None:
        return None
    declared = pack.acceptance.get("analyzer_schema")
    if not isinstance(declared, dict):
        return None
    return {key: int(value) for key, value in declared.items()}


# ── verbs ────────────────────────────────────────────────────────────────────


def _check(args) -> int:
    packs = _discover_packs(args.pack)
    worst = 0
    for pack in packs:
        if args.update_invariants:
            path = check_module.write_invariants(pack)
            print(f"[{pack.name}] wrote {path.relative_to(REPO_ROOT)}")
        host = _resolve_host(args.host) if args.host else None
        findings = check_module.check_pack(
            pack, host=host, metric_names=metric_names(_acceptance_schemas(pack))
        )
        errors = [item for item in findings if item.level == "error"]
        for finding in findings:
            print(f"[{pack.name}] {finding}")
        print(
            f"[{pack.name}] {len(pack.cases)} case(s), {len(pack.variants)} variant(s): "
            f"{len(errors)} error(s), {len(findings) - len(errors)} warning(s)"
        )
        worst = max(worst, 1 if errors else 0)
    return worst


def _discover_packs(value: str | None) -> list[Pack]:
    """One pack, or every pack under `presets/alignment/` when none is named —
    so a new pack is covered by the gate the moment it lands."""
    if value is not None:
        pack = _resolve_pack(value)
        assert pack is not None
        return [pack]
    if not PACK_ROOT.is_dir():
        return []
    return [
        load_pack(child)
        for child in sorted(PACK_ROOT.iterdir())
        if child.is_dir() and (child / "campaign.yaml").is_file()
    ]


def _render(args) -> int:
    pack = _resolve_pack(args.pack)
    assert pack is not None
    host = _resolve_host(args.host)
    out_root = Path(args.out_root)
    for case in _selected_cases(pack, args.case):
        rendered = render_module.render_case(pack, case, host, out_root, REPO_ROOT)
        print(f"[render] {rendered.directory}  ({len(rendered.files)} files)")
    first_phase = render_module.phase_names(next(iter(pack.variants.values())))[0]
    print(f"[render] host {host.name}; next: `alignment-campaign run --phase {first_phase}`")
    return 0


def _run(args) -> int:
    pack = _resolve_pack(args.pack)
    assert pack is not None
    cases = _selected_cases(pack, args.case)
    out_root = Path(args.out_root)
    if args.dry_run:
        plans = execute.plan_phase(
            pack,
            out_root,
            args.phase,
            cases=cases,
            refresh=args.refresh,
            resume=args.resume,
        )
        print(execute.describe_plan(args.phase, plans))
        _print_calibration_prerequisites(pack)
        return 0
    report = execute.run_phase(
        pack,
        out_root,
        args.phase,
        cases=cases,
        refresh=args.refresh,
        resume=args.resume,
        parallelism=args.parallelism,
    )
    for result in report.results:
        status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
        print(f"[{args.phase}] {result.plan.case_slug}: {status}")
    for plan in report.blocked:
        print(f"[{args.phase}] {plan.case_slug}: not run — {'; '.join(plan.reasons)}")
    if report.failures:
        print(
            f"[{args.phase}] {len(report.failures)} case(s) failed; the others kept their "
            "artifacts and markers"
        )
        return 1
    return 0


def _print_calibration_prerequisites(pack: Pack) -> None:
    """List the preflight measurements a run still depends on. Never executed
    automatically: their result is an edit to the pack's calibrated block, which
    is a change a person should see."""
    provisional = pack.provisional_fields
    if not provisional:
        return
    print("")
    print("[calibration] these inputs are still provisional and were not measured:")
    for name in provisional:
        print(f"  {name}")
    print(
        "  Run the saturation / workload-only preflight, then edit the pack's "
        "`calibrated` values; `compare --record` refuses these unless "
        "--accept-provisional is given."
    )


def _label(args) -> int:
    pack = _resolve_pack(args.pack)
    assert pack is not None
    out_root = Path(args.out_root)
    failed = 0
    for case in _selected_cases(pack, args.case):
        variant = pack.variant_of(case)
        kernel_pass = next((item for item in variant.profile_passes if item.kind == "nsys"), None)
        if kernel_pass is None:
            print(f"[label] {case.slug}: variant has no nsys pass; skipped")
            continue
        case_dir = out_root / case.slug
        if not case_dir.is_dir():
            print(f"[label] {case.slug}: no run directory under {out_root}")
            failed += 1
            continue
        try:
            report = label_module.label_case(
                pack, variant, case_dir, kernel_pass.name, refresh=args.refresh
            )
        except (PackError, OSError, ValueError) as exc:
            print(f"[label] {case.slug}: {exc}")
            failed += 1
            continue
        print(report.format())
        if not report.ok:
            failed += 1
    return 1 if failed else 0


def _extract(args) -> int:
    pack = _resolve_pack(args.pack)
    roots = [Path(item) for item in args.runs.split(":") if item]
    extraction = extract_module.extract(roots, pack)
    path = extract_module.write_extraction(extraction, Path(args.out))
    print(f"[extract] {len(extraction.cases)} case(s) -> {path}")
    for key in extraction.unavailable:
        issues = extraction.cases[key].issues
        print(f"[extract] {key}: incomplete — {'; '.join(issues[:3])}")
    return 0


def _compare(args) -> int:
    pack = _resolve_pack(args.pack)
    document = extract_module.load_extraction(Path(args.measured))
    comparison = compare_module.compare(document, pack)

    if args.json:
        print(json.dumps(compare_module.as_json(comparison), indent=2, sort_keys=True))
    elif args.markdown:
        print(compare_module.render_markdown(comparison, metric_specs(_acceptance_schemas(pack))))
    else:
        print(compare_module.render_text(comparison), end="")

    if args.record:
        if pack is None:
            print("[record] refused: recording a baseline needs --pack", file=sys.stderr)
            return 2
        try:
            written = compare_module.record(
                document, pack, accept_provisional=args.accept_provisional
            )
        except compare_module.RecordRefused as exc:
            print(f"[record] {exc}", file=sys.stderr)
            return 2
        for path in written:
            print(f"[record] wrote {path}")
        return 0
    return 1 if comparison.failures or comparison.unavailable else 0


# ── parser ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m launcher alignment-campaign",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="verb", required=True)

    check_command = commands.add_parser("check", help="validate a pack (pure CPU)")
    check_command.add_argument("--pack", help="pack directory or name; omit to check every pack")
    check_command.add_argument("--host", help="also validate against this host profile")
    check_command.add_argument(
        "--update-invariants",
        action="store_true",
        help="record a newly added trace in traces/invariants.json; never rewrites an "
        "existing entry, so re-baselining stays a deliberate edit",
    )
    check_command.set_defaults(handler=_check)

    render = commands.add_parser("render", help="write each case's run directory")
    render.add_argument("--pack", required=True)
    render.add_argument("--host", required=True)
    render.add_argument("--out-root", required=True, help="directory to hold one dir per case")
    render.add_argument("--case", action="append", help="restrict to these cases (repeatable)")
    render.set_defaults(handler=_render)

    run = commands.add_parser("run", help="run ONE phase across every ready case")
    run.add_argument("--pack", required=True)
    run.add_argument("--out-root", required=True)
    run.add_argument(
        "--phase",
        required=True,
        help="artifact directory name, e.g. profile_nsys / timing_predict / analysis_kernel",
    )
    run.add_argument("--case", action="append")
    run.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    run.add_argument("--refresh", action="store_true", help="ignore .complete markers")
    run.add_argument(
        "--resume",
        action="store_true",
        help="forwarded to `alignment profile`: reuse the capture, redo extraction only",
    )
    run.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="in-process concurrency budget (default 1: GPU phases contend for one device)",
    )
    run.set_defaults(handler=_run)

    label = commands.add_parser("label", help="apply the rule set to a fixpoint")
    label.add_argument("--pack", required=True)
    label.add_argument("--out-root", required=True)
    label.add_argument("--case", action="append")
    label.add_argument(
        "--refresh", action="store_true", help="re-initialize from the parsed inventory"
    )
    label.set_defaults(handler=_label)

    extract = commands.add_parser("extract", help="completed reports -> metrics document")
    extract.add_argument("--pack", help="optional; without it any run directory is readable")
    extract.add_argument("--runs", required=True, help="colon-separated run roots")
    extract.add_argument("--out", required=True)
    extract.set_defaults(handler=_extract)

    compare = commands.add_parser("compare", help="judge a metrics document")
    compare.add_argument("--pack", help="optional; needed for tolerances, drift and --record")
    compare.add_argument("--measured", required=True, help="metrics document from `extract`")
    compare.add_argument("--json", action="store_true")
    compare.add_argument("--markdown", action="store_true")
    compare.add_argument("--record", action="store_true", help="write the golden baseline")
    compare.add_argument(
        "--accept-provisional",
        action="store_true",
        help="allow --record while calibrated inputs are still provisional",
    )
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.handler(args))
    except PackError as exc:
        print(f"[invalid] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
