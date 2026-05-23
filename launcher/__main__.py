"""CLI entry — thin dispatch, no sim logic (design §1.2.6).

    python -m launcher path.json [more.json ...] [--override k=v ...]
    python -m launcher list-params [--human]

Positional `.json` paths only; single vs batch is decided by count. Sweep
expansion happens inside `launcher.schema` regardless of how many presets were
passed. No `--web` / `--tui` / `--gui` (UI deleted per discussion.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .schema import (
    _format_log_dir,
    expand_sweep_params,
    normalize_params,
    validate_params,
    validate_unique_log_dirs,
)
from .schema.loader import SchemaNotFound, load_schema


def _build_argparse():
    """The launcher's argparse: positional preset JSON paths + repeatable
    `--override key=value`. Adding a schema param needs no edit here (params come
    from the preset / `--override`, validated against the Rust schema)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m launcher",
        description="MLSim launcher — run / sweep simulations from preset JSON.",
    )
    parser.add_argument(
        "presets",
        nargs="*",
        help="Preset JSON path(s). Multiple presets run as one batch.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a preset param (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate + expand only; do not launch subprocesses.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run every run, ignoring `.complete` markers (default: resume — "
        "skip runs already marked complete).",
    )
    parser.add_argument(
        "--build-type",
        default="release",
        help="Cargo profile / target subdir for the schema + binary (default: release).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap the run with `perf record` to profile simulator wallclock; writes "
        "`<log_dir>/perf.data`. Requires exactly one run (skill profile-sim-speed).",
    )
    parser.add_argument(
        "--profile-freq",
        type=int,
        default=499,
        metavar="HZ",
        help="perf sampling frequency for --profile (default: 499).",
    )
    return parser


def _apply_overrides(preset: dict, overrides: list[str]) -> dict:
    """Apply `--override key=value`. Values are parsed as JSON when possible so
    `tp_size=4`→int, `request_rate=2.5`→float, `fp8=true`/`false`→bool,
    `null`→None, `[1,2]`→list; anything that is not valid JSON (e.g.
    `cp_plan=ring`, `model_config=model/x.json`) stays a bare string."""
    out = dict(preset)
    for item in overrides:
        if "=" not in item:
            sys.exit(f"bad --override {item!r}; expected key=value")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw  # bare string
        out[key.strip()] = value
    return out


def _format_param_row(param_name: str, pdef: dict) -> str:
    tag = "required" if pdef.get("required") else f"default={pdef.get('default')!r}"
    return f"    {param_name:<30} {pdef['type']:<10} {tag:<22} {pdef.get('description', '')}"


def _print_params_table(schema, human: bool) -> None:
    if not human:
        # Raw JSON dump mirrors `simulator list-params`.
        schema_doc = {
            "deployment_schemas": {
                name: list(dep_schema.params.values())
                for name, dep_schema in schema.deployment_schemas.items()
            },
            "pool_fragments": schema.pool_fragments,
        }
        print(json.dumps(schema_doc, indent=2))
        return

    # --human: group each deployment's params under their source pool fragment
    # (declaration order from `schema.pool_fragments`), with deployment-own
    # params last. pool_fragments is display-only (INV-7); the authoritative set
    # is still `dep_schema.params`, so we only display names that appear there.
    for name, dep_schema in schema.deployment_schemas.items():
        print(f"\n== {name} ==")
        grouped: set[str] = set()
        for fragment_name, member_names in schema.pool_fragments.items():
            rows = [pn for pn in member_names if pn in dep_schema.params]
            if not rows:
                continue
            print(f"  [{fragment_name}]")
            for param_name in rows:
                print(_format_param_row(param_name, dep_schema.params[param_name]))
                grouped.add(param_name)
        own = [pn for pn in dep_schema.params if pn not in grouped]
        if own:
            print(f"  [{name}-own]")
            for param_name in own:
                print(_format_param_row(param_name, dep_schema.params[param_name]))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `list-params` subcommand short-circuits before any preset handling.
    if argv and argv[0] == "list-params":
        human = "--human" in argv[1:]
        build_type = "release"
        if "--build-type" in argv:
            build_type = argv[argv.index("--build-type") + 1]
        try:
            schema = load_schema(build_type)
        except SchemaNotFound as exc:
            sys.exit(str(exc))
        _print_params_table(schema, human)
        return 0

    args = _build_argparse().parse_args(argv)
    if not args.presets:
        sys.exit("no preset given; usage: python -m launcher path.json [...]")

    # INV-8 / design §1.2.3: the single shared build per batch. Building also
    # runs `list-params` → deployment_schema.json (§1.2.7), so this is what
    # guarantees the load_schema below reads a schema matching the just-built
    # binary — there is no read-without-build path for a run (including dry-run,
    # which still validates/expands against the schema).
    from .exec import cargo_build

    if not cargo_build(args.build_type):
        sys.exit("build failed; cannot produce deployment schema (see errors above)")

    try:
        schema = load_schema(args.build_type)
    except SchemaNotFound as exc:
        sys.exit(str(exc))

    all_candidates: list[dict] = []
    last_preset: dict = {}
    for preset_path in args.presets:
        preset = json.loads(Path(preset_path).read_text())
        preset = _apply_overrides(preset, args.override)
        last_preset = preset

        errors = validate_params(preset, schema)
        if errors:
            for error in errors:
                print(f"[invalid] {preset_path}: {error}", file=sys.stderr)
            return 2

        candidates = [
            _format_log_dir(normalize_params(candidate, schema))
            for candidate in expand_sweep_params(preset, schema)
        ]
        all_candidates.extend(candidates)

    print(f"[plan] {len(all_candidates)} run(s) across {len(args.presets)} preset(s)")
    if not validate_unique_log_dirs(all_candidates):
        return 2

    # --profile records one representative run; perf on a parallel sweep is
    # meaningless. Enforce a single expanded candidate (skill profile-sim-speed).
    if args.profile and len(all_candidates) != 1:
        sys.exit(
            f"--profile requires exactly one run, but the preset expands to "
            f"{len(all_candidates)}; narrow it (e.g. via --override) to a single run"
        )

    if args.dry_run:
        for candidate in all_candidates:
            print(json.dumps({k: v for k, v in candidate.items() if not k.startswith("_")}))
        return 0

    if args.profile:
        from .exec import perf_available

        if not perf_available():
            sys.exit("--profile needs the `perf` CLI on PATH (install linux-perf / perf)")

    # Defer importing sweep (and its asyncio/subprocess deps) until we launch.
    from .sweep import run_single, run_sweep

    if len(all_candidates) == 1:
        ok = run_single(
            all_candidates[0],
            last_preset,
            schema,
            args.build_type,
            refresh=args.refresh,
            profile=args.profile,
            profile_freq=args.profile_freq,
        )
        return 0 if ok else 1
    return run_sweep(
        all_candidates, last_preset, schema, args.build_type, refresh=args.refresh
    )


if __name__ == "__main__":
    raise SystemExit(main())
