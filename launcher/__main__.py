"""VibeSim launcher CLI dispatch — no simulation logic (L7 design).

Modes:

    python -m launcher PRESET.{yaml,yml,json} [MORE_PRESETS ...] [run options]
    python -m launcher timing-predict CONFIG.{yaml,yml,json} [MORE_CONFIGS ...]
    python -m launcher kernel-profile {list,query,count-missing,run,measure} ...
    python -m launcher alignment {sim,profile,timing-predict,analyze} ...
    python -m launcher list-params [--human] [--build-type PROFILE]

The default mode runs or sweeps deployment simulations. Its run options also
expose validation-only (`--dry-run`), cache coverage (`--cache-report`), backend
enumeration (`--emit-backends`), and simulator wallclock profiling (`--profile`)
paths. YAML and JSON are accepted for simulation presets, timing-predict configs,
and alignment stage configs.

Single versus batch execution is decided by the number of expanded configs.
Sweep expansion belongs to `launcher.schema`. No `--web` / `--tui` / `--gui`
(UI deleted per discussion.md).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from .schema import (
    _format_log_dir,
    expand_sweep_params,
    normalize_params,
    strip_internal,
    validate_distinct_configs,
    validate_expanded,
    validate_params,
    validate_unique_log_dirs,
)
from .schema.loader import (
    PresetError,
    Registry,
    SchemaNotFound,
    _load_preset,
    _merge_backends_file,
    load_schema,
    schema_path,
)


def _build_argparse():
    """The launcher's argparse: positional preset paths + repeatable
    `--override key=value`. Adding a schema param needs no edit here (params come
    from the preset / `--override`, validated against the Rust schema)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m launcher",
        description="VibeSim launcher simulation mode — run or sweep YAML/JSON presets.",
        epilog=(
            "Other modes:\n"
            "  python -m launcher timing-predict CONFIG.yaml|json [...]\n"
            "  python -m launcher kernel-profile {list,query,count-missing,run,measure} ...\n"
            "  python -m launcher alignment {sim,profile,timing-predict,analyze} ...\n"
            "  python -m launcher list-params [--human]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "presets",
        nargs="*",
        help="Simulation preset YAML/JSON path(s). Multiple presets run as one batch.",
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
        "--cache-report",
        action="store_true",
        help="Report profile.db coverage — how many kernel specs are missing "
        "(would be JIT-profiled) per kernel, for each unique cache key — then "
        "exit without building caches or running.",
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
        "`<log_dir>/perf.data`. Requires exactly one run (skill operate-profile-sim-speed).",
    )
    parser.add_argument(
        "--profile-freq",
        type=int,
        default=499,
        metavar="HZ",
        help="perf sampling frequency for --profile (default: 499).",
    )
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip the per-run analyzer (Rust SLO compute + Python plots) that "
        "otherwise runs after each successful run.",
    )
    parser.add_argument(
        "--emit-backends",
        nargs="?",
        const="-",
        default=None,
        metavar="FILE",
        help="Enumerate this preset's distinct kernels (no GPU / no profiling) and "
        "write a per-kernel `backends` skeleton to FILE (or stdout if omitted), "
        "then exit. Edit it + point `backends_file:` at it to tailor backends.",
    )
    return parser


def _set_path(tree: dict, dotted: str, value) -> None:
    """Set `value` at a dotted path into the config tree, e.g.
    `io.log_dir`, `pools.main.groups.0.arch.tp_size`. Integer segments index
    into lists; everything else into dicts (auto-vivifying missing dicts)."""
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _apply_overrides(preset: dict, overrides: list[str]) -> dict:
    """Apply `--override path=value` into the config tree. The path is dotted
    (`io.log_dir`, `pools.main.groups.0.arch.tp_size`); values are parsed as JSON
    when possible (`tp_size=4`→int, `fp8=true`→bool, `[1,2]`→list) and otherwise
    kept as a bare string (`model_config=model/x.json`)."""
    out = copy.deepcopy(preset)
    for item in overrides:
        if "=" not in item:
            sys.exit(f"bad --override {item!r}; expected path=value")
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw  # bare string
        _set_path(out, key.strip(), value)
    return out


def _params_line(pdef: dict) -> str:
    tag = "required" if pdef.get("required") else f"default={pdef.get('default')!r}"
    cache = " [cache-key]" if pdef.get("affects_cache") else ""
    return f"{pdef['name']:<22} {pdef['type']:<10} {tag:<20}{cache}  {pdef.get('description', '')}"


def _print_params_table(registry: Registry, human: bool, build_type: str) -> None:
    if not human:
        # Raw JSON dump mirrors `simulator list-params` (re-read the file the
        # build wrote rather than reconstruct it from the parsed Registry).
        print(schema_path(build_type).read_text().rstrip())
        return

    print("== deployments (role → contract class) ==")
    for dep, body in registry.deployments.items():
        roles = ", ".join(f"{r}→{c}" for r, c in body["pools"].items())
        print(f"  {dep:<10} {roles}")

    for kind, providers in (
        ("arch", registry.arch_providers),
        ("worker", registry.worker_providers),
    ):
        print(f"\n== {kind} providers ==")
        for contract, tags in providers.items():
            print(f"  [{contract}]")
            for tag, body in tags.items():
                params = body.get("params", [])
                names = ", ".join(p["name"] for p in params) or "(no params)"
                print(f"    {tag:<20} {names}")

    for title, params in (
        ("arch_common (carried by every arch tag)", registry.arch_common),
        ("group_common (carried by every group)", registry.group_common),
        ("pool_common (carried by every pool)", registry.pool_common),
        ("common.workload", registry.workload_common),
        ("common.io", registry.io_common),
    ):
        print(f"\n== {title} ==")
        for pdef in params:
            print(f"  {_params_line(pdef)}")


def _emit_backends(args, schema: Registry) -> int:
    """`--emit-backends`: enumerate ONE preset's distinct kernels and write the
    per-kernel `backends` skeleton (to a file, or stdout for `-`). Builds the
    cost-tree structure only — no GPU, no profiling. Strips any existing
    `backends` / `backends_file` (the skeleton shows the arch defaults), then
    resolves one concrete config (role names are sweep-stable, so the first
    combo's roles are the whole set)."""
    from .backends import BackendEnumError, emit_roles, pool_arch_map, render_skeleton

    if len(args.presets) != 1:
        print("[invalid] --emit-backends takes exactly one preset", file=sys.stderr)
        return 2
    source = args.presets[0]
    try:
        preset = _apply_overrides(_load_preset(Path(source)), args.override)
    except PresetError as exc:
        print(f"[invalid] {source}: {exc}", file=sys.stderr)
        return 2
    # The skeleton reports the arch's const-default backends, so drop any override
    # inputs; the pool→arch labels come from the raw preset before that.
    arch_map = pool_arch_map(preset)
    for key in ("backends", "backends_file", "analyze_subjects"):
        preset.pop(key, None)

    errors = validate_params(preset, schema)
    if errors:
        for error in errors:
            print(f"[invalid] {source}: {error}", file=sys.stderr)
        return 2
    candidates = expand_sweep_params(preset, schema)
    if not candidates:
        print(f"[invalid] {source}: preset expands to no runs", file=sys.stderr)
        return 2
    # Enumerate EVERY swept run (not just run 0): a sweep whose runs have different
    # kernel role sets (e.g. tp=1 has no all_reduce) can't share one backends file,
    # so reject at emit; runs that differ only in shape share it (shapes → `(varies)`).
    # `_format_log_dir` templates each `log_dir` so a reject names the run (…/tp1).
    normalized = [_format_log_dir(normalize_params(c, schema)) for c in candidates]
    try:
        roles = emit_roles(normalized, args.build_type)
    except BackendEnumError as exc:
        print(f"[invalid] {source}: emit-backends failed: {exc}", file=sys.stderr)
        return 2

    text = render_skeleton(roles, arch_map)
    dest = args.emit_backends
    if dest == "-":
        sys.stdout.write(text)
    else:
        Path(dest).write_text(text)
        print(f"[emit-backends] wrote {len(roles)} kernel roles to {dest}", file=sys.stderr)
    return 0


def _expand_preset(
    preset: dict,
    schema: Registry,
    source: str,
    *,
    axis: str | None = None,
    label: str | None = None,
) -> list[dict] | None:
    """Validate + expand ONE config preset into normalized, log_dir-templated
    candidates. Returns the candidate list, or `None` if the preset is invalid
    (errors already printed). When `axis`/`label` are given (a `variants` manifest
    branch), tag each run's `_env` with `{axis: label}` so the file becomes a named
    aggregation axis, and prefix its `log_dir` with the label so cross-file runs
    never collide."""
    # Fold `backends_file` + inline `backends` into one nested `backends` block
    # BEFORE validation (so `backends_file` is gone) and before expansion (so its
    # `${var}` values are swept). Un-flattens the file's `pool/role` keys.
    try:
        preset = _merge_backends_file(preset, source)
    except PresetError as exc:
        print(f"[invalid] {source}: {exc}", file=sys.stderr)
        return None
    errors = validate_params(preset, schema)
    if errors:
        for error in errors:
            print(f"[invalid] {source}: {error}", file=sys.stderr)
        return None

    candidates: list[dict] = []
    for candidate in expand_sweep_params(preset, schema):
        # Post-expansion gate: placeholders are now concrete, so the deferred
        # type/choice checks run for real BEFORE normalize coerces.
        post_errors = validate_expanded(candidate, schema)
        if post_errors:
            for error in post_errors:
                print(f"[invalid] {source}: {error}", file=sys.stderr)
            return None
        if axis is not None:
            env = candidate.setdefault("_env", {})
            if axis in env:
                # The manifest's file axis would overwrite an inner sweep/compound/
                # derived binding of the same name, corrupting _env + log_dir.
                print(
                    f"[invalid] {source}: variants axis {axis!r} collides with a "
                    "sweep/compound/derived name in this preset; rename the manifest "
                    "axis so the file axis stays distinct",
                    file=sys.stderr,
                )
                return None
            env[axis] = label
        cand = _format_log_dir(normalize_params(candidate, schema))
        if axis is not None:
            cand["io"]["log_dir"] = f"{label}/{cand['io']['log_dir']}"
        candidates.append(cand)
    return candidates


def _expand_manifest(
    manifest: dict, schema: Registry, source: str, overrides: list[str]
) -> list[dict] | None:
    """Expand a `variants` manifest (no `deployment:`, one named file axis) into
    the union of its referenced presets' candidates. Strictly ONE axis; each label
    maps to a complete standalone preset run with that label tagged on its file
    axis. Returns the merged candidates or `None` if anything is invalid."""
    # Strict parse: a manifest is `variants` plus launcher-only keys; reject extras
    # so a typo (e.g. a stray `sweep:`) is an error, not a silent no-op.
    extra = set(manifest) - {"variants", "analyze_subjects"}
    if extra:
        print(
            f"[invalid] {source}: unknown manifest key(s) {sorted(extra)}; a "
            "`variants` manifest allows only 'variants' (+ launcher-only "
            "'analyze_subjects')",
            file=sys.stderr,
        )
        return None
    variants = manifest["variants"]
    if not isinstance(variants, dict) or len(variants) != 1:
        got = len(variants) if isinstance(variants, dict) else type(variants).__name__
        print(
            f"[invalid] {source}: `variants` must declare exactly one file axis "
            f"(got {got}); split further structure into separate manifests",
            file=sys.stderr,
        )
        return None
    ((axis, branches),) = variants.items()
    if not (isinstance(axis, str) and axis.isidentifier()):
        print(
            f"[invalid] {source}: variants axis name {axis!r} must be a string "
            "identifier (it becomes an aggregation axis / `_env` key)",
            file=sys.stderr,
        )
        return None
    if not isinstance(branches, dict) or not branches:
        print(
            f"[invalid] {source}: variants axis {axis!r} must be a non-empty mapping "
            "of {label: preset_path}",
            file=sys.stderr,
        )
        return None

    merged: list[dict] = []
    for label, ref_path in branches.items():
        if not (isinstance(label, str) and label and "/" not in label and label not in (".", "..")):
            print(
                f"[invalid] {source}: variants.{axis} label {label!r} must be a "
                "non-empty path-safe string (it prefixes each run's log_dir)",
                file=sys.stderr,
            )
            return None
        if not isinstance(ref_path, str):
            print(
                f"[invalid] {source}: variants.{axis}.{label} must be a preset path "
                f"string, got {type(ref_path).__name__}",
                file=sys.stderr,
            )
            return None
        path = Path(ref_path)
        if not path.is_absolute():
            path = Path(source).parent / path  # resolve relative to the manifest
        if not path.is_file():
            print(
                f"[invalid] {source}: variants.{axis}.{label} -> {ref_path!r} not found",
                file=sys.stderr,
            )
            return None
        try:
            ref = _apply_overrides(_load_preset(path), overrides)
        except PresetError as exc:
            print(f"[invalid] {source}: variants.{axis}.{label} -> {exc}", file=sys.stderr)
            return None
        ref.pop("analyze_subjects", None)  # manifest-level only; ref-level ignored
        cands = _expand_preset(ref, schema, ref_path, axis=axis, label=str(label))
        if cands is None:
            return None
        merged.extend(cands)
    return merged


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `alignment` exposes explicit workflow stages. Its `sim` handler re-enters
    # this main function, so dispatch must short-circuit before preset parsing.
    if argv and argv[0] == "alignment":
        from .alignment import main as run_alignment

        return run_alignment(argv[1:])

    # `timing-predict` runs the offline per-building-block timing predictor. Its
    # config is a minimal arch+cases file, NOT a deployment RunConfig, so it has its
    # own handler that bypasses schema expansion / sweeps entirely.
    if argv and argv[0] == "timing-predict":
        from .timing_predict import main as run_timing_predict

        return run_timing_predict(argv[1:])

    # Keep L1 profiling ownership in ``profiling.cli`` while exposing one
    # operator-facing VibeSim command surface alongside timing-predict and sim.
    if argv and argv[0] == "kernel-profile":
        from profiling.cli import main as run_kernel_profile

        return run_kernel_profile(
            argv[1:],
            prog="python -m launcher kernel-profile",
        )

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
        _print_params_table(schema, human, build_type)
        return 0

    args = _build_argparse().parse_args(argv)
    if not args.presets:
        sys.exit(
            "no preset given; usage: python -m launcher PRESET.yaml|json [...] "
            "(or choose timing-predict, kernel-profile, alignment, or list-params)"
        )

    # INV-8 / design §1.2.3: the single shared build per batch. Building also
    # runs `list-params` → deployment_schema.json (§1.2.7), so this is what
    # guarantees the load_schema below reads a schema matching the just-built
    # binary — there is no read-without-build path for a run (including dry-run,
    # which still validates/expands against the schema).
    from .exec import cargo_build

    if not cargo_build(args.build_type, build_analyzer=not args.no_analyze):
        sys.exit("build failed; cannot produce deployment schema (see errors above)")

    try:
        schema = load_schema(args.build_type)
    except SchemaNotFound as exc:
        sys.exit(str(exc))

    if args.emit_backends is not None:
        return _emit_backends(args, schema)

    all_candidates: list[dict] = []
    last_preset: dict = {}
    analyze_subjects: list[str] | None = None
    for preset_path in args.presets:
        try:
            preset = _load_preset(Path(preset_path))
        except PresetError as exc:
            print(f"[invalid] {preset_path}: {exc}", file=sys.stderr)
            return 2
        preset = _apply_overrides(preset, args.override)
        # `analyze_subjects` is a launcher-only key (which post-run analyzer
        # subjects to render). Pop it BEFORE schema validation / sweep expansion
        # so it never reaches the simulator CLI; omit = all applicable subjects.
        subjects = preset.pop("analyze_subjects", None)
        if subjects is not None:
            if not (isinstance(subjects, list) and all(isinstance(s, str) for s in subjects)):
                print(
                    f"[invalid] {preset_path}: analyze_subjects must be a list of "
                    f"strings, got {subjects!r}",
                    file=sys.stderr,
                )
                return 2
            analyze_subjects = subjects
        last_preset = preset

        # A `variants` manifest (no `deployment:`) selects among whole preset files
        # along one named axis; otherwise it is an ordinary config preset.
        if "variants" in preset and "deployment" not in preset:
            cands = _expand_manifest(preset, schema, preset_path, args.override)
        else:
            cands = _expand_preset(preset, schema, preset_path)
        if cands is None:
            return 2
        all_candidates.extend(cands)

    print(f"[plan] {len(all_candidates)} run(s) across {len(args.presets)} preset(s)")
    if not all_candidates:
        sys.exit(
            "no runs after expansion — an empty sweep dim or constraints that reject "
            "every combination"
        )
    if not validate_unique_log_dirs(all_candidates):
        return 2
    if not validate_distinct_configs(all_candidates):
        return 2

    # Backend-map gate (compiler front-end): for every candidate carrying a
    # `backends` block, enumerate its kernels + check the map — unknown role,
    # strict coverage, or a backend incompatible with the role's dtype. No-op when
    # no candidate has backends. Grouped by structure, so a backend-only sweep
    # enumerates once.
    from .backends import BackendEnumError, validate_backends_for_candidates

    try:
        backend_errors = validate_backends_for_candidates(all_candidates, args.build_type)
    except BackendEnumError as exc:
        print(f"[invalid] backend enumeration failed: {exc}", file=sys.stderr)
        return 2
    for err in backend_errors:
        print(f"[invalid] backends: {err}", file=sys.stderr)
    if backend_errors:
        return 2

    # --profile records one representative run; perf on a parallel sweep is
    # meaningless. Enforce a single expanded candidate (skill operate-profile-sim-speed).
    if args.profile and len(all_candidates) != 1:
        sys.exit(
            f"--profile requires exactly one run, but the preset expands to "
            f"{len(all_candidates)}; narrow it (e.g. via --override) to a single run"
        )

    if args.dry_run:
        for candidate in all_candidates:
            print(json.dumps(strip_internal(candidate), default=str))
        return 0

    if args.cache_report:
        from .cache_build import report_cache_coverage

        return report_cache_coverage(all_candidates, schema, args.build_type)

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
            analyze=not args.no_analyze,
            analyze_subjects=analyze_subjects,
        )
        return 0 if ok else 1
    return run_sweep(
        all_candidates,
        last_preset,
        schema,
        args.build_type,
        refresh=args.refresh,
        analyze=not args.no_analyze,
        analyze_subjects=analyze_subjects,
    )


if __name__ == "__main__":
    raise SystemExit(main())
