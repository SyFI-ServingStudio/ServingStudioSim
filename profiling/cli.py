"""Command line facade for existing L1 profiling entries.

This module is intentionally thin: it parses human/agent inputs, then calls the
generated public functions on ``profiling.perf_api``. It must not call runners
or ``run_profile_batch`` directly, preserving the L1 entry-point invariant.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from profiling import perf_api
from profiling.db.args import KernelArgs
from profiling.db.registry import KernelProfilerSpec, iter_kernel_profiler_specs
from profiling.db.table import MissingEntry
from profiling.runners.metrics import CommMetrics, ComputeMetrics, Metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.command_fn(args))
    except Exception as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m profiling",
        description="Run/query existing VibeSim L1 profile entries through profiling.perf_api.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List registered kernel/backend profilers.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    list_parser.set_defaults(command_fn=_cmd_list)

    for command, help_text in (
        ("query", "Read profile rows without JIT or refresh."),
        ("count-missing", "Count missing rows without running profilers."),
        ("run", "Fill missing rows or force-refresh rows through perf_api."),
    ):
        command_parser = subparsers.add_parser(command, help=help_text)
        _add_common_profile_args(command_parser)
        if command == "run":
            command_parser.add_argument(
                "--force",
                action="store_true",
                help="Refresh all specs even when profile.db already has rows.",
            )
        command_parser.set_defaults(
            command_fn={
                "query": _cmd_query,
                "count-missing": _cmd_count_missing,
                "run": _cmd_run,
            }[command]
        )

    measure_parser = subparsers.add_parser(
        "measure",
        help="Sustained per-launch trend + NVML telemetry for one CUPTI kernel spec.",
    )
    _add_common_profile_args(measure_parser)
    measure_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to write CSV / summary.json / plots. Defaults to ./measure_<table>_<backend>.",
    )
    measure_parser.add_argument("--duration-s", type=float, default=10.0)
    measure_parser.add_argument("--telemetry-hz", type=float, default=20.0)
    measure_parser.add_argument(
        "--no-clear-l2",
        dest="clear_l2",
        action="store_false",
        help="Warm continuous window (no per-launch L2 displacement); reveals power/clock drift.",
    )
    measure_parser.set_defaults(command_fn=_cmd_measure, clear_l2=True)
    return parser


def _add_common_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("table", help="Registry table stem, e.g. single_gemm.")
    parser.add_argument("--backend", required=True, help="Registered backend, e.g. torch.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help="One JSON object spec. May be repeated.",
    )
    parser.add_argument(
        "--specs",
        type=Path,
        help="Path to JSON/JSONL specs. JSON may be an object, a list, or {'specs': [...]}.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="profile.db path. Defaults to profiling.perf_api.DB_PATH.",
    )
    parser.add_argument("--gpu-name", help="DB gpu_name key. Defaults to CUDA device 0 name.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _cmd_list(args: argparse.Namespace) -> int:
    rows = [_profiler_spec_summary(profiler_spec) for profiler_spec in iter_kernel_profiler_specs()]
    if args.json:
        print(json.dumps({"ok": True, "profilers": rows}, indent=2, sort_keys=True))
        return 0

    if not rows:
        print("No profiler specs registered.")
        return 0
    _print_table(
        rows,
        columns=[
            "table",
            "kernel_kind",
            "backend",
            "args",
            "metric_family",
            "subprocess_env",
            "get_fn",
            "count_fn",
        ],
    )
    return 0


def _cmd_count_missing(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    count_fn = _count_facade(args.table)
    missing_count = count_fn(specs, backend=args.backend, gpu_name=args.gpu_name)
    payload = _base_payload(args, specs) | {"missing_count": missing_count}
    if args.json:
        print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    else:
        _print_run_summary(payload)
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    results = _get_facade(args.table)(
        specs,
        backend=args.backend,
        gpu_name=args.gpu_name,
        force=False,
    )
    missing_count = _count_facade(args.table)(specs, backend=args.backend, gpu_name=args.gpu_name)
    payload = _result_payload(args, specs, results, mode="query", missing_count=missing_count)
    if args.json:
        print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    else:
        _print_result_payload(payload)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    _set_db_path(args.db)
    mode = "force-refresh" if args.force else "jit-fill"
    get_fn = _get_facade(args.table)

    if args.force:
        results = get_fn(specs, backend=args.backend, gpu_name=args.gpu_name, force=True)
    else:
        perf_api.enable_jit_profiling()
        try:
            results = get_fn(specs, backend=args.backend, gpu_name=args.gpu_name, force=False)
        finally:
            perf_api.disable_jit_profiling()

    missing_count = _count_facade(args.table)(specs, backend=args.backend, gpu_name=args.gpu_name)
    payload = _result_payload(args, specs, results, mode=mode, missing_count=missing_count)
    command_ok = missing_count == 0
    if args.json:
        print(json.dumps({"ok": command_ok, **payload}, indent=2, sort_keys=True))
    else:
        _print_result_payload(payload)
    return 0 if command_ok else 1


def _cmd_measure(args: argparse.Namespace) -> int:
    specs = _load_specs(args.spec, args.specs)
    if len(specs) != 1:
        raise ValueError(f"measure takes exactly one spec, got {len(specs)}")
    output_dir = args.output_dir or Path(f"measure_{args.table}_{args.backend}")
    result = perf_api.measure_kernel(
        args.table,
        specs[0],
        backend=args.backend,
        gpu_name=args.gpu_name,
        output_dir=output_dir,
        duration_s=args.duration_s,
        telemetry_hz=args.telemetry_hz,
        clear_l2=args.clear_l2,
    )
    if args.json:
        print(json.dumps({"ok": True, **result}, indent=2, sort_keys=True))
    else:
        _print_measure_result(result)
    return 0


def _print_measure_result(result: Mapping[str, Any]) -> None:
    print(f"kernel: {result['kernel_kind']}:{result['backend']}")
    print(f"gpu_index: {result['gpu_index']}")
    print(f"output_dir: {result['output_dir']}")
    if result.get("time_ms") is not None:
        print(f"trend_median_ms: {float(result['time_ms']):.6f}")
    if result.get("runner_error"):
        print(f"runner_error: {result['runner_error']}")
    print("artifacts:")
    for artifact in result.get("artifacts", []):
        print(f"  {artifact}")


def _load_specs(spec_values: Sequence[str], specs_path: Path | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec_value in spec_values:
        specs.extend(_coerce_specs_payload(json.loads(spec_value), label="--spec"))
    if specs_path is not None:
        if str(specs_path) == "-":
            text = sys.stdin.read()
        else:
            text = specs_path.read_text(encoding="utf-8")
        specs.extend(_parse_specs_text(text, label=str(specs_path)))
    if not specs:
        raise ValueError("provide at least one --spec JSON object or --specs file")
    return specs


def _parse_specs_text(text: str, *, label: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{label} is empty")
    try:
        return _coerce_specs_payload(json.loads(stripped), label=label)
    except json.JSONDecodeError:
        specs = []
        for line_number, line in enumerate(stripped.splitlines(), start=1):
            if line.strip():
                specs.extend(
                    _coerce_specs_payload(
                        json.loads(line),
                        label=f"{label}:{line_number}",
                    )
                )
        return specs


def _coerce_specs_payload(payload: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and "specs" in payload:
        return _coerce_specs_payload(payload["specs"], label=label)
    if isinstance(payload, Mapping):
        return [dict(payload)]
    if isinstance(payload, list):
        specs = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"{label}[{index}] must be a JSON object")
            specs.append(dict(item))
        return specs
    raise ValueError(f"{label} must be a JSON object, list, or object with a 'specs' list")


def _set_db_path(db_path: Path | None) -> None:
    if db_path is not None:
        perf_api.DB_PATH = db_path


def _get_facade(table_name: str):
    name = f"get_{_validate_table_name(table_name)}_times"
    try:
        return getattr(perf_api, name)
    except AttributeError as exc:
        raise ValueError(f"no generated perf_api facade {name!r}") from exc


def _count_facade(table_name: str):
    name = f"count_missing_{_validate_table_name(table_name)}"
    try:
        return getattr(perf_api, name)
    except AttributeError as exc:
        raise ValueError(f"no generated perf_api facade {name!r}") from exc


def _validate_table_name(table_name: str) -> str:
    known_tables = {profiler_spec.table_name for profiler_spec in iter_kernel_profiler_specs()}
    if table_name in known_tables:
        return table_name
    raise ValueError(
        f"unknown profiler table {table_name!r}; known tables: {sorted(known_tables)}"
    )


def _profiler_spec_summary(profiler_spec: KernelProfilerSpec) -> dict[str, str]:
    args_fields = ",".join(field.name for field in fields(profiler_spec.args_schema))
    stem = profiler_spec.table_name
    return {
        "table": stem,
        "kernel_kind": profiler_spec.kernel_kind,
        "backend": profiler_spec.backend,
        "args": args_fields,
        "metric_family": profiler_spec.metric_family.value,
        "subprocess_env": profiler_spec.subprocess_env or "default_env",
        "get_fn": f"get_{stem}_times",
        "count_fn": f"count_missing_{stem}",
    }


def _base_payload(args: argparse.Namespace, specs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "table": args.table,
        "backend": args.backend,
        "db_path": str(perf_api.DB_PATH),
        "gpu_name": args.gpu_name,
        "spec_count": len(specs),
        "specs": specs,
    }


def _result_payload(
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    results: Sequence[Metrics | MissingEntry],
    *,
    mode: str,
    missing_count: int,
) -> dict[str, Any]:
    return _base_payload(args, specs) | {
        "mode": mode,
        "missing_count": missing_count,
        "results": [_result_to_payload(index, result) for index, result in enumerate(results)],
    }


def _result_to_payload(index: int, result: Metrics | MissingEntry) -> dict[str, Any]:
    if isinstance(result, MissingEntry):
        return {
            "index": index,
            "status": "missing",
            "kernel_kind": result.kernel_kind,
            "backend": result.backend,
            "gpu_name": result.gpu_name,
            "args": _jsonable(result.args),
        }
    if isinstance(result, ComputeMetrics):
        return {
            "index": index,
            "status": "ok",
            "metric_family": "compute",
            "metrics": _jsonable(result),
        }
    if isinstance(result, CommMetrics):
        return {
            "index": index,
            "status": "ok",
            "metric_family": "comm",
            "metrics": _jsonable(result),
        }
    raise TypeError(f"unexpected result type {type(result).__name__}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, KernelArgs):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _print_run_summary(payload: Mapping[str, Any]) -> None:
    print(f"table: {payload['table']}")
    print(f"backend: {payload['backend']}")
    print(f"db_path: {payload['db_path']}")
    if payload.get("gpu_name"):
        print(f"gpu_name: {payload['gpu_name']}")
    print(f"spec_count: {payload['spec_count']}")
    print(f"missing_count: {payload['missing_count']}")


def _print_result_payload(payload: Mapping[str, Any]) -> None:
    _print_run_summary(payload)
    print(f"mode: {payload['mode']}")
    rows = []
    for result in payload["results"]:
        row = {
            "index": str(result["index"]),
            "status": result["status"],
            "args": json.dumps(result.get("args", {}), sort_keys=True),
        }
        metrics = result.get("metrics", {})
        row.update({key: _format_value(value) for key, value in metrics.items()})
        rows.append(row)
    if rows:
        metric_columns = sorted({key for row in rows for key in row} - {"index", "status", "args"})
        _print_table(rows, columns=["index", "status", *metric_columns, "args"])


def _print_table(rows: Sequence[Mapping[str, Any]], *, columns: Sequence[str]) -> None:
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
