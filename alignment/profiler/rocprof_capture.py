"""Wrap an annotated vLLM launch under rocprofv3 and produce a rocpd database.

The AMD counterpart of ``alignment/profiler/nsys_capture.py``. Where nsys runs
``nsys profile [flags] <server argv>`` and exports a ``.sqlite``, rocprofv3 runs
``rocprofv3 [flags] -- <server argv>`` and writes a rocpd SQLite database
directly — so there is no separate export step. The flags that matter for
alignment:

- ``--kernel-trace`` records every GPU kernel dispatch (the rows
  ``profiling.profilers.rocprof_kernel_profiler.kernel_dispatch_records_from_rocpd``
  reads and the producer attributes to iterations).
- ``--marker-trace`` records the roctx ranges the
  :mod:`alignment.profiler.roctx_shim` shim emits — the ``vllm_iteration(N):
  <phase>`` iteration windows the timestamp-containment join owns dispatches by.
- ``--output-format rocpd`` selects the SQLite form the readers understand.

``--kernel-trace`` is the AMD analog of nsys's ``--cuda-graph-trace=node``: it is
mandatory, because without it a trace can carry roctx ranges and still hold no
GPU dispatch rows. The eager (no-HIP-graph) path does not need host HIP-API
correlation, so ``--hip-trace`` is off by default; HIP-graph correlation
ownership is a later chunk.

The executable, output directory, and output name are parameters, never
hardcoded — the 8-GPU orchestrator passes a per-rank output path. The subprocess
is the only GPU-touching step; it is injectable so the capture assembly and the
rocpd validation are unit-testable on-host without rocprofv3 or a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..rocpd import parse as rocpd_parse
from ..rocpd.evidence import iteration_regions
from .config import RocprofConfig
from .roctx_shim import ROCTX_SCOPES_ENV

try:  # the readers live in the profiling package (C0)
    from profiling.profilers.rocprof_kernel_profiler import (
        kernel_dispatch_records_from_rocpd,
        roctx_regions_from_rocpd,
    )
except Exception:  # pragma: no cover - import shape guarded; validate() re-raises
    kernel_dispatch_records_from_rocpd = None  # type: ignore[assignment]
    roctx_regions_from_rocpd = None  # type: ignore[assignment]

#: How a rocpd output is named/placed. rocprofv3 writes ``<dir>/<name>_results.db``
#: for ``--output-format rocpd`` (older builds used ``<name>.db``); the locator
#: accepts either so a version bump does not silently break discovery.
_ROCPD_SUFFIXES = ("_results.db", ".db")

Runner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class ResolvedRocprofExecutable:
    """One verified rocprofv3 binary, with its reported version for provenance."""

    path: Path
    version: str

    def provenance(self) -> dict[str, str]:
        return {"executable": str(self.path), "version": self.version}


def resolve_rocprof_executable(configured_path: str | None) -> ResolvedRocprofExecutable:
    """Resolve and verify the exact rocprofv3 binary from config or ``ROCPROF_BIN``.

    Mirrors ``resolve_nsys_executable``: an absolute path is required (the profiler
    version is run provenance, so PATH guessing is refused), the file must be
    executable, and ``--version`` must return a non-empty string.
    """
    selected_path = configured_path or os.environ.get("ROCPROF_BIN")
    if not selected_path:
        raise ValueError(
            "rocprofv3 executable is not configured; set rocprof.executable or "
            "ROCPROF_BIN to the exact rocprofv3 executable"
        )
    executable_path = Path(selected_path).expanduser()
    if not executable_path.is_absolute():
        raise ValueError(f"rocprofv3 executable must be an absolute path: {selected_path}")
    try:
        executable_path = executable_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"rocprofv3 executable does not exist: {selected_path}") from error
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise PermissionError(f"rocprofv3 executable is not executable: {executable_path}")
    version_result = subprocess.run(
        [str(executable_path), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version = (version_result.stdout or version_result.stderr).strip()
    if not version:
        raise RuntimeError(f"rocprofv3 returned an empty version string: {executable_path}")
    return ResolvedRocprofExecutable(path=executable_path, version=version)


def build_rocprof_prefix(
    executable: ResolvedRocprofExecutable,
    config: RocprofConfig,
    output_dir: Path,
    output_name: str,
) -> list[str]:
    """The ``rocprofv3 ... --`` argv that prefixes the server command.

    The returned list ends with ``--``; the caller appends the server argv.
    """
    config.validate()
    prefix = [str(executable.path)]
    if config.kernel_trace:
        prefix.append("--kernel-trace")
    if config.marker_trace:
        prefix.append("--marker-trace")  # roctx iteration ranges
    if config.hip_trace:
        prefix.append("--hip-trace")
    prefix += [
        "--output-format",
        config.output_format,
        "-d",
        str(output_dir),
        "-o",
        output_name,
    ]
    prefix += list(config.extra_args)
    prefix.append("--")
    return prefix


def build_capture_server_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """The env for the launched server that guarantees the roctx shim fires.

    The AMD analog of what ``vllm_server.build_server_env`` does for NVTX: the
    capture driver — not the operator — turns the iteration annotation on, so a
    capture cannot silently record kernel dispatches with no
    ``vllm_iteration(N)`` roctx ranges to own them.

    vLLM already discovers and calls this package's ``vllm.general_plugins``
    entry point (:func:`roctx_shim.install_vllm_roctx_shim`) in every worker
    process; the plugin, however, no-ops unless :data:`ROCTX_SCOPES_ENV` is
    ``"1"``. Setting it here is the one thing that makes a real capture actually
    bracket each served forward. ``VLLM_PLUGINS`` is intentionally left untouched:
    stock vLLM loads all general plugins by default, so narrowing it would only
    risk disabling unrelated plugins. When a caller does set ``VLLM_PLUGINS`` in
    ``base_env``, it must already include
    :data:`roctx_shim.ROCTX_PLUGIN_ENTRY_POINT_NAME`.
    """
    env = dict(os.environ if base_env is None else base_env)
    env[ROCTX_SCOPES_ENV] = "1"
    return env


def build_capture_argv(
    executable: ResolvedRocprofExecutable,
    config: RocprofConfig,
    server_argv: list[str],
    output_dir: Path,
    output_name: str,
) -> list[str]:
    """The full ``rocprofv3 ... -- <server argv>`` command line."""
    if not server_argv:
        raise ValueError("server_argv must name the command to profile")
    return build_rocprof_prefix(executable, config, output_dir, output_name) + list(server_argv)


def locate_rocpd(output_dir: Path, output_name: str) -> Path:
    """Find the rocpd database rocprofv3 wrote for ``output_name`` under ``output_dir``.

    Accepts both the current ``<name>_results.db`` and the older ``<name>.db``
    spelling; raises when neither exists so a silent capture miss is not mistaken
    for an empty trace.
    """
    for suffix in _ROCPD_SUFFIXES:
        candidate = output_dir / f"{output_name}{suffix}"
        if candidate.exists():
            return candidate
    # rocprofv3 builds have also nested the db under <dir>/<name>/; accept a single
    # *.db there (name-scoped) before giving up.
    nested = sorted(output_dir.glob(f"{output_name}/*.db"))
    if nested:
        return nested[0]
    raise FileNotFoundError(
        f"no rocpd database for output name {output_name!r} under {output_dir} "
        f"(looked for {_ROCPD_SUFFIXES} and nested *.db)"
    )


def validate_rocpd(db_path: Path) -> dict:
    """Sanity-check a fresh rocpd against the iteration-marker + dispatch contract.

    Mirrors ``nsys_capture.validate_export``: catch the two failure modes before a
    confusing empty alignment —

    - no ``(vllm|sglang)_iteration(N)`` roctx ranges → the shim did not run (env
      gate off / ``--marker-trace`` missing / plugin not installed);
    - no kernel dispatch rows → ``--kernel-trace`` never recorded GPU work.

    Uses the real C0 readers, so a passing validation means the same rows the
    offline producer will attribute are present.
    """
    if roctx_regions_from_rocpd is None or kernel_dispatch_records_from_rocpd is None:
        raise RuntimeError(
            "profiling.profilers.rocprof_kernel_profiler is not importable; cannot "
            "read the rocpd capture"
        )
    regions = roctx_regions_from_rocpd(str(db_path))
    iters = iteration_regions(regions)
    dispatches = kernel_dispatch_records_from_rocpd(str(db_path))
    starts = [region.start_ns for _, _, region in iters]
    ends = [region.end_ns for _, _, region in iters]
    span_ms = ((max(ends) - min(starts)) / 1e6) if iters else 0.0
    return {
        "iteration_ranges": len(iters),
        "kernel_rows": len(dispatches),
        "iteration_span_ms": span_ms,
        "ok": bool(iters) and bool(dispatches),
    }


def run_capture(
    executable: ResolvedRocprofExecutable,
    config: RocprofConfig,
    server_argv: list[str],
    output_dir: Path,
    output_name: str,
    *,
    env: dict[str, str] | None = None,
    runner: Runner = subprocess.run,
) -> Path:
    """Run the annotated server under rocprofv3 and return the produced rocpd path.

    ``runner`` defaults to :func:`subprocess.run` and is injectable so a test can
    stand in for the GPU subprocess (writing a synthetic rocpd) without the real
    binary. The output directory is created first because rocprofv3 does not.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_capture_argv(executable, config, server_argv, output_dir, output_name)
    completed = runner(argv, env=env, check=True)  # type: ignore[call-arg]
    if getattr(completed, "returncode", 0) not in (0, None):
        raise RuntimeError(f"rocprofv3 exited with code {completed.returncode}")
    return locate_rocpd(output_dir, output_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alignment rocpd-capture",
        description=(
            "Run an annotated vLLM command under rocprofv3, producing a rocpd "
            "database, then (unless --no-parse) run the offline rocpd->Check-1 "
            "producer on it. The command to profile follows a literal '--'."
        ),
    )
    parser.add_argument(
        "--rocprof",
        dest="rocprof",
        default=None,
        help="absolute path to the rocprofv3 executable (else ROCPROF_BIN)",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="rocpd output directory")
    parser.add_argument(
        "--output-name", required=True, help="rocpd output name (rocprofv3 -o stem)"
    )
    parser.add_argument("--hip-trace", action="store_true", help="also collect the HIP API stream")
    parser.add_argument(
        "--rocprof-arg",
        dest="rocprof_args",
        action="append",
        default=[],
        help="extra rocprofv3 flag, repeatable; appended before '--'",
    )
    parser.add_argument(
        "--parsed-output",
        type=Path,
        default=None,
        help="write parsed.json (+ parsed.kernels.parquet) here after capture",
    )
    parser.add_argument(
        "--sequences-output",
        type=Path,
        default=None,
        help="also write the folded label-ready kernel-sequence catalog here",
    )
    parser.add_argument(
        "--iteration-start", type=int, default=None, help="first iteration index to keep"
    )
    parser.add_argument(
        "--iteration-end", type=int, default=None, help="last iteration index to keep"
    )
    parser.add_argument("--default-stage", default="all", help="stage for ranges lacking one")
    parser.add_argument(
        "--no-parse",
        action="store_true",
        help="only capture + validate the rocpd; skip the parse step",
    )
    return parser


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``<capture flags> -- <server argv>`` on the first literal ``--``."""
    if "--" not in argv:
        raise SystemExit(
            "rocpd-capture requires the command to profile after a literal '--', e.g. "
            "rocpd-capture --output-dir D --output-name N -- python -m vllm ... serve ..."
        )
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1 :]


def main(
    argv: list[str] | None = None,
    *,
    resolver: Callable[[str | None], ResolvedRocprofExecutable] = resolve_rocprof_executable,
    runner: Runner = subprocess.run,
) -> int:
    """`alignment rocpd-capture` entry point.

    ``resolver`` / ``runner`` are injectable so the capture assembly, rocpd
    validation, and parse chain can be exercised on-host with the rocprofv3 binary
    and GPU subprocess stood in for; production uses the real defaults.
    """
    import sys

    flags, server_argv = _split_command(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(flags)
    if not server_argv:
        raise SystemExit("no command to profile after '--'")

    executable = resolver(args.rocprof)
    config = RocprofConfig(
        executable=str(executable.path),
        hip_trace=args.hip_trace,
        extra_args=list(args.rocprof_args),
        analyze_iteration_start=args.iteration_start,
        analyze_iteration_end=args.iteration_end,
    )
    db_path = run_capture(
        executable,
        config,
        server_argv,
        args.output_dir,
        args.output_name,
        env=build_capture_server_env(),
        runner=runner,
    )
    report = validate_rocpd(db_path)
    print(json.dumps({"rocpd": str(db_path), "validation": report}, separators=(",", ":")))
    if not report["ok"]:
        print(
            "rocpd capture incomplete: need both iteration roctx ranges and kernel "
            "dispatch rows (see validation above)",
            file=sys.stderr,
        )
        return 1
    if args.no_parse:
        return 0

    parse_argv = ["--db", str(db_path), "--default-stage", args.default_stage]
    if args.iteration_start is not None:
        parse_argv += ["--iteration-start", str(args.iteration_start)]
    if args.iteration_end is not None:
        parse_argv += ["--iteration-end", str(args.iteration_end)]
    if args.parsed_output is not None:
        parse_argv += ["--output", str(args.parsed_output)]
    if args.sequences_output is not None:
        parse_argv += ["--sequences-output", str(args.sequences_output)]
    return rocpd_parse.main(parse_argv)


if __name__ == "__main__":
    raise SystemExit(main())
