"""Wrap a vLLM launch under Nsight Systems and export the trace to SQLite.

`nsys profile [flags] <server argv>` runs the server *inside* the profiler; the
NVTX iteration scopes emitted by the fork mark iteration/phase boundaries that
`alignment/nsys/parse.py` attributes kernels to. Three capture modes:

- **cuda_profiler_api** (default for the OpenAI server): the launcher calls
  vLLM's `/start_profile` and `/stop_profile` endpoints, so the CUDA-owning
  worker arms and stops CUPTI itself.
- **nvtx**: start capturing when the trigger
  NVTX range `vllm_iteration(<trigger>): forward` is pushed, and keep going to
  process end (`--capture-range-end=none`). Skips the noisy model-load prologue.
- **full**: capture the whole process. Simplest for a tiny controlled workload;
  the parser's iteration window filters out warmup anyway.

`--cuda-graph-trace=node` is mandatory: without it, decode/graph-replay
iterations have NVTX ranges but no overlapping CUDA kernel rows in the export.
`--trace-fork-before-exec=true` is equally important for the OpenAI server:
vLLM starts the CUDA-owning EngineCore in a child interpreter, so Nsight must
instrument that child before its `exec` rather than only tracing the API server.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import NsysConfig


@dataclass(frozen=True)
class ResolvedNsysExecutable:
    """One verified profiler binary shared by capture and export."""

    path: Path
    version: str

    def provenance(self) -> dict[str, str]:
        return {"executable": str(self.path), "version": self.version}


def resolve_nsys_executable(configured_path: str | None) -> ResolvedNsysExecutable:
    """Resolve and verify the exact profiler selected by config or ``NSYS_BIN``."""
    selected_path = configured_path or os.environ.get("NSYS_BIN")
    if not selected_path:
        raise ValueError(
            "NSYS executable is not configured; set nsys.executable or NSYS_BIN "
            "to the exact Nsight Systems executable"
        )
    executable_path = Path(selected_path).expanduser()
    if not executable_path.is_absolute():
        raise ValueError(f"NSYS executable must be an absolute path: {selected_path}")
    try:
        executable_path = executable_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"NSYS executable does not exist: {selected_path}") from error
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise PermissionError(f"NSYS executable is not executable: {executable_path}")
    version_result = subprocess.run(
        [str(executable_path), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    version = (version_result.stdout or version_result.stderr).strip()
    if not version:
        raise RuntimeError(f"NSYS returned an empty version string: {executable_path}")
    return ResolvedNsysExecutable(path=executable_path, version=version)


def build_nsys_prefix(
    executable: ResolvedNsysExecutable,
    config: NsysConfig,
    report_stem: Path,
) -> list[str]:
    """The `nsys profile ...` argv that prefixes the server command."""
    prefix = [
        str(executable.path),
        "profile",
        # Required by vLLM's own Nsight profiling guide. Without this, a trace
        # can contain EngineCore NVTX and graph-creation metadata yet omit every
        # graph-replay kernel from the spawned CUDA worker.
        "--trace-fork-before-exec=true",
        "--trace=cuda,nvtx",
        f"--sample={config.sample}",
        f"--cpuctxsw={config.cpuctxsw}",
        f"--cuda-graph-trace={config.cuda_graph_trace}",
        f"--cuda-event-trace={str(config.cuda_event_trace).lower()}",
        "--force-overwrite=true",
        "-o",
        str(report_stem),
    ]
    if config.capture_mode == "nvtx":
        prefix += [
            "--capture-range=nvtx",
            "--capture-range-end=none",
            # range@domain; '*' matches any NVTX domain (PyTorch/vLLM vary).
            f"--nvtx-capture={config.nvtx_trigger}@*",
        ]
    elif config.capture_mode == "cuda_profiler_api":
        prefix += [
            "--capture-range=cudaProfilerApi",
            # One launcher run owns one capture. The target server remains alive
            # after stop so the runner can flush logs and shut it down cleanly.
            "--capture-range-end=stop",
        ]
    return prefix


def export_sqlite(executable: ResolvedNsysExecutable, report_path: Path) -> Path:
    """`nsys export --type sqlite` → sibling `.sqlite`; returns its path."""
    report_path = Path(report_path)
    if not report_path.exists():
        # nsys may append .nsys-rep to the -o stem.
        alternate_report_path = report_path.with_suffix(".nsys-rep")
        if alternate_report_path.exists():
            report_path = alternate_report_path
        else:
            raise FileNotFoundError(f"nsys report not found: {report_path}")
    sqlite_path = report_path.with_suffix(".sqlite")
    subprocess.run(
        [
            str(executable.path),
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "-o",
            str(sqlite_path),
            str(report_path),
        ],
        check=True,
    )
    return sqlite_path


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def validate_export(sqlite_path: Path) -> dict:
    """Sanity-check a fresh export against the indexed-marker contract.

    Two failure modes this catches before a confusing empty gap table:
    - No forward NVTX ranges → the scopes weren't emitted (missing
      `VLLM_NVTX_SCOPES_FOR_PROFILING`) or the capture window missed them.
    - No `CUPTI_ACTIVITY_KIND_KERNEL` table/rows → GPU tracing never recorded
      kernels, usually because the targeted capture never armed or graph-node
      tracing was omitted.

    Only indexed `vllm_iteration(N)` / `sglang_iteration(N)` ranges are valid.
    Nsight may intern their text into `StringIds`, so validation resolves both
    SQLite storage representations without accepting unindexed aliases.
    """
    con = sqlite3.connect(str(sqlite_path))
    try:
        (n_forward,) = con.execute(
            """
            SELECT COUNT(*) FROM NVTX_EVENTS n
            LEFT JOIN StringIds s ON n.textId = s.id
            WHERE COALESCE(n.text, s.value) LIKE 'vllm_iteration(%): forward'
               OR COALESCE(n.text, s.value) LIKE 'sglang_iteration(%): forward'
            """
        ).fetchone()
        if _table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
            kmin, kmax, krows = con.execute(
                "SELECT MIN(start), MAX(end), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()
        else:
            kmin, kmax, krows = None, None, 0
    finally:
        con.close()
    return {
        "nvtx_forward_ranges": int(n_forward or 0),
        "kernel_rows": int(krows or 0),
        "kernel_span_ms": ((kmax - kmin) / 1e6) if (kmin and kmax) else 0.0,
        "ok": bool(n_forward) and bool(krows),
    }
