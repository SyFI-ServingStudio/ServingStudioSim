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

import shutil
import sqlite3
import subprocess
from pathlib import Path

from .config import NsysConfig


def nsys_binary() -> str:
    """Resolve the `nsys` executable (PATH, then the CUDA-12.8 default location)."""
    found = shutil.which("nsys")
    if found:
        return found
    for cand in ("/usr/local/cuda-12.8/bin/nsys", "/usr/local/cuda/bin/nsys"):
        if Path(cand).exists():
            return cand
    raise FileNotFoundError("nsys not found on PATH or in the CUDA bin dir")


def build_nsys_prefix(cfg: NsysConfig, out_rep: Path) -> list[str]:
    """The `nsys profile ...` argv that prefixes the server command."""
    prefix = [
        nsys_binary(),
        "profile",
        # Required by vLLM's own Nsight profiling guide. Without this, a trace
        # can contain EngineCore NVTX and graph-creation metadata yet omit every
        # graph-replay kernel from the spawned CUDA worker.
        "--trace-fork-before-exec=true",
        "--trace=cuda,nvtx",
        f"--sample={cfg.sample}",
        f"--cpuctxsw={cfg.cpuctxsw}",
        f"--cuda-graph-trace={cfg.cuda_graph_trace}",
        "--force-overwrite=true",
        "-o",
        str(out_rep),
    ]
    if cfg.capture_mode == "nvtx":
        prefix += [
            "--capture-range=nvtx",
            "--capture-range-end=none",
            # range@domain; '*' matches any NVTX domain (PyTorch/vLLM vary).
            f"--nvtx-capture={cfg.nvtx_trigger}@*",
        ]
    elif cfg.capture_mode == "cuda_profiler_api":
        prefix += [
            "--capture-range=cudaProfilerApi",
            # One launcher run owns one capture. The target server remains alive
            # after stop so the runner can flush logs and shut it down cleanly.
            "--capture-range-end=stop",
        ]
    return prefix


def find_ray_nsight_trace(since_ts: float, ray_tmp: str = "/tmp/ray") -> Path:
    """Locate the worker `.nsys-rep` Ray's nsight plugin wrote for this run.

    Ray runs each model worker as `nsys profile -o <session>/logs/nsight/
    worker_process_<pid> ... python`, so the trace lands under
    `/tmp/ray/session_*/logs/nsight/`. Pick the newest `worker_process_*.nsys-rep`
    modified after `since_ts` (the server launch time) to avoid a stale session.
    """
    candidates = [
        p
        for p in Path(ray_tmp).glob("session_*/logs/nsight/worker_process_*.nsys-rep")
        if p.stat().st_mtime >= since_ts - 1.0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no Ray nsight worker trace under {ray_tmp}/session_*/logs/nsight/ "
            f"(after ts={since_ts}); did --ray-workers-use-nsight run and finalize?"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def export_sqlite(rep_path: Path) -> Path:
    """`nsys export --type sqlite` → sibling `.sqlite`; returns its path."""
    rep_path = Path(rep_path)
    if not rep_path.exists():
        # nsys may append .nsys-rep to the -o stem.
        alt = rep_path.with_suffix(".nsys-rep")
        if alt.exists():
            rep_path = alt
        else:
            raise FileNotFoundError(f"nsys report not found: {rep_path}")
    sqlite_path = rep_path.with_suffix(".sqlite")
    subprocess.run(
        [nsys_binary(), "export", "--type", "sqlite", "--force-overwrite", "true",
         "-o", str(sqlite_path), str(rep_path)],
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
