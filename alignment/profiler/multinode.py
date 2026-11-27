"""One vLLM server across several nodes: a head and headless peers.

A DP-attention / EP deployment wider than one node (DP16/EP16 on four 4-GPU
GB200 trays) runs one `vllm serve` per node. The head runs DP ranks
``0..L-1`` and the API servers; node ``k`` runs ``--headless
--data-parallel-start-rank k*L``. All of them join the head's DP coordinator
(``--data-parallel-address/--data-parallel-rpc-port``), so the head's HTTP API,
``/start_profile`` and ``/stop_profile`` reach every engine.

The capture runner is one process, on the head. Each other node runs a peer
agent (``python -m alignment.profiler.multinode peer``, started by the job's
launcher on every node but the first) that waits in a shared directory for the
head's word. Everything goes through files there, because the nodes share a
filesystem and nothing else the runner owns:

``launch.json``      head -> peers: argv per node, environment overlay, cwd, logs
``node<k>.clock``    each node: CLOCK_REALTIME/CLOCK_MONOTONIC pairs at start
``node<k>.started``  peer: its server process is up (pid)
``mark``             head: measurement starts now (replay window opens)
``node<k>.mark``     peer: its log's size when it saw ``mark``
``stop``             head: shut down
``node<k>.exit``     peer: its server's exit status, after nsys finalized

Afterwards the head folds the peers back into the single-node shape every
consumer reads (`merge_server_logs`, `merge_nsys_sqlite`): one server log whose
peer records carry the head's CLOCK_MONOTONIC, and one Nsight SQLite whose
devices are numbered ``node*L + local``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PEER_DIR_ENV = "VIBESIM_PEER_DIR"
NODE_RANK_ENV = "VIBESIM_NODE_RANK"
NODE_COUNT_ENV = "VIBESIM_NODE_COUNT"
HEAD_ADDR_ENV = "VIBESIM_HEAD_ADDR"
RPC_PORT = 13345
# torch.distributed rendezvous of a replica spanning nodes (`--master-port`).
MASTER_PORT = 13355

# Fields of the fork's records stated on the emitting host's CLOCK_MONOTONIC.
MONOTONIC_FIELDS = (
    "observed_start_monotonic_ns",
    "observed_end_monotonic_ns",
    "observed_monotonic_ns",
)
_MONOTONIC_RE = re.compile(r'"(' + "|".join(MONOTONIC_FIELDS) + r')":\s*(\d+)')


@dataclass(frozen=True)
class Topology:
    peer_dir: Path
    node_rank: int
    nodes: int
    head_addr: str

    @property
    def is_head(self) -> bool:
        return self.node_rank == 0


def topology() -> Topology | None:
    """This node's place in a multi-node job, from the launcher's environment."""

    count = int(os.environ.get(NODE_COUNT_ENV, "1") or 1)
    if count <= 1:
        return None
    return Topology(
        peer_dir=Path(os.environ[PEER_DIR_ENV]),
        node_rank=int(os.environ[NODE_RANK_ENV]),
        nodes=count,
        head_addr=os.environ[HEAD_ADDR_ENV],
    )


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def clock_pairs(samples: int = 21) -> dict:
    """CLOCK_REALTIME and CLOCK_MONOTONIC read back to back; the tightest pair wins."""

    best = None
    for _ in range(samples):
        r0 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        m = time.monotonic_ns()
        r1 = time.clock_gettime_ns(time.CLOCK_REALTIME)
        if best is None or r1 - r0 < best[2]:
            best = ((r0 + r1) // 2, m, r1 - r0)
    assert best is not None
    return {
        "realtime_ns": best[0],
        "monotonic_ns": best[1],
        "uncertainty_ns": best[2],
        "host": socket.gethostname(),
    }


def write_clock(peer_dir: Path, node_rank: int) -> None:
    _atomic_write(peer_dir / f"node{node_rank}.clock", json.dumps(clock_pairs()) + "\n")


def monotonic_offset_ns(peer_dir: Path, node_rank: int) -> int:
    """Add to node ``node_rank``'s CLOCK_MONOTONIC to read the head's.

    Both nodes' CLOCK_REALTIME are NTP-disciplined, so ``mono - real`` is each
    host's boot epoch on a shared wall clock; the difference of the two is the
    offset between their monotonic clocks.
    """

    head = json.loads((peer_dir / "node0.clock").read_text())
    peer = json.loads((peer_dir / f"node{node_rank}.clock").read_text())
    return (head["monotonic_ns"] - head["realtime_ns"]) - (
        peer["monotonic_ns"] - peer["realtime_ns"]
    )


# -- argv -----------------------------------------------------------------------


def _drop_option(argv: list[str], name: str) -> list[str]:
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == name:
            skip = True
            continue
        if item.startswith(name + "="):
            continue
        out.append(item)
    return out


def head_argv(argv: list[str], topo: Topology, local_size: int) -> list[str]:
    """The head's argv. ``local_size`` is the DP ranks per node; 0 means one
    replica spans every node (TP wider than a node), joined over vLLM's
    multi-node executor instead of the DP coordinator."""

    if local_size == 0:
        return [
            *argv,
            "--nnodes",
            str(topo.nodes),
            "--node-rank",
            "0",
            "--master-addr",
            topo.head_addr,
            "--master-port",
            str(MASTER_PORT),
        ]
    return [
        *argv,
        "--data-parallel-size-local",
        str(local_size),
        "--data-parallel-address",
        topo.head_addr,
        "--data-parallel-rpc-port",
        str(RPC_PORT),
    ]


def peer_argv(head: list[str], node_rank: int, local_size: int) -> list[str]:
    """The head's argv for node ``node_rank``: headless, owning its own DP ranks
    (or, for a replica spanning the nodes, its node rank of that replica)."""

    argv = _drop_option(head, "--api-server-count")
    if local_size == 0:
        return [*_drop_option(argv, "--node-rank"), "--node-rank", str(node_rank), "--headless"]
    return [*argv, "--headless", "--data-parallel-start-rank", str(node_rank * local_size)]


# -- head ------------------------------------------------------------------------


class Peers:
    """The head's handle on the peer agents of one capture."""

    def __init__(self, topo: Topology, run_name: str):
        self.topo = topo
        self.dir = topo.peer_dir / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        for stale in self.dir.iterdir():
            if stale.is_file():
                stale.unlink()
        write_clock(self.dir, 0)
        # Peers look for the newest run directory named here.
        _atomic_write(topo.peer_dir / "current", run_name + "\n")

    def ranks(self) -> range:
        return range(1, self.topo.nodes)

    def launch(self, nodes: dict[int, dict]) -> None:
        """``nodes[k]``: argv, env overlay, unset keys, cwd and log path for node k."""

        _atomic_write(self.dir / "launch.json", json.dumps({str(k): v for k, v in nodes.items()}))

    def wait_started(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            failure = self.failure()
            if failure:
                raise RuntimeError(failure)
            if all((self.dir / f"node{k}.started").exists() for k in self.ranks()):
                return
            time.sleep(1)
        missing = [k for k in self.ranks() if not (self.dir / f"node{k}.started").exists()]
        raise TimeoutError(f"peer node(s) {missing} never started their servers")

    def failure(self) -> str | None:
        for k in self.ranks():
            exit_file = self.dir / f"node{k}.exit"
            if exit_file.exists() and not (self.dir / "stop").exists():
                return f"peer node {k} server exited with {exit_file.read_text().strip()}"
            error = self.dir / f"node{k}.error"
            if error.exists():
                return f"peer node {k}: {error.read_text().strip()}"
        return None

    def mark(self, timeout: float = 30.0) -> dict[int, int]:
        """Open the measurement window; each peer's log offset at that moment."""

        _atomic_write(self.dir / "mark", f"{time.monotonic_ns()}\n")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all((self.dir / f"node{k}.mark").exists() for k in self.ranks()):
                return {k: int((self.dir / f"node{k}.mark").read_text()) for k in self.ranks()}
            time.sleep(0.05)
        raise TimeoutError("a peer node did not acknowledge the measurement mark")

    def request_stop(self) -> None:
        _atomic_write(self.dir / "stop", "\n")

    def wait_stopped(self, timeout: float = 900.0) -> dict[int, str]:
        self.request_stop()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all((self.dir / f"node{k}.exit").exists() for k in self.ranks()):
                break
            time.sleep(1)
        return {
            k: (
                (self.dir / f"node{k}.exit").read_text().strip()
                if (self.dir / f"node{k}.exit").exists()
                else "timeout"
            )
            for k in self.ranks()
        }

    def offsets(self) -> dict[int, int]:
        return {k: monotonic_offset_ns(self.dir, k) for k in self.ranks()}


# What only the node it was read on can say; a peer keeps its own.
_NODE_LOCAL_ENV_PREFIXES = ("SLURM", "SLURMD", "PMI", "PMIX", "OMPI", "ENROOT", "PYXIS")
_NODE_LOCAL_ENV = {"HOSTNAME", "HOST", NODE_RANK_ENV, "TMPDIR", "XDG_RUNTIME_DIR"}


def env_overlay(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """The head's server environment as a peer applies it: set, and unset.

    Everything the head's runner put into its own process environment before
    building ``env`` (the autotune policy, the python runtime's variables) is
    part of it, so the whole environment travels, less what names the node.
    """

    def local(key: str) -> bool:
        return key in _NODE_LOCAL_ENV or key.startswith(_NODE_LOCAL_ENV_PREFIXES)

    overlay = {k: v for k, v in env.items() if not local(k)}
    unset = [k for k in os.environ if k not in env and not local(k)]
    return overlay, unset


# -- merging ---------------------------------------------------------------------


def _shift_line(line: str, offset: int) -> str:
    if offset == 0 or "monotonic_ns" not in line:
        return line
    return _MONOTONIC_RE.sub(lambda m: f'"{m.group(1)}": {int(m.group(2)) + offset}', line)


def _split_log(path: Path, mark: int | None, offset: int) -> tuple[bytes, bytes]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw += b"\n"
    if mark is None:
        parts = [raw, b""]
    else:
        cut = raw.rfind(b"\n", 0, mark) + 1
        parts = [raw[:cut], raw[cut:]]
    if offset:
        parts = [
            "".join(
                _shift_line(line, offset) for line in part.decode(errors="replace").splitlines(True)
            ).encode()
            for part in parts
        ]
    return parts[0], parts[1]


def merge_server_logs(
    head_log: Path,
    peer_logs: dict[int, Path],
    offsets: dict[int, int],
    out: Path,
    *,
    head_mark: int | None,
    peer_marks: dict[int, int] | None,
) -> int | None:
    """One log in the single-node shape; the merged measurement offset.

    Before-mark parts of every node, then after-mark parts, so one byte offset
    still divides startup from measurement. Peer records' monotonic fields are
    moved onto the head's clock.
    """

    befores, afters = [], []
    head_before, head_after = _split_log(head_log, head_mark, 0)
    befores.append(head_before)
    afters.append(head_after)
    for k in sorted(peer_logs):
        mark = None if head_mark is None else (peer_marks or {}).get(k)
        if head_mark is not None and mark is None:
            raise ValueError(f"peer node {k} has no measurement mark")
        before, after = _split_log(peer_logs[k], mark, offsets[k])
        befores.append(before)
        afters.append(after)
    before = b"".join(befores)
    out.write_bytes(before + b"".join(afters))
    return len(before) if head_mark is not None else None


# Nsight packs the OS pid into bits 24.. of globalPid/globalTid; high bits name
# the target. A peer's ids move up by this per node so no two nodes collide,
# and `globalTid & ~0xFFFFFF == globalPid` still holds.
_NODE_ID_SHIFT = 1 << 52


def _columns(con: sqlite3.Connection, table: str, schema: str = "main") -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA {schema}.table_info({table})")]


def _session_start_ns(con: sqlite3.Connection, schema: str = "main") -> int | None:
    try:
        row = con.execute(
            f"SELECT utcEpochNs FROM {schema}.TARGET_INFO_SESSION_START_TIME"
        ).fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else None


MERGED_TABLES = (
    # PROCESSES carries each worker's OS pid, which joins its devices to the
    # rank banners in the server log.
    "PROCESSES",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "NVTX_EVENTS",
    "OSRT_API",
)
_STRING_COLUMNS = ("nameId", "shortName", "demangledName", "mangledName", "textId", "domainId")
_TIME_COLUMNS = ("start", "end", "timestamp")
_ID_COLUMNS = ("globalPid", "globalTid")


def merge_nsys_sqlite(head: Path, peers: dict[int, Path], local_size: int, out: Path) -> dict:
    """Fold peer exports into a copy of the head's; peers' devices become ``k*L+d``.

    Times move onto the head's session timeline by the difference of the
    sessions' UTC start stamps. String ids are re-based past the head's table.
    """

    import shutil

    tmp = out.with_suffix(".sqlite.partial")
    shutil.copyfile(head, tmp)
    con = sqlite3.connect(str(tmp))
    report = {"head": str(head), "peers": {}}
    try:
        head_start = _session_start_ns(con)
        for k, path in sorted(peers.items()):
            con.execute("ATTACH DATABASE ? AS peer", (str(path),))
            peer_start = _session_start_ns(con, "peer")
            shift = (peer_start - head_start) if head_start and peer_start else 0
            (string_base,) = con.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM StringIds"
            ).fetchone()
            con.execute(
                "INSERT INTO StringIds(id, value) SELECT id + ?, value FROM peer.StringIds",
                (string_base,),
            )
            rows = {}
            for table in MERGED_TABLES:
                if table not in {
                    r[0]
                    for r in con.execute("SELECT name FROM peer.sqlite_master WHERE type='table'")
                }:
                    continue
                if not _columns(con, table):
                    con.execute(f"CREATE TABLE main.{table} AS SELECT * FROM peer.{table} WHERE 0")
                head_cols = set(_columns(con, table))
                cols = [c for c in _columns(con, table, "peer") if c in head_cols]
                select = []
                for c in cols:
                    if c in _STRING_COLUMNS:
                        select.append(
                            f"CASE WHEN {c} IS NULL THEN NULL ELSE {c} + {string_base} END"
                        )
                    elif c in _TIME_COLUMNS:
                        select.append(f"CASE WHEN {c} IS NULL THEN NULL ELSE {c} + {shift} END")
                    elif c in _ID_COLUMNS:
                        select.append(f"{c} + {k * _NODE_ID_SHIFT}")
                    elif c == "deviceId":
                        select.append(f"{c} + {k * local_size}")
                    elif c == "id" and table == "NVTX_EVENTS":
                        select.append("NULL")
                    else:
                        select.append(c)
                # correlationId stays: it is unique only per process, and every
                # join on it also keys the (now distinct) globalPid.
                cur = con.execute(
                    f"INSERT INTO main.{table}({', '.join(cols)}) "
                    f"SELECT {', '.join(select)} FROM peer.{table}"
                )
                rows[table] = cur.rowcount
            con.commit()
            con.execute("DETACH DATABASE peer")
            report["peers"][k] = {"sqlite": str(path), "time_shift_ns": shift, "rows": rows}
        con.commit()
    finally:
        con.close()
    os.replace(tmp, out)
    return report


# -- peer agent -------------------------------------------------------------------


def _wait_for(path: Path, *, poll: float = 1.0, stop: Path | None = None) -> bool:
    while not path.exists():
        if stop is not None and stop.exists():
            return False
        time.sleep(poll)
    return True


def peer_main(topo: Topology, *, idle_exit: Path | None = None) -> int:
    """Run node ``topo.node_rank``'s part of each capture the head starts, in turn.

    A job may run several cases (and a warm-up pass) one after another; the
    agent serves each launch the head publishes until the head writes ``done``.
    """

    root = topo.peer_dir
    done = root / "done"
    served: set[str] = set()
    while True:
        current = root / "current"
        while not current.exists() or current.read_text().strip() in served:
            if done.exists():
                return 0
            time.sleep(1)
        run = current.read_text().strip()
        served.add(run)
        rundir = root / run
        if not _wait_for(rundir / "launch.json", stop=done):
            return 0
        _serve_one(topo, rundir)


def _serve_one(topo: Topology, rundir: Path) -> None:
    from alignment.runner import _shutdown  # the head's own stop sequence

    k = topo.node_rank
    write_clock(rundir, k)
    try:
        spec = json.loads((rundir / "launch.json").read_text())[str(k)]
        env = {key: value for key, value in os.environ.items() if key not in spec["unset"]}
        env.update(spec["env"])
        log_path = Path(spec["log"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        proc = subprocess.Popen(
            spec["argv"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=spec["cwd"],
            start_new_session=True,
        )
    except Exception as exc:  # reported to the head, which stops the capture
        _atomic_write(rundir / f"node{k}.error", f"{type(exc).__name__}: {exc}\n")
        return
    _atomic_write(rundir / f"node{k}.started", f"{proc.pid}\n")
    marked = False
    while proc.poll() is None and not (rundir / "stop").exists():
        if not marked and (rundir / "mark").exists():
            log.flush()
            _atomic_write(rundir / f"node{k}.mark", f"{log_path.stat().st_size}\n")
            marked = True
        time.sleep(0.05)
    if proc.poll() is None:
        _shutdown(proc, wrapped=spec.get("wrapped", True))
    log.close()
    _atomic_write(rundir / f"node{k}.exit", f"{proc.returncode}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alignment.profiler.multinode")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("peer", help="serve this node's part of the head's captures")
    args = parser.parse_args(argv)
    topo = topology()
    if topo is None or topo.is_head:
        print("peer: not a peer node of a multi-node job", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.command == "peer":
        return peer_main(topo)
    return 2


if __name__ == "__main__":
    sys.exit(main())
