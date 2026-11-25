"""The multi-node fold: peer logs onto the head's clock, peer traces onto its devices."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from alignment.profiler import multinode


def test_peer_argv_is_headless_and_owns_its_ranks():
    topo = multinode.Topology(Path("/x"), 0, 4, "head-node")
    head = multinode.head_argv(
        ["vllm", "serve", "m", "--api-server-count", "4", "--data-parallel-size", "16"], topo, 4
    )
    assert head[-6:] == [
        "--data-parallel-size-local",
        "4",
        "--data-parallel-address",
        "head-node",
        "--data-parallel-rpc-port",
        str(multinode.RPC_PORT),
    ]
    peer = multinode.peer_argv(head, 2, 4)
    assert "--api-server-count" not in peer and "4" in peer
    assert peer[-3:] == ["--headless", "--data-parallel-start-rank", "8"]


def test_a_replica_spanning_nodes_joins_by_node_rank():
    topo = multinode.Topology(Path("/x"), 0, 4, "head-node")
    head = multinode.head_argv(
        ["vllm", "serve", "m", "--api-server-count", "2", "--tensor-parallel-size", "16"], topo, 0
    )
    assert head[-8:] == [
        "--nnodes", "4", "--node-rank", "0", "--master-addr", "head-node",
        "--master-port", str(multinode.MASTER_PORT),
    ]
    peer = multinode.peer_argv(head, 3, 0)
    assert "--api-server-count" not in peer and "--data-parallel-start-rank" not in peer
    assert peer.count("--node-rank") == 1 and peer[-3:] == ["--node-rank", "3", "--headless"]


def test_merge_server_logs_splits_at_marks_and_shifts_peer_clocks(tmp_path):
    head = tmp_path / "head.log"
    head.write_text("h-start\nh-measure\n")
    peer = tmp_path / "peer.log"
    record = {"observed_start_monotonic_ns": 100, "observed_end_monotonic_ns": 150, "x": 1}
    peer.write_text("p-start\nALIGNMENT " + json.dumps(record) + "\n")
    out = tmp_path / "merged.log"
    offset = multinode.merge_server_logs(
        head,
        {1: peer},
        {1: 1000},
        out,
        head_mark=len("h-start\n"),
        peer_marks={1: len("p-start\n")},
    )
    text = out.read_text()
    assert text[:offset] == "h-start\np-start\n"
    after = text[offset:].splitlines()
    assert after[0] == "h-measure"
    shifted = json.loads(after[1].removeprefix("ALIGNMENT "))
    assert shifted["observed_start_monotonic_ns"] == 1100
    assert shifted["observed_end_monotonic_ns"] == 1150
    assert shifted["x"] == 1


def test_merge_server_logs_without_a_mark_keeps_everything_before(tmp_path):
    head = tmp_path / "head.log"
    head.write_text("a\n")
    peer = tmp_path / "peer.log"
    peer.write_text("b")
    out = tmp_path / "merged.log"
    assert (
        multinode.merge_server_logs(head, {1: peer}, {1: 0}, out, head_mark=None, peer_marks=None)
        is None
    )
    assert out.read_text() == "a\nb\n"


def _trace(path: Path, start_utc: int, pid: int, names: list[str]) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER,
            globalPid INTEGER, correlationId INTEGER, shortName INTEGER, demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER);
        CREATE TABLE NVTX_EVENTS(id INTEGER PRIMARY KEY, start INTEGER, end INTEGER,
            globalTid INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE PROCESSES(globalPid INTEGER, pid INTEGER, name TEXT);
        """
    )
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names, 1)))
    con.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)", (start_utc,))
    gpid = pid << 24
    con.execute("INSERT INTO PROCESSES VALUES (?, ?, 'VLLM::Worker_DP')", (gpid, pid))
    for device in range(2):
        con.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, 1, 2)",
            (10, 20, device, gpid + device, 7),
        )
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (5, 9, ?, 7, 1)", (gpid + pid,))
    con.execute(
        "INSERT INTO NVTX_EVENTS(start, end, globalTid, text) VALUES (1, 30, ?, 'it')",
        (gpid + pid,),
    )
    con.commit()
    con.close()


def test_merge_nsys_sqlite_renumbers_devices_and_moves_times(tmp_path):
    head, peer, out = tmp_path / "h.sqlite", tmp_path / "p.sqlite", tmp_path / "m.sqlite"
    _trace(head, 1_000, 100, ["kernA", "kernA_full"])
    _trace(peer, 1_500, 100, ["kernB", "kernB_full"])
    report = multinode.merge_nsys_sqlite(head, {1: peer}, 2, out)
    assert report["peers"][1]["time_shift_ns"] == 500
    con = sqlite3.connect(out)
    rows = con.execute(
        "SELECT k.deviceId, k.start, s.value, k.globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.deviceId"
    ).fetchall()
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    assert [r[2] for r in rows] == ["kernA", "kernA", "kernB", "kernB"]
    assert rows[2][1] == 510
    # Same OS pid on both nodes, distinct Nsight ids, low 24 bits untouched.
    assert rows[0][3] != rows[2][3] and (rows[2][3] >> 24) & 0xFFFFFF == 100
    (tids,) = con.execute(
        "SELECT COUNT(DISTINCT globalTid) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    assert tids == 2
    assert con.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone() == (2,)
    # Each node's worker keeps its OS pid under its node's Nsight id, so the
    # parser can join a peer's devices to that worker's rank banner.
    procs = con.execute("SELECT globalPid, pid FROM PROCESSES ORDER BY globalPid").fetchall()
    assert [p for _, p in procs] == [100, 100]
    assert {g >> 24 for g, _ in procs} == {r[3] >> 24 for r in rows}
    con.close()


def test_monotonic_offset_from_clock_pairs(tmp_path):
    (tmp_path / "node0.clock").write_text(json.dumps({"realtime_ns": 10_000, "monotonic_ns": 300}))
    (tmp_path / "node1.clock").write_text(json.dumps({"realtime_ns": 10_002, "monotonic_ns": 52}))
    # head boot epoch = 300-10000 = -9700; peer = 52-10002 = -9950; peer + 250 = head.
    assert multinode.monotonic_offset_ns(tmp_path, 1) == 250


def test_a_pyxis_job_keeps_its_cache_on_every_nodes_local_disk():
    from pathlib import PurePosixPath

    from launcher.remote.launch import Container, pyxis_script

    container = Container(
        image="vibesim-profiler-vllm:abc",
        python=PurePosixPath("/opt/vibesim-capture/.venv/bin/python"),
        workdir=PurePosixPath("/shared/work"),
        node_local=PurePosixPath("/tmp"),
        cache_label="vllm-0123",
        runtime="pyxis",
        squashfs="/shared/images/x.sqsh",
        nodes=4,
    )
    script = pyxis_script(
        container,
        job_id="j",
        jobdir=PurePosixPath("/shared/work/jobs/j"),
        cwd=PurePosixPath("/shared/work/src/v"),
    )
    # Not under the shared workdir, where 16 ranks' JIT writes race on NFS.
    assert "cache=/tmp/vibesim-cache/vllm-0123" in script
    # Made on all four nodes before the containers mount it.
    mkdir = script.index('srun --nodes=4 --ntasks-per-node=1 mkdir -p "$cache/home"')
    assert mkdir < script.index("--container-image=")
    assert "--container-writable" in script


def test_a_held_allocation_takes_queued_jobs_instead_of_sbatch(tmp_path):
    from pathlib import PurePosixPath

    from launcher.remote import launch
    from launcher.remote.config import Remote, Slurm

    class Transport:
        def __init__(self):
            self.files, self.commands = {}, []

        def write_text(self, path, text):
            self.files[str(path)] = text

        def run(self, command, check=True):
            self.commands.append(command)

            class Done:
                stdout = "RUNNING\n" if command.startswith("squeue") else ""

            return Done()

    remote = Remote(
        name="cluster",
        workdir=PurePosixPath("/w"),
        ssh=None,
        slurm=Slurm(time="04:00:00", nodes=4, allocation="4242"),
        files="shared",
    )
    transport = Transport()
    jobdir = PurePosixPath("/w/jobs/j1")
    submitted = launch._enqueue(
        transport,
        remote,
        job_id="j1",
        jobdir=jobdir,
        cwd=PurePosixPath("/w/src/v"),
        command="bash x",
    )
    assert submitted.handle == "4242" and submitted.queued == PurePosixPath(
        "/w/allocations/4242/j1"
    )
    assert not any(c.startswith("sbatch") for c in transport.commands)
    line = transport.files["/w/allocations/4242/j1.tmp"]
    assert "cd /w/src/v && exec bash x > /w/jobs/j1/slurm.out 2>&1" in line
    assert transport.commands[-1] == "mv /w/allocations/4242/j1.tmp /w/allocations/4242/j1.job"
    status = launch.status_script(submitted, 0)
    assert "squeue -h -j 4242" in status and "/w/allocations/4242/j1.running" in status
    launch.cancel(transport, submitted)
    assert transport.commands[-1].endswith("touch /w/allocations/4242/j1.cancel")
