"""`alignment.runner._shutdown` stops a server nsys runs in a process group of its own.

The fake tree mirrors `nsys profile`: a launcher that ignores SIGINT and exits
only once its child has (nsys waits for the application it launched), and a
child in its own process group, the way nsys starts the server.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from alignment import runner

_LAUNCHER = """
import signal, subprocess, sys
signal.signal(signal.SIGINT, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", sys.argv[1]], process_group=0)
open(sys.argv[2], "w").write(str(child.pid))
sys.exit(child.wait())
"""

_SERVER = """
import signal, sys, time
log = {log!r}
def stop(signum, _frame):
    open(log, "a").write(signal.Signals(signum).name + "\\n")
    {on_sigint}
signal.signal(signal.SIGINT, stop)
while True:
    time.sleep(0.05)
"""


def _tree(tmp_path: Path, *, server_exits: bool) -> tuple[subprocess.Popen, int, Path]:
    log, pid_file = tmp_path / "server.log", tmp_path / "server.pid"
    server = _SERVER.format(log=str(log), on_sigint="sys.exit(0)" if server_exits else "pass")
    proc = subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER, server, str(pid_file)], start_new_session=True
    )
    deadline = time.monotonic() + 10
    while not (pid_file.is_file() and pid_file.read_text()):
        assert time.monotonic() < deadline, "fake server never started"
        time.sleep(0.05)
    server_pid = int(pid_file.read_text())
    assert os.getpgid(server_pid) != os.getpgid(proc.pid)  # two groups, as under nsys
    time.sleep(0.3)  # let the server install its handler
    return proc, server_pid, log


def test_the_server_in_its_own_group_is_signalled_and_the_launcher_follows(tmp_path) -> None:
    proc, server_pid, log = _tree(tmp_path, server_exits=True)
    assert runner._descendant_groups(proc.pid) == [os.getpgid(server_pid)]

    started = time.monotonic()
    runner._shutdown(proc, timeout=30)

    assert time.monotonic() - started < 10  # not the timeout
    assert proc.returncode == 0
    assert log.read_text().split() == ["SIGINT"]


def test_a_server_ignoring_sigint_is_killed_with_its_launcher(tmp_path) -> None:
    proc, server_pid, log = _tree(tmp_path, server_exits=False)

    runner._shutdown(proc, timeout=1)

    proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL
    deadline = time.monotonic() + 10
    while True:
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            break
        # A killed child of a dead launcher is reparented and reaped; until
        # then it is a zombie, which counts as gone. It may also be reaped
        # between the two looks.
        try:
            stat = Path(f"/proc/{server_pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            break
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            break
        assert time.monotonic() < deadline, "server survived the shutdown"
        time.sleep(0.05)
    assert "SIGINT" in log.read_text()


def test_a_server_sharing_the_launchers_group_gets_one_sigint(tmp_path) -> None:
    log = tmp_path / "server.log"
    server = _SERVER.format(log=str(log), on_sigint="sys.exit(0)")
    proc = subprocess.Popen([sys.executable, "-c", server], start_new_session=True)
    time.sleep(0.5)
    assert runner._descendant_groups(proc.pid) == []

    runner._shutdown(proc, timeout=30)

    assert proc.returncode == 0
    assert log.read_text().split() == ["SIGINT"]


_HELPER_PARENT = """
import signal, subprocess, sys, time
log = sys.argv[2]
# A helper in a group of its own that ignores SIGINT (vLLM's dp supervisor
# calls setpgrp; DeepGEMM starts a session).
subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)",
    ],
    process_group=0,
)
def stop(signum, _frame):
    open(log, "a").write("SIGINT\\n")
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
open(sys.argv[1], "w").write("ready")
while True:
    time.sleep(0.05)
"""


def test_an_unwrapped_server_is_signalled_at_once_despite_a_stubborn_helper(tmp_path) -> None:
    ready, log = tmp_path / "ready", tmp_path / "server.log"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HELPER_PARENT, str(ready), str(log)], start_new_session=True
    )
    deadline = time.monotonic() + 10
    while not ready.is_file():
        assert time.monotonic() < deadline, "fake server never started"
        time.sleep(0.05)
    time.sleep(0.3)
    helper_groups = runner._descendant_groups(proc.pid)
    assert helper_groups

    started = time.monotonic()
    runner._shutdown(proc, timeout=30, wrapped=False)

    assert time.monotonic() - started < 10  # not the helper-wait timeout
    assert proc.returncode == 0
    assert log.read_text().split() == ["SIGINT"]
    # The helper ignored the SIGINT; it does not outlive the server.
    deadline = time.monotonic() + 10
    while runner._groups_alive(helper_groups):
        assert time.monotonic() < deadline, "the helper outlived the server"
        time.sleep(0.05)


# A server that leaves a child behind: `child_group` 0 puts the child in a
# group of its own, None keeps it in the server's. The child ignores SIGINT.
_LEAVES_A_CHILD = """
import signal, subprocess, sys, time
kwargs = {{"process_group": 0}} if {own_group} else {{}}
child = subprocess.Popen(
    [sys.executable, "-c",
     "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"],
    **kwargs,
)
open(sys.argv[1], "w").write(str(child.pid))
if {exit_at_once}:
    sys.exit(3)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.05)
"""


def _leaves_a_child(tmp_path: Path, *, own_group: bool, exit_at_once: bool):
    child_file = tmp_path / "child.pid"
    program = _LEAVES_A_CHILD.format(own_group=own_group, exit_at_once=exit_at_once)
    proc = subprocess.Popen(
        [sys.executable, "-c", program, str(child_file)], start_new_session=True
    )
    deadline = time.monotonic() + 10
    while not (child_file.is_file() and child_file.read_text()):
        assert time.monotonic() < deadline, "fake server never started"
        time.sleep(0.05)
    time.sleep(0.3)
    return proc, int(child_file.read_text())


def _gone(pid: int) -> bool:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return True
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.05)
    return False


def test_a_worker_left_in_the_servers_own_group_is_killed(tmp_path) -> None:
    proc, child = _leaves_a_child(tmp_path, own_group=False, exit_at_once=False)
    runner._shutdown(proc, timeout=30, wrapped=False)
    assert proc.returncode == 0
    assert _gone(child), "the server's worker outlived it"


def test_a_crashed_servers_leftovers_are_killed(tmp_path) -> None:
    proc, child = _leaves_a_child(tmp_path, own_group=True, exit_at_once=True)
    proc.wait(timeout=10)
    runner._shutdown(proc, timeout=30, wrapped=False)  # already exited
    assert _gone(child), "a crashed server's helper outlived it"


def test_a_wrapped_launchers_leftover_helper_is_killed(tmp_path) -> None:
    proc, child = _leaves_a_child(tmp_path, own_group=True, exit_at_once=False)
    runner._shutdown(proc, timeout=2)  # the helper ignores its SIGINT: waits 2 s
    assert proc.returncode == 0
    assert _gone(child), "the helper outlived the launcher"


def test_a_hung_servers_leftovers_are_swept_after_the_kill(tmp_path) -> None:
    # The server ignores SIGINT: the timeout SIGKILLs it, then its worker
    # (same session, a group of its own, also ignoring SIGINT) must go too.
    proc, child = _leaves_a_child(tmp_path, own_group=True, exit_at_once=False)
    os.kill(proc.pid, signal.SIGSTOP)  # stands in for a hung server
    runner._shutdown(proc, timeout=1, wrapped=False)
    assert proc.returncode == -signal.SIGKILL
    assert _gone(child), "the hung server's worker outlived the kill"


def test_a_reused_leader_pid_is_not_swept(monkeypatch) -> None:
    # The leader is reaped; a live process now holds its pid (a stranger),
    # so its session is not ours.
    killed = []
    table = {500: (1, 500, 500, 9), 600: (500, 600, 500, 9)}
    monkeypatch.setattr(runner, "_processes", lambda: table)
    monkeypatch.setattr(runner, "_signal_groups", lambda groups, _sig: killed.extend(groups))
    runner._kill_leftovers(500, {})
    assert killed == []
    runner._kill_leftovers(400, {})  # a leader whose pid nobody holds: sweep
    assert killed == []  # no process is in session 400
    monkeypatch.setattr(runner, "_processes", lambda: {600: (1, 600, 500, 9)})
    runner._kill_leftovers(500, {})
    assert killed == [600]


def test_an_unreaped_leader_still_has_its_session_swept(monkeypatch) -> None:
    # SIGKILLed but not reaped in time (stuck in the kernel): its pid is
    # still its own, so its session is swept.
    killed = []
    table = {500: (1, 500, 500, 5), 600: (500, 600, 500, 6)}
    monkeypatch.setattr(runner, "_processes", lambda: table)
    monkeypatch.setattr(runner, "_signal_groups", lambda groups, _sig: killed.extend(groups))
    runner._kill_leftovers(500, {}, reaped=False)
    assert killed == [500, 600]


def test_a_helper_group_is_matched_by_a_recorded_member(monkeypatch) -> None:
    killed = []
    monkeypatch.setattr(runner, "_signal_groups", lambda groups, _sig: killed.extend(groups))
    helpers = {700: (700, frozenset({(700, 10), (701, 11)}))}
    # A recorded member is alive (its later fork 702 goes with the group).
    table = {701: (1, 700, 700, 11), 702: (701, 700, 700, 30)}
    monkeypatch.setattr(runner, "_processes", lambda: table)
    runner._kill_leftovers(400, helpers)
    assert killed == [700]
    killed.clear()
    # All members gone; pid 700 went to a stranger that started a session.
    monkeypatch.setattr(runner, "_processes", lambda: {700: (1, 700, 700, 40)})
    runner._kill_leftovers(400, helpers)
    assert killed == []
