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
