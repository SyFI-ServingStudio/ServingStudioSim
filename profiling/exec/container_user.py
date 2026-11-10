"""The ``--user`` a profiler container runs as, for rootful and rootless runtimes.

Rootless Docker and Podman (cayenne's Slurm jobs run a job-scoped rootless
Podman) map container root to the invoking user; any other container uid lands
on a subordinate host uid, which cannot read the caller's 0700 exchange
directory or write files the caller owns. A rootful daemon is the opposite:
container root is host root, so the container runs as the caller's uid.
"""

from __future__ import annotations

import functools
import os
import subprocess


@functools.cache
def rootless_runtime() -> bool:
    """Whether ``docker info`` reports a rootless runtime (Docker or Podman).

    Cached per process: the runtime does not change under a running profiler.
    A ``docker info`` that cannot run reports rootful; the ``docker run`` that
    follows fails on its own with the real error.
    """
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{json .SecurityOptions}}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return False
    return "rootless" in out


def container_user(rootless: bool | None = None) -> str:
    """``uid:gid`` for ``docker run --user``: ``0:0`` under a rootless runtime."""
    if rootless is None:
        rootless = rootless_runtime()
    return "0:0" if rootless else f"{os.getuid()}:{os.getgid()}"
