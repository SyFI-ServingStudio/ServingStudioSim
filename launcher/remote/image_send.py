"""Sending a profiler image to a remote: only the layers it lacks.

`docker save | docker load` ships every layer of a 30 GB image even when the
remote already holds all but the last few, which is the usual case: a new tag
differs from the previous one in its source, self-check and manifest layers.
Instead the image is pushed to a registry on this machine and pulled through an
ssh reverse forward, so each side moves only the layers the other lacks:

1. A ``registry`` container serves a persistent store on ``127.0.0.1`` here
   (`registry_dir`), for the length of one send. A push uploads only the layers
   the store lacks, so only the first send of an image compresses all of it.
2. The remote pulls ``127.0.0.1:<port>/<tag>`` through ``ssh -R``; docker trusts a
   loopback registry without configuration. The pull fetches only missing layers.
3. The remote retags it to ``<tag>``, and the caller checks the image id: a
   registry round trip keeps the id (the config digest) on the overlay2 store.

The store is this machine's cache: it grows by each new layer and is removed
with `python -m launcher.remote.image_send clear`, never with `docker prune`.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .transport import Transport, q

REGISTRY_IMAGE = "registry:2.8.3"
REGISTRY_DIR_ENV = "VIBESIM_PROFILE_REGISTRY_DIR"
# Remote loopback ports tried for the forward; one taken is retried.
FORWARD_ATTEMPTS = 5


class ImageSendError(RuntimeError):
    """An image could not be sent; the message says where it stopped."""


def registry_dir() -> Path:
    configured = os.environ.get(REGISTRY_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "vibesim-profiler" / "registry"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise ImageSendError(f"docker {' '.join(args)}: {result.stderr.strip()}")
    return result


@contextmanager
def local_registry() -> Iterator[int]:
    """A registry on 127.0.0.1 over the persistent store; yields its port.

    One send at a time uses the store (a lock beside it), and the container
    lives only as long as the send.
    """

    store = registry_dir()
    store.mkdir(parents=True, exist_ok=True)
    with open(store / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        port = _free_port()
        name = f"vibesim-registry-{uuid.uuid4().hex[:8]}"
        _docker(
            "run", "-d", "--rm", "--name", name,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-p", f"127.0.0.1:{port}:5000",
            "-v", f"{store.resolve()}:/var/lib/registry",
            REGISTRY_IMAGE,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v2/", timeout=2):
                        break
                except (urllib.error.URLError, OSError):
                    if time.monotonic() > deadline:
                        logs = _docker("logs", name, check=False)
                        raise ImageSendError(
                            f"the local registry did not start: {logs.stderr.strip()[-500:]}"
                        ) from None
                    time.sleep(0.5)
            yield port
        finally:
            _docker("stop", name, check=False)


def send(transport: Transport, tag: str, *, say=print) -> None:
    """Make ``tag`` present on the remote, moving only the layers it lacks."""

    if transport.remote.ssh is None:
        raise ImageSendError("an access: local remote shares this machine's images")
    with local_registry() as port:
        staged = f"127.0.0.1:{port}/{tag}"
        _docker("tag", tag, staged)
        try:
            say(f"staging {tag} in the local registry (new layers only)")
            pushed = subprocess.run(
                ["docker", "push", "-q", staged], stdout=subprocess.DEVNULL, check=False
            )
            if pushed.returncode != 0:
                raise ImageSendError(f"docker push {staged} failed")
        finally:
            _docker("rmi", staged, check=False)
        for _ in range(FORWARD_ATTEMPTS):
            remote_port = random.randint(20000, 60000)
            pulled = f"127.0.0.1:{remote_port}/{tag}"
            script = (
                f"docker pull -q {q(pulled)} >/dev/null && docker tag {q(pulled)} {q(tag)} "
                f"&& docker rmi {q(pulled)} >/dev/null"
            )
            say(f"pulling {tag} on {transport.remote.name} (missing layers only)")
            result = subprocess.run(
                transport.forward_argv(script, remote_port=remote_port, local_port=port),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return
            if "remote port forwarding failed" not in result.stderr:
                raise ImageSendError(
                    f"pulling {tag} on {transport.remote.name} failed: {result.stderr.strip()}"
                )
        raise ImageSendError(
            f"no free loopback port for the registry forward on {transport.remote.name}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m launcher.remote.image_send")
    parser.add_argument("command", choices=["clear"], help="remove the local layer store")
    parser.parse_args(argv)
    store = registry_dir()
    if store.exists():
        with open(store / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            shutil.rmtree(store)
    print(f"removed {store}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ImageSendError", "REGISTRY_IMAGE", "local_registry", "registry_dir", "send"]
