"""Sending a profiler image to a remote: only the layers it lacks.

`docker save | docker load` ships every layer of a 30 GB image even when the
remote already holds all but the last few, which is the usual case: a new tag
differs from the previous one in its source, self-check and manifest layers.
Instead the image is pushed to a registry on this machine and pulled through an
ssh reverse forward, so each side moves only the layers the other lacks:

1. A ``registry`` container serves a persistent store on ``127.0.0.1`` here
   (`registry_dir`), for the length of one send. `_push` uploads only the layers
   the store lacks. It reads them from ``docker save`` and compresses each with
   multi-threaded zstd, not ``docker push`` (single-threaded gzip, ~19 MB/s for
   a 30 GB image). `LAYER_INDEX` beside the store maps each layer (diff id) to
   its blob, so a layer is compressed once and keeps its digest, which lets the
   remote's pull skip the layers it already holds.
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
import hashlib
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
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
#: diff id -> {"digest", "size", "repository"} of the zstd blob in the store.
LAYER_INDEX = ".zstd-layers.json"
ZSTD_LAYER = "application/vnd.oci.image.layer.v1.tar+zstd"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_CONFIG = "application/vnd.oci.image.config.v1+json"


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
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-p",
            f"127.0.0.1:{port}:5000",
            "-v",
            f"{store.resolve()}:/var/lib/registry",
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


def _request(method: str, url: str, *, data=None, headers: dict | None = None, length=None):
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if length is not None:
        request.add_header("Content-Length", str(length))
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _has_blob(base: str, repository: str, digest: str) -> bool:
    status, _, _ = _request("HEAD", f"{base}/v2/{repository}/blobs/{digest}")
    return status == 200


def _upload_blob(base: str, repository: str, digest: str, path: Path | None, data=None) -> None:
    status, headers, body = _request("POST", f"{base}/v2/{repository}/blobs/uploads/", data=b"")
    if status != 202:
        raise ImageSendError(f"registry refused an upload ({status}): {body[:200]!r}")
    location = headers["Location"]
    if location.startswith("/"):
        location = base + location
    url = f"{location}{'&' if '?' in location else '?'}digest={digest}"
    if path is not None:
        with open(path, "rb") as handle:
            status, _, body = _request(
                "PUT",
                url,
                data=handle,
                length=path.stat().st_size,
                headers={"Content-Type": "application/octet-stream"},
            )
    else:
        status, _, body = _request(
            "PUT", url, data=data, headers={"Content-Type": "application/octet-stream"}
        )
    if status != 201:
        raise ImageSendError(f"registry refused blob {digest} ({status}): {body[:200]!r}")


def _mount_blob(base: str, repository: str, digest: str, source: str) -> bool:
    status, _, _ = _request(
        "POST", f"{base}/v2/{repository}/blobs/uploads/?mount={digest}&from={source}", data=b""
    )
    return status == 201


def _compress(member, scratch: Path) -> tuple[Path, str, int]:
    """zstd (all cores) of one uncompressed layer into ``scratch``; path, digest, size."""

    out = scratch / "layer.tar.zst"
    zstd = subprocess.Popen(
        ["zstd", "-T0", "-3", "-q", "-c"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )

    def feed() -> None:
        try:
            shutil.copyfileobj(member, zstd.stdin, 8 << 20)
        finally:
            zstd.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    digest = hashlib.sha256()
    size = 0
    with open(out, "wb") as sink:
        while chunk := zstd.stdout.read(8 << 20):
            digest.update(chunk)
            sink.write(chunk)
            size += len(chunk)
    feeder.join()
    if zstd.wait() != 0:
        raise ImageSendError("zstd failed while compressing a layer")
    return out, f"sha256:{digest.hexdigest()}", size


def _push(tag: str, port: int, *, say=print) -> str:
    """Put ``tag`` into the local registry (new layers only); its manifest digest.

    Layers the store already holds (by diff id, `LAYER_INDEX`) are linked, not
    re-read; the rest come from one ``docker save`` stream, each compressed
    with zstd on every core and uploaded once.
    """

    base = f"http://127.0.0.1:{port}"
    repository = tag.rpartition(":")[0]
    inspected = json.loads(_docker("image", "inspect", "--format", "{{json .}}", tag).stdout)
    image_id = inspected["Id"]
    diff_ids: list[str] = inspected["RootFS"]["Layers"]
    index_path = registry_dir() / LAYER_INDEX
    index: dict = json.loads(index_path.read_text()) if index_path.is_file() else {}

    missing: set[str] = set()
    for diff_id in dict.fromkeys(diff_ids):
        entry = index.get(diff_id)
        if entry is None:
            missing.add(diff_id)
        elif not _has_blob(base, repository, entry["digest"]) and not _mount_blob(
            base, repository, entry["digest"], entry["repository"]
        ):
            missing.add(diff_id)
    need_config = not _has_blob(base, repository, image_id)
    say(
        f"staging {tag}: {len(missing)} of {len(set(diff_ids))} layer(s) new to the "
        "local registry (zstd)"
    )

    config: bytes | None = None
    if missing or need_config:
        wanted = {d.removeprefix("sha256:") for d in missing}
        config_name = image_id.removeprefix("sha256:")
        save = subprocess.Popen(["docker", "save", tag], stdout=subprocess.PIPE)
        started = time.monotonic()
        moved = 0
        try:
            with (
                tempfile.TemporaryDirectory(dir=registry_dir()) as scratch,
                tarfile.open(fileobj=save.stdout, mode="r|") as archive,
            ):
                for member in archive:
                    name = member.name.rpartition("/")[2]
                    if not member.isfile() or not member.name.startswith("blobs/sha256/"):
                        continue
                    if name == config_name:
                        config = archive.extractfile(member).read()
                    elif name in wanted:
                        path, digest, size = _compress(archive.extractfile(member), Path(scratch))
                        if not _has_blob(base, repository, digest):
                            _upload_blob(base, repository, digest, path)
                        path.unlink()
                        index[f"sha256:{name}"] = {
                            "digest": digest,
                            "size": size,
                            "repository": repository,
                        }
                        index_path.write_text(json.dumps(index, indent=1))
                        wanted.discard(name)
                        moved += member.size
        finally:
            save.stdout.close()
            if save.wait() != 0 and (wanted or config is None):
                raise ImageSendError(f"docker save {tag} failed")
        if wanted:
            raise ImageSendError(f"docker save {tag} lacked layer(s) {sorted(wanted)}")
        elapsed = max(time.monotonic() - started, 1e-6)
        say(f"staged {moved / 1e9:.1f} GB of layers in {elapsed:.0f} s")
    if config is None:
        status, _, config = _request("GET", f"{base}/v2/{repository}/blobs/{image_id}")
        if status != 200:
            raise ImageSendError(f"docker save {tag} had no config blob")
    if need_config:
        _upload_blob(base, repository, image_id, None, data=config)

    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": OCI_MANIFEST,
            "config": {"mediaType": OCI_CONFIG, "digest": image_id, "size": len(config)},
            "layers": [
                {
                    "mediaType": ZSTD_LAYER,
                    "digest": index[diff_id]["digest"],
                    "size": index[diff_id]["size"],
                }
                for diff_id in diff_ids
            ],
        },
        separators=(",", ":"),
    ).encode()
    digest = f"sha256:{hashlib.sha256(manifest).hexdigest()}"
    status, _, body = _request(
        "PUT",
        f"{base}/v2/{repository}/manifests/{digest}",
        data=manifest,
        headers={"Content-Type": OCI_MANIFEST},
    )
    if status != 201:
        raise ImageSendError(f"registry refused the manifest of {tag} ({status}): {body[:300]!r}")
    return digest


def send(transport: Transport, tag: str, *, say=print) -> None:
    """Make ``tag`` present on the remote, moving only the layers it lacks."""

    if transport.remote.ssh is None:
        raise ImageSendError("an access: local remote shares this machine's images")
    with local_registry() as port:
        # Pulled by digest: the forward is open to every user of the remote
        # host, and a digest names exactly what was pushed here.
        digest = _push(tag, port, say=say)
        repository = tag.rpartition(":")[0]
        for _ in range(FORWARD_ATTEMPTS):
            remote_port = random.randint(20000, 60000)
            pulled = f"127.0.0.1:{remote_port}/{repository}@{digest}"
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
