"""Running commands and moving files on a remote's access host.

`access: local` runs on this machine; `access: {ssh: ...}` runs over one shared
ssh connection (ControlMaster), so the many short commands of a remote run pay
for one handshake. ssh never prompts (BatchMode): an agent turn has no terminal
and no ssh-agent, so a remote must be reachable with a key file alone. When
``.remote/known_hosts`` sits beside remote.yaml it is the only host-key store
consulted, which keeps a sandbox that cannot write ``~/.ssh`` working.

Remote commands are bash scripts run through ``bash -lc``: a login shell picks
up the remote user's PATH (uv, cargo, sbatch live there on most hosts).
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import Remote


class TransportError(RuntimeError):
    """A remote command or copy failed; the message carries its output."""


# A unix socket path holds 107 bytes, and ssh appends a 17-byte random suffix
# while it creates the master's socket.
CONTROL_PATH_LIMIT = 107 - 17


@dataclass(frozen=True)
class Transport:
    remote: Remote

    @property
    def is_local(self) -> bool:
        return self.remote.ssh is None

    @property
    def host_label(self) -> str:
        return "localhost" if self.remote.ssh is None else self.remote.ssh.host

    def control_path(self) -> Path:
        """The master connection's socket under TMPDIR, one per ssh setting.

        Named by a short hash rather than ssh's 40-character ``%C``, so a
        TMPDIR as deep as an agent turn's still fits a unix socket path.
        """

        ssh = self.remote.ssh
        assert ssh is not None
        key = json.dumps([ssh.host, ssh.port, ssh.identity_file, list(ssh.options)])
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        path = Path(tempfile.gettempdir()).resolve() / f"vssh-{digest}"
        if len(str(path).encode()) > CONTROL_PATH_LIMIT:
            raise TransportError(
                f"the ssh control socket {path} is too long for a unix socket "
                f"(over {CONTROL_PATH_LIMIT} bytes); point TMPDIR at a shorter directory"
            )
        return path

    def ssh_argv(self) -> list[str]:
        """The ssh command (without the destination) every call shares."""

        ssh = self.remote.ssh
        assert ssh is not None
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={self.control_path()}",
            "-o",
            "ControlPersist=120",
            "-o",
            "ServerAliveInterval=30",
        ]
        known_hosts = self.remote.config_dir / ".remote" / "known_hosts"
        if known_hosts.is_file():
            argv += [
                "-o",
                f"UserKnownHostsFile={known_hosts}",
                "-o",
                "StrictHostKeyChecking=yes",
            ]
        if ssh.port is not None:
            argv += ["-p", str(ssh.port)]
        if ssh.identity_file is not None:
            argv += ["-i", ssh.identity_file, "-o", "IdentitiesOnly=yes"]
        return argv + list(ssh.options)

    def command_argv(self, script: str) -> list[str]:
        if self.remote.ssh is None:
            return ["bash", "-lc", script]
        # ssh joins its arguments into one string for the remote shell, so the
        # script is quoted once here and unquoted once there.
        return [*self.ssh_argv(), self.remote.ssh.host, "bash -lc " + shlex.quote(script)]

    def run(
        self,
        script: str,
        *,
        check: bool = True,
        input: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run ``script`` on the access host and capture its output."""

        try:
            result = subprocess.run(
                self.command_argv(script),
                input=input,
                capture_output=True,
                text=True,
                # A log chunk can end inside a multi-byte character.
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            message = f"{self.host_label}: timed out after {timeout}s: {_first_line(script)}"
            raise TransportError(message) from exc
        if check and result.returncode != 0:
            raise TransportError(
                f"{self.host_label}: exit {result.returncode}: {_first_line(script)}\n"
                f"{(result.stderr or result.stdout).strip()}"
            )
        return result

    def stream(self, script: str) -> int:
        """Run ``script`` with its output on this terminal; return its exit code."""

        return subprocess.run(self.command_argv(script), check=False).returncode

    def write_text(self, path: PurePosixPath, text: str) -> None:
        self.run(f"mkdir -p {q(path.parent)} && cat > {q(path)}", input=text)

    def read_text(self, path: PurePosixPath) -> str | None:
        result = self.run(f"cat {q(path)} 2>/dev/null", check=False)
        return result.stdout if result.returncode == 0 else None

    def rsync(
        self,
        source: str,
        destination: str,
        *,
        options: Sequence[str] = (),
        input: str | None = None,
    ) -> None:
        """``rsync -a`` between this machine and the access host.

        One side is a local path and the other a remote one, written as
        ``remote_spec(path)``. Paths ending in ``/`` copy a directory's contents.
        """

        argv = ["rsync", "-a", "--partial", *options]
        if self.remote.ssh is not None:
            argv += ["-e", shlex.join(self.ssh_argv())]
        argv += [source, destination]
        result = subprocess.run(argv, input=input, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise TransportError(
                f"rsync {source} -> {destination}: exit {result.returncode}\n"
                f"{result.stderr.strip()}"
            )

    def remote_spec(self, path: PurePosixPath | str, *, directory: bool = False) -> str:
        text = str(path) + ("/" if directory else "")
        return text if self.remote.ssh is None else f"{self.remote.ssh.host}:{text}"


def q(path: PurePosixPath | str) -> str:
    return shlex.quote(str(path))


def _first_line(script: str) -> str:
    line = script.strip().splitlines()[0] if script.strip() else ""
    return line if len(line) <= 160 else line[:157] + "..."
