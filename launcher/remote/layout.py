"""The directory layout `--remote` keeps under a remote's ``workdir``.

Only code is kept per version; everything heavy exists once per remote::

    src/<version>/          one shipped tree, hard-linked to the previous version
        .source-stamp.json  where it came from (`launcher/source_stamp.py`)
        .venv -> ../../venvs/<venv_key>
        bin   -> ../../bin/<build_hash>
        logs  -> ../../logs
    venvs/<venv_key>/       made by `uv sync` once per lock and group set
    build/src/              the one checkout cargo builds in; a fixed path keeps
                            its incremental state valid across versions
    build/target/           the one cargo target dir (`build/src/target` links here)
    bin/<build_hash>/       binaries of one set of build inputs (`launcher/binaries.py`)
    logs/                   every experiment, kept (the same relative `logs/...`
                            paths as on the sending machine, so resume works)
    jobs/<job_id>/          one remote invocation: job.json, stdout.log, exit_code

The workdir may sit on a filesystem the access host shares with compute
nodes; nothing here relies on file locks working across hosts (`node.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class Layout:
    workdir: PurePosixPath

    def src(self, version: str) -> PurePosixPath:
        return self.workdir / "src" / version

    def venv(self, key: str) -> PurePosixPath:
        return self.workdir / "venvs" / key

    @property
    def build_src(self) -> PurePosixPath:
        return self.workdir / "build" / "src"

    @property
    def build_target(self) -> PurePosixPath:
        return self.workdir / "build" / "target"

    def bin(self, build_hash: str) -> PurePosixPath:
        return self.workdir / "bin" / build_hash

    @property
    def logs(self) -> PurePosixPath:
        return self.workdir / "logs"

    def job(self, job_id: str) -> PurePosixPath:
        return self.workdir / "jobs" / job_id

    def python(self, version: str) -> PurePosixPath:
        return self.src(version) / ".venv" / "bin" / "python"
