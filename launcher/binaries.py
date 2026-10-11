"""Where the launcher finds the simulator, analyzer and parameter schema.

Normally they are the cargo outputs under ``target/<build_type>/``, built by
``exec.cargo_build`` at the start of every batch. With ``VIBESIM_BIN_DIR`` set,
the launcher uses an already built set instead and never runs cargo: `--remote`
builds once per code version on the remote and runs every job from that
version's own copy (`launcher/remote`), so a later build cannot replace a binary
under a running job. The directory holds ``simulator``, ``analyze`` (optional:
without it post-run analysis is skipped, as after a failed analyzer build),
``deployment_schema.json``, and ``.build-hash``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .source_stamp import read_source_stamp

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR_ENV = "VIBESIM_BIN_DIR"
BUILD_HASH_NAME = ".build-hash"
PREBUILT_BUILD_TYPE = "release"


class PrebuiltBinaryError(RuntimeError):
    """``VIBESIM_BIN_DIR`` names a set that cannot serve this checkout."""


def prebuilt_dir() -> Path | None:
    value = os.environ.get(BIN_DIR_ENV, "").strip()
    return Path(value) if value else None


def build_output_dir(build_type: str) -> Path:
    directory = prebuilt_dir()
    if directory is None:
        return REPO_ROOT / "target" / build_type
    if build_type != PREBUILT_BUILD_TYPE:
        raise PrebuiltBinaryError(
            f"{BIN_DIR_ENV} holds {PREBUILT_BUILD_TYPE} binaries; --build-type {build_type} "
            "needs a cargo build (unset the variable)"
        )
    return directory


def check_prebuilt(directory: Path) -> None:
    """Refuse a prebuilt set that is incomplete or was built from other code.

    When this checkout carries a source stamp (a tree shipped by `--remote`),
    the set's ``.build-hash`` must equal the stamp's ``build_hash``: the Rust
    side of a run would otherwise come from different sources than the Python
    side it embeds.
    """

    simulator = directory / "simulator"
    if not os.access(simulator, os.X_OK):
        raise PrebuiltBinaryError(f"{BIN_DIR_ENV}: no executable simulator in {directory}")
    if not (directory / "deployment_schema.json").is_file():
        raise PrebuiltBinaryError(f"{BIN_DIR_ENV}: no deployment_schema.json in {directory}")
    stamp = read_source_stamp(REPO_ROOT)
    if stamp is None:
        return
    hash_path = directory / BUILD_HASH_NAME
    built_from = hash_path.read_text("utf-8").strip() if hash_path.is_file() else None
    if built_from != stamp["build_hash"]:
        raise PrebuiltBinaryError(
            f"{BIN_DIR_ENV}: {directory} was built from {built_from or 'unrecorded sources'}, "
            f"but this checkout's build inputs are {stamp['build_hash']}"
        )
