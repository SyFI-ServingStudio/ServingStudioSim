"""Where the launcher finds the simulator, analyzer and parameter schema.

Normally they are the cargo outputs under ``target/<build_type>/``, built by
``exec.cargo_build`` at the start of every batch. With ``VIBESIM_BIN_DIR`` set,
the launcher uses an already built set instead and never runs cargo: `--remote`
builds once per code version on the remote and runs every job from its own
copy (`launcher/remote`), so a later build cannot replace a binary under a
running job.

A set holds ``simulator``, ``analyze`` (optional: without it post-run analysis
is skipped, as after a failed analyzer build), ``deployment_schema.json``, and
two files that tie it to its sources: ``.build-inputs`` lists every file the
binaries were compiled from (relative to the checkout; taken from cargo's
dep-info, so ``include_str!`` data such as ``gpu/spec.json`` is covered), and
``.build-hash`` is `build_inputs_hash` over them. The launcher recomputes that
hash against its own checkout and refuses a set built from other sources.
"""

from __future__ import annotations

import hashlib
import os
import sysconfig
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR_ENV = "VIBESIM_BIN_DIR"
BUILD_HASH_NAME = ".build-hash"
BUILD_INPUTS_NAME = ".build-inputs"
PREBUILT_BUILD_TYPE = "release"
# Inputs cargo reads but does not list in a binary's dep-info.
MANIFEST_INPUTS = ("Cargo.toml", "Cargo.lock", ".cargo/config.toml")


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


def dep_info_inputs(root: Path, dep_info_files: Iterable[Path]) -> list[str]:
    """The checkout-relative sources named by cargo ``.d`` files, plus manifests.

    A dep-info file is one make rule, ``target: dep dep ...``, with spaces in
    paths escaped as ``\\ ``. Paths outside ``root`` (none, in practice) are not
    part of the checkout and are left out.
    """

    root = root.resolve()
    inputs: set[str] = set()
    for dep_info in dep_info_files:
        text = dep_info.read_text("utf-8")
        rule = text.split("\n", 1)[0]
        _, _, deps = rule.partition(": ")
        for token in deps.replace("\\ ", "\0").split():
            # Resolved: cargo may spell the workspace through a symlinked parent.
            path = Path(token.replace("\0", " ")).resolve()
            if path.is_relative_to(root):
                inputs.add(path.relative_to(root).as_posix())
    # Each compiled crate's own manifest, found by walking up from its sources.
    for name in list(inputs):
        for parent in Path(name).parents:
            if (root / parent / "Cargo.toml").is_file():
                inputs.add((parent / "Cargo.toml").as_posix())
                break
    inputs.update(name for name in MANIFEST_INPUTS if (root / name).is_file())
    return sorted(inputs)


def build_inputs_hash(root: Path, inputs: Iterable[str]) -> str:
    """Content hash of a build's inputs and the interpreter it links against.

    build.rs bakes the embedding Python's LIBDIR into the binary's rpath, so a
    set built against another interpreter does not run here either.
    """

    digest = hashlib.sha256()
    digest.update(f"libdir\0{sysconfig.get_config_var('LIBDIR') or ''}\0".encode())
    for name in sorted(inputs):
        path = root / name
        digest.update(name.encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b"-")
    return digest.hexdigest()


def record_build(directory: Path, root: Path, inputs: list[str]) -> str:
    build_hash = build_inputs_hash(root, inputs)
    (directory / BUILD_INPUTS_NAME).write_text("\n".join(inputs) + "\n", "utf-8")
    (directory / BUILD_HASH_NAME).write_text(build_hash + "\n", "utf-8")
    return build_hash


def check_prebuilt(directory: Path, root: Path | None = None) -> None:
    """Refuse a prebuilt set that is incomplete or was built from other code."""

    root = REPO_ROOT if root is None else root
    simulator = directory / "simulator"
    if not os.access(simulator, os.X_OK):
        raise PrebuiltBinaryError(f"{BIN_DIR_ENV}: no executable simulator in {directory}")
    if not (directory / "deployment_schema.json").is_file():
        raise PrebuiltBinaryError(f"{BIN_DIR_ENV}: no deployment_schema.json in {directory}")
    inputs_path, hash_path = directory / BUILD_INPUTS_NAME, directory / BUILD_HASH_NAME
    if not (inputs_path.is_file() and hash_path.is_file()):
        raise PrebuiltBinaryError(
            f"{BIN_DIR_ENV}: {directory} does not record its build inputs "
            f"({BUILD_INPUTS_NAME}, {BUILD_HASH_NAME})"
        )
    inputs = inputs_path.read_text("utf-8").split()
    expected = hash_path.read_text("utf-8").strip()
    if build_inputs_hash(root, inputs) != expected:
        raise PrebuiltBinaryError(
            f"{BIN_DIR_ENV}: {directory} was built from other sources (or another Python) "
            f"than {root}"
        )
