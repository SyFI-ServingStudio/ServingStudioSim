"""Worker-only loader for the pinned vLLM persistent top-k CUDA extension.

The registry and runner modules may import this module safely: it imports no
Torch/CUDA package and creates no cache files. ``load_persistent_topk_op`` is the
only entry point that validates the committed source, acquires the cross-process
build lock, and loads or builds the private Torch operator.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ASSET_ROOT = Path(__file__).resolve().parent
_MANIFEST_PATH = _ASSET_ROOT / "source_manifest.json"
_CACHE_ENV = "VIBESIM_DSA_PERSISTENT_TOPK_CACHE_DIR"
_EXPECTED_TORCH = "2.10.0+cu128"
_EXPECTED_CUDA = "12.8"
_SCHEMA = (
    "_C_pinned_topk::persistent_topk(Tensor logits, Tensor lengths, "
    "Tensor($0! -> ) output, Tensor workspace, int k, int max_seq_len) -> ()"
)
_REQUIRED_UPSTREAM_HASHES = {
    "upstream/topk.cu": "2c90ef9391e1d6bd6ca65c05841597569cf629451f25a1ed4446aa5b34f1d917",
    "upstream/persistent_topk.cuh": (
        "1d92c234493599e4d57d793eda2cf3b8efa246415425dd2ac881935b25b950ee"
    ),
}
_CFLAGS = (
    "-O3",
    "-std=c++20",
    "-DPy_LIMITED_API=3",
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DUSE_CUDA",
)
_CUDA_CFLAGS = (*_CFLAGS, "-gencode=arch=compute_90,code=sm_90")


class NativeExtensionUnsupported(RuntimeError):
    """The worker stack cannot build the verified pinned extension."""


class NativeExtensionBuildError(RuntimeError):
    """The verified source failed to compile or link."""


class NativeExtensionLoadError(RuntimeError):
    """A built library failed source/schema validation or dynamic loading."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(asset_root: Path = _ASSET_ROOT) -> dict[str, Any]:
    manifest_path = asset_root / _MANIFEST_PATH.name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeExtensionLoadError(
            f"cannot read pinned persistent-topk source manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("format_version") != 1:
        raise NativeExtensionLoadError(
            "unsupported persistent-topk source manifest format: "
            f"{manifest.get('format_version')!r}"
        )
    return manifest


def verify_vendored_sources(asset_root: Path = _ASSET_ROOT) -> dict[str, str]:
    """Hash every manifest source and return its verified local digest."""
    manifest = _read_manifest(asset_root)
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise NativeExtensionLoadError("persistent-topk source manifest has no files")

    expected_by_path: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise NativeExtensionLoadError("persistent-topk source manifest has an invalid record")
        relative = record.get("local_path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise NativeExtensionLoadError(
                "persistent-topk source manifest records require local_path and sha256"
            )
        path = (asset_root / relative).resolve()
        try:
            path.relative_to(asset_root.resolve())
        except ValueError as exc:
            raise NativeExtensionLoadError(
                f"persistent-topk manifest path escapes asset root: {relative!r}"
            ) from exc
        if not path.is_file():
            raise NativeExtensionLoadError(f"missing pinned persistent-topk source: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise NativeExtensionLoadError(
                f"pinned persistent-topk source hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        expected_by_path[relative] = actual

    for relative, required_hash in _REQUIRED_UPSTREAM_HASHES.items():
        if expected_by_path.get(relative) != required_hash:
            raise NativeExtensionLoadError(
                f"manifest does not pin required upstream source {relative} to {required_hash}"
            )
    return expected_by_path


def _validate_runtime(torch: Any) -> None:
    torch_version = str(getattr(torch, "__version__", "unknown"))
    cuda_version = str(getattr(getattr(torch, "version", None), "cuda", None))
    if torch_version != _EXPECTED_TORCH:
        raise NativeExtensionUnsupported(
            "pinned persistent-topk is verified only with "
            f"Torch {_EXPECTED_TORCH}, got {torch_version}"
        )
    if cuda_version != _EXPECTED_CUDA:
        raise NativeExtensionUnsupported(
            "pinned persistent-topk is verified only with CUDA "
            f"{_EXPECTED_CUDA}, got {cuda_version}"
        )


def _workspace_root() -> Path:
    # loader.py -> native -> attention -> runners -> profiling -> workspace
    return Path(__file__).resolve().parents[4]


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return _workspace_root() / "tmp" / "native_extensions" / "dsa_persistent_topk_decode"


def _build_fingerprint(torch: Any, source_hashes: dict[str, str]) -> str:
    payload = {
        "sources": source_hashes,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "python": platform.python_version(),
        "cflags": _CFLAGS,
        "cuda_cflags": _CUDA_CFLAGS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _exclusive_build_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _registered_op(torch: Any) -> Any | None:
    try:
        op = torch.ops._C_pinned_topk.persistent_topk
        schema = str(op.default._schema)
    except (AttributeError, RuntimeError):
        return None
    if schema != _SCHEMA:
        raise NativeExtensionLoadError(
            f"private persistent-topk schema mismatch: expected {_SCHEMA!r}, got {schema!r}"
        )
    return op


def _nvcc_version(cuda_home: Path) -> str:
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise NativeExtensionUnsupported(f"CUDA nvcc is unavailable at {nvcc}")
    try:
        completed = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise NativeExtensionUnsupported(f"cannot execute CUDA nvcc at {nvcc}: {exc}") from exc
    if "release 12.8" not in completed.stdout:
        raise NativeExtensionUnsupported(
            "pinned persistent-topk requires CUDA toolkit 12.8; "
            f"nvcc reported: {completed.stdout.strip()}"
        )
    return completed.stdout.strip().splitlines()[-1]


def _load_library(torch: Any, artifact: Path) -> Any:
    try:
        torch.ops.load_library(str(artifact))
    except (OSError, RuntimeError) as exc:
        raise NativeExtensionLoadError(
            f"failed to load pinned persistent-topk library {artifact}: {exc}"
        ) from exc
    op = _registered_op(torch)
    if op is None:
        raise NativeExtensionLoadError(f"library {artifact} loaded without registering {_SCHEMA}")
    return op


def _build_library(
    torch: Any,
    *,
    build_dir: Path,
    extension_name: str,
) -> Path:
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except (ImportError, OSError) as exc:
        raise NativeExtensionUnsupported(
            f"Torch CUDA extension build support is unavailable: {exc}"
        ) from exc
    if CUDA_HOME is None:
        raise NativeExtensionUnsupported("Torch CUDA_HOME is unset; CUDA 12.8 toolkit is required")
    _nvcc_version(Path(CUDA_HOME))

    build_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifact = load(
            name=extension_name,
            sources=[
                str(_ASSET_ROOT / "upstream" / "topk.cu"),
                str(_ASSET_ROOT / "binding.cpp"),
            ],
            extra_include_paths=[str(_ASSET_ROOT / "upstream")],
            extra_cflags=list(_CFLAGS),
            extra_cuda_cflags=list(_CUDA_CFLAGS),
            extra_ldflags=["-Wl,--no-as-needed"],
            with_cuda=True,
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=os.environ.get("VIBESIM_NATIVE_BUILD_VERBOSE") == "1",
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise NativeExtensionBuildError(
            "failed to build pinned persistent-topk extension from verified vendored "
            f"sources in {build_dir}: {exc}"
        ) from exc
    artifact_path = Path(artifact).resolve()
    if not artifact_path.is_file():
        raise NativeExtensionBuildError(
            f"Torch extension build did not produce a library: {artifact_path}"
        )
    return artifact_path


def _write_complete_marker(
    marker: Path,
    *,
    artifact: Path,
    fingerprint: str,
    source_hashes: dict[str, str],
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "source_sha256": source_hashes,
        "schema": _SCHEMA,
    }
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)


def load_persistent_topk_op(torch: Any) -> Any:
    """Return the verified private op, building it once under a process lock."""
    source_hashes = verify_vendored_sources()
    _validate_runtime(torch)

    existing = _registered_op(torch)
    if existing is not None:
        return existing

    fingerprint = _build_fingerprint(torch, source_hashes)
    extension_name = f"vibesim_dsa_persistent_topk_{fingerprint[:16]}"
    cache_root = _cache_root()
    build_dir = cache_root / fingerprint
    lock_path = cache_root / f"{fingerprint}.lock"
    artifact = build_dir / f"{extension_name}.so"
    marker = build_dir / "complete.json"

    with _exclusive_build_lock(lock_path):
        # Another worker may have completed while this process waited.
        existing = _registered_op(torch)
        if existing is not None:
            return existing

        if marker.is_file() and artifact.is_file():
            try:
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise NativeExtensionLoadError(
                    f"cannot read pinned persistent-topk completion marker {marker}: {exc}"
                ) from exc
            if marker_payload.get("fingerprint") != fingerprint:
                raise NativeExtensionLoadError(
                    f"persistent-topk cache marker fingerprint mismatch in {marker}"
                )
            expected_artifact_hash = marker_payload.get("artifact_sha256")
            actual_artifact_hash = _sha256(artifact)
            if expected_artifact_hash != actual_artifact_hash:
                raise NativeExtensionLoadError(
                    f"persistent-topk cached library hash mismatch for {artifact}: "
                    f"expected {expected_artifact_hash}, got {actual_artifact_hash}"
                )
            return _load_library(torch, artifact)

        artifact = _build_library(
            torch,
            build_dir=build_dir,
            extension_name=extension_name,
        )
        op = _registered_op(torch)
        if op is None:
            # ``cpp_extension.load(..., is_python_module=False)`` loads the
            # library itself; retain an explicit fallback for toolchain changes.
            op = _load_library(torch, artifact)
        _write_complete_marker(
            marker,
            artifact=artifact,
            fingerprint=fingerprint,
            source_hashes=source_hashes,
        )
        return op


__all__ = [
    "NativeExtensionBuildError",
    "NativeExtensionLoadError",
    "NativeExtensionUnsupported",
    "load_persistent_topk_op",
    "verify_vendored_sources",
]
