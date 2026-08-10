"""Worker-only loader for the corrected vLLM persistent top-k CUDA extension.

The registry and runner modules may import this module safely: it imports no
Torch/CUDA package and creates no cache files. ``load_persistent_topk_op`` is the
only entry point that validates the pinned source and correction asset, derives
the corrected build source under the cross-process lock, and loads or builds the
private Torch operator.
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
_CORRECTION_MANIFEST_PATH = _ASSET_ROOT / "correction_manifest.json"
_CACHE_ENV = "VIBESIM_DSA_PERSISTENT_TOPK_CACHE_DIR"
_EXPECTED_TORCH = "2.11.0+cu128"
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
_CORRECTION_ID = "all-cooperative-radix-with-radix-iteration-v1"
_CORRECTION_ASSET_SHA256 = "a6e00159afa31b110950e1fbfcbe3118df8c549bf56f01052f7eeb99c549d001"
_CORRECTED_SOURCE_HASHES = {
    "topk.cu": "937bece0a889b353b1c2e9148a55f00c37a1791cd1d5e6f05bb7fb98da675084",
    "persistent_topk.cuh": ("e0aeea8a0bb7d4be12c45b7d643c24489054d8410aa0e07648cb291050411a94"),
}
_CFLAGS = (
    "-O3",
    "-std=c++20",
    "-DPy_LIMITED_API=3",
    "-DTORCH_TARGET_VERSION=0x020B000000000000",
    "-DUSE_CUDA",
)
_CUDA_CFLAGS = (*_CFLAGS, "-gencode=arch=compute_90,code=sm_90")


class NativeExtensionUnsupported(RuntimeError):
    """The worker stack cannot build the verified corrected extension."""


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


def _read_correction_manifest(asset_root: Path = _ASSET_ROOT) -> dict[str, Any]:
    path = asset_root / _CORRECTION_MANIFEST_PATH.name
    try:
        correction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeExtensionLoadError(
            f"cannot read persistent-topk correction manifest {path}: {exc}"
        ) from exc
    if correction.get("format_version") != 1:
        raise NativeExtensionLoadError(
            f"unsupported persistent-topk correction format: {correction.get('format_version')!r}"
        )
    if correction.get("correction_id") != _CORRECTION_ID:
        raise NativeExtensionLoadError(
            "persistent-topk correction identity mismatch: expected "
            f"{_CORRECTION_ID!r}, got {correction.get('correction_id')!r}"
        )
    targets = correction.get("targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise NativeExtensionLoadError(
            "persistent-topk correction must describe exactly two source targets"
        )
    expected_targets = {
        "upstream/topk.cu": (
            _REQUIRED_UPSTREAM_HASHES["upstream/topk.cu"],
            _CORRECTED_SOURCE_HASHES["topk.cu"],
            3,
        ),
        "upstream/persistent_topk.cuh": (
            _REQUIRED_UPSTREAM_HASHES["upstream/persistent_topk.cuh"],
            _CORRECTED_SOURCE_HASHES["persistent_topk.cuh"],
            4,
        ),
    }
    actual_paths: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise NativeExtensionLoadError("persistent-topk correction target is invalid")
        relative = target.get("local_path")
        if relative not in expected_targets or relative in actual_paths:
            raise NativeExtensionLoadError(
                f"unexpected persistent-topk correction target: {relative!r}"
            )
        actual_paths.add(relative)
        expected_input, expected_result, expected_operation_count = expected_targets[relative]
        if target.get("input_sha256") != expected_input:
            raise NativeExtensionLoadError(
                f"persistent-topk correction input hash mismatch for {relative}"
            )
        if target.get("result_sha256") != expected_result:
            raise NativeExtensionLoadError(
                f"persistent-topk correction result hash mismatch for {relative}"
            )
        operations = target.get("operations")
        if not isinstance(operations, list) or len(operations) != expected_operation_count:
            raise NativeExtensionLoadError(
                f"persistent-topk correction operation count mismatch for {relative}"
            )
        for operation in operations:
            if (
                not isinstance(operation, dict)
                or operation.get("kind") != "exact_utf8_replacement"
                or not isinstance(operation.get("before"), str)
                or not isinstance(operation.get("after"), str)
                or operation.get("expected_occurrences") != 1
            ):
                raise NativeExtensionLoadError(
                    f"invalid persistent-topk correction operation for {relative}"
                )
    return correction


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
    if expected_by_path.get(_CORRECTION_MANIFEST_PATH.name) != _CORRECTION_ASSET_SHA256:
        raise NativeExtensionLoadError(
            "manifest does not pin the required persistent-topk correction asset to "
            f"{_CORRECTION_ASSET_SHA256}"
        )
    _read_correction_manifest(asset_root)
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


def _correction_identity(correction: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": correction["correction_id"],
        "asset_sha256": _CORRECTION_ASSET_SHA256,
        "result_sha256": {
            target["local_path"]: target["result_sha256"] for target in correction["targets"]
        },
    }


def _expected_derived_hashes(source_hashes: dict[str, str]) -> dict[str, str]:
    return {
        "binding.cpp": source_hashes["binding.cpp"],
        "persistent_topk.cuh": _CORRECTED_SOURCE_HASHES["persistent_topk.cuh"],
        "topk.cu": _CORRECTED_SOURCE_HASHES["topk.cu"],
        "torch_utils.h": source_hashes["upstream/torch_utils.h"],
    }


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def _derive_corrected_sources(
    source_dir: Path,
    *,
    asset_root: Path = _ASSET_ROOT,
) -> dict[str, str]:
    """Materialize the verified overlay without modifying pinned assets."""
    correction = _read_correction_manifest(asset_root)
    source_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, bytes] = {
        "binding.cpp": (asset_root / "binding.cpp").read_bytes(),
        "torch_utils.h": (asset_root / "upstream" / "torch_utils.h").read_bytes(),
    }
    for target in correction["targets"]:
        relative = target["local_path"]
        source = (asset_root / relative).read_text(encoding="utf-8")
        for operation in target["operations"]:
            before = operation["before"]
            occurrences = source.count(before)
            if occurrences != operation["expected_occurrences"]:
                raise NativeExtensionLoadError(
                    "persistent-topk correction occurrence mismatch for "
                    f"{relative}: expected {operation['expected_occurrences']}, "
                    f"got {occurrences}"
                )
            source = source.replace(before, operation["after"])
        output_name = Path(relative).name
        generated[output_name] = source.encode("utf-8")
        actual_result = hashlib.sha256(generated[output_name]).hexdigest()
        if actual_result != target["result_sha256"]:
            raise NativeExtensionLoadError(
                f"persistent-topk corrected source hash mismatch for {relative}: "
                f"expected {target['result_sha256']}, got {actual_result}"
            )
    for name, contents in generated.items():
        _atomic_write_bytes(source_dir / name, contents)
    return {path.name: _sha256(path) for path in sorted(source_dir.iterdir()) if path.is_file()}


def _verify_derived_sources(source_dir: Path, expected_hashes: dict[str, str]) -> None:
    actual = (
        {path.name: _sha256(path) for path in sorted(source_dir.iterdir()) if path.is_file()}
        if source_dir.is_dir()
        else {}
    )
    if actual != expected_hashes:
        raise NativeExtensionLoadError(
            f"persistent-topk corrected build-source mismatch in {source_dir}: "
            f"expected {expected_hashes}, got {actual}"
        )


def _build_fingerprint(
    torch: Any,
    source_hashes: dict[str, str],
    correction: dict[str, Any],
) -> str:
    payload = {
        "sources": source_hashes,
        "correction": _correction_identity(correction),
        "derived_sources": _expected_derived_hashes(source_hashes),
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
    source_dir: Path,
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
                str(source_dir / "topk.cu"),
                str(source_dir / "binding.cpp"),
            ],
            extra_include_paths=[str(source_dir)],
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
            "failed to build corrected persistent-topk extension from verified derived "
            f"sources in {source_dir}: {exc}"
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
    derived_source_hashes: dict[str, str],
    correction: dict[str, Any],
) -> None:
    payload = {
        "fingerprint": fingerprint,
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "source_sha256": source_hashes,
        "derived_source_sha256": derived_source_hashes,
        "correction": _correction_identity(correction),
        "schema": _SCHEMA,
    }
    temporary = marker.with_name(f"{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)


def _validate_complete_marker(
    marker: Path,
    *,
    artifact: Path,
    fingerprint: str,
    source_hashes: dict[str, str],
    derived_source_hashes: dict[str, str],
    correction: dict[str, Any],
    source_dir: Path,
) -> None:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeExtensionLoadError(
            f"cannot read corrected persistent-topk completion marker {marker}: {exc}"
        ) from exc
    expected_fields = {
        "fingerprint": fingerprint,
        "artifact": artifact.name,
        "source_sha256": source_hashes,
        "derived_source_sha256": derived_source_hashes,
        "correction": _correction_identity(correction),
        "schema": _SCHEMA,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise NativeExtensionLoadError(
                f"persistent-topk cache marker {field} mismatch in {marker}"
            )
    _verify_derived_sources(source_dir, derived_source_hashes)
    actual_artifact_hash = _sha256(artifact)
    if payload.get("artifact_sha256") != actual_artifact_hash:
        raise NativeExtensionLoadError(
            f"persistent-topk cached library hash mismatch for {artifact}: "
            f"expected {payload.get('artifact_sha256')}, got {actual_artifact_hash}"
        )


def load_persistent_topk_op(torch: Any) -> Any:
    """Return the corrected private op, building it once under a process lock."""
    source_hashes = verify_vendored_sources()
    correction = _read_correction_manifest()
    _validate_runtime(torch)

    existing = _registered_op(torch)
    if existing is not None:
        return existing

    fingerprint = _build_fingerprint(torch, source_hashes, correction)
    extension_name = f"vibesim_dsa_persistent_topk_{fingerprint[:16]}"
    cache_root = _cache_root()
    build_dir = cache_root / fingerprint
    source_dir = build_dir / "corrected_source"
    lock_path = cache_root / f"{fingerprint}.lock"
    artifact = build_dir / f"{extension_name}.so"
    marker = build_dir / "complete.json"
    expected_derived_hashes = _expected_derived_hashes(source_hashes)

    with _exclusive_build_lock(lock_path):
        # Another worker may have completed while this process waited.
        existing = _registered_op(torch)
        if existing is not None:
            return existing

        if marker.is_file() and artifact.is_file():
            _validate_complete_marker(
                marker,
                artifact=artifact,
                fingerprint=fingerprint,
                source_hashes=source_hashes,
                derived_source_hashes=expected_derived_hashes,
                correction=correction,
                source_dir=source_dir,
            )
            return _load_library(torch, artifact)

        derived_source_hashes = _derive_corrected_sources(source_dir)
        if derived_source_hashes != expected_derived_hashes:
            raise NativeExtensionLoadError(
                "persistent-topk corrected source derivation produced unexpected hashes: "
                f"expected {expected_derived_hashes}, got {derived_source_hashes}"
            )
        artifact = _build_library(
            torch,
            build_dir=build_dir,
            source_dir=source_dir,
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
            derived_source_hashes=derived_source_hashes,
            correction=correction,
        )
        return op


__all__ = [
    "NativeExtensionBuildError",
    "NativeExtensionLoadError",
    "NativeExtensionUnsupported",
    "load_persistent_topk_op",
    "verify_vendored_sources",
]
