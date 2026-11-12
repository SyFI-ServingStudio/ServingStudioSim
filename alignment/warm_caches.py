"""Autotune policy of a capture, and the warm-caches manifest that admits it.

A capture measures the engine as it serves, so it must run the kernels an
earlier, deliberate warm-up compiled and tuned, not tune them itself: a
FlashInfer tactic search inside a capture took 8.5 min of a TP4 vLLM start,
and its winners depend on whatever it happened to run first. The policy
(`VIBESIM_AUTOTUNE`, read by `reference_engines/autotune_guard.py` in every
engine process) is decided here, for every capture:

    capture (`launcher alignment profile --runtime image`)   forbid
    capture with --allow-autotune                             allow
    `python -m reference_engines warm-caches`                 allow

Two layers enforce it. Before the container starts, `preflight` looks for the
manifest a warm-up of this deployment wrote, and fails at once, naming
warm-caches, when there is none. The guard in the engine is the backstop: if
the manifest said warm and the engine still tunes, it reports a
manifest/actual mismatch, which means `deployment_key` misses an input of the
tuning and must be completed.

The key is deliberately the deployment as the config states it, not a copy of
an engine's own cache hashes (vLLM's autotune file hashes its whole compile
config): those are the engine's to compute, and the guard catches the cases
where the two disagree.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from reference_engines.autotune_guard import ALLOW, FORBID, AutotuneForbiddenError

MANIFEST_DIR = "warm-manifests"

# Server fields that reach the engine but not what it compiles or tunes.
_NOT_TUNING = ("host", "port", "startup_timeout", "served_model_name", "model_path")


def capture_policy(*, allow_autotune: bool, warm_caches: bool) -> str:
    """`VIBESIM_AUTOTUNE` for one engine-image capture container."""

    return ALLOW if allow_autotune or warm_caches else FORBID


def deployment_key(cfg: Any) -> dict[str, Any]:
    """What a warm-up has to have matched: model, parallel layout, server args."""

    server = dataclasses.asdict(cfg.server)
    for name in _NOT_TUNING:
        server.pop(name, None)
    devices = [token for token in cfg.cuda_visible_devices.split(",") if token.strip()]
    return {
        "engine": cfg.engine,
        "model": str(Path(cfg.server.model_path).resolve()),
        "gpus": len(devices),
        "server": server,
        "environment": dict(sorted((cfg.python_runtime.environment or {}).items()))
        if cfg.python_runtime is not None
        else {},
    }


def manifest_path(cache_dir: Path, cache_key: str, key: dict[str, Any]) -> Path:
    digest = hashlib.sha256(
        json.dumps({"cache_key": cache_key, **key}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return cache_dir / MANIFEST_DIR / f"{digest}.json"


def _differences(wanted: dict[str, Any], found: dict[str, Any]) -> list[str]:
    out = []
    for name in sorted(set(wanted) | set(found)):
        if name == "server" and isinstance(found.get(name), dict):
            out += [
                f"server.{field}: {found['server'].get(field)!r} != {wanted['server'].get(field)!r}"
                for field in sorted(set(wanted["server"]) | set(found["server"]))
                if wanted["server"].get(field) != found["server"].get(field)
            ]
        elif wanted.get(name) != found.get(name):
            out.append(f"{name}: {found.get(name)!r} != {wanted.get(name)!r}")
    return out


def preflight(cfg: Any, config_path: Path, cache_dir: Path, cache_key: str) -> Path:
    """The manifest admitting this capture, or `AutotuneForbiddenError` at once."""

    key = deployment_key(cfg)
    path = manifest_path(cache_dir, cache_key, key)
    if path.is_file():
        return path
    nearest = []
    for other in sorted((cache_dir / MANIFEST_DIR).glob("*.json")):
        try:
            recorded = json.loads(other.read_text())["key"]
        except (OSError, ValueError, KeyError):
            continue
        if recorded.get("engine") == key["engine"] and recorded.get("model") == key["model"]:
            nearest.append(f"  {other.name}: " + "; ".join(_differences(key, recorded)))
    raise AutotuneForbiddenError(
        f"no warm-caches manifest for this deployment in {cache_dir / MANIFEST_DIR} "
        f"(cache key {cache_key}), so the capture would tune kernels, which a capture "
        f"may not. Warm it first: `just ref warm-caches {cfg.engine} {config_path}`, "
        "or pass --allow-autotune to tune inside this capture deliberately."
        + (
            "\nManifests of this model differ in (recorded != wanted):\n" + "\n".join(nearest)
            if nearest
            else ""
        )
    )


def snapshot(cache_dir: Path) -> dict[str, tuple[int, int]]:
    """Every file under `cache_dir` (except the manifests): size and mtime."""

    files = {}
    for path in cache_dir.rglob("*"):
        if path.is_file() and MANIFEST_DIR not in path.relative_to(cache_dir).parts[:1]:
            stat = path.stat()
            files[path.relative_to(cache_dir).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return files


def write_manifest(
    cfg: Any,
    config_path: Path,
    cache_dir: Path,
    cache_key: str,
    *,
    image: str,
    image_id: str,
    before: dict[str, tuple[int, int]],
    seconds: float,
) -> Path:
    """Record a finished warm-up: the key it satisfies and the files it wrote."""

    key = deployment_key(cfg)
    after = snapshot(cache_dir)
    produced = sorted(name for name, stat in after.items() if before.get(name) != stat)
    path = manifest_path(cache_dir, cache_key, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "cache_key": cache_key,
        "key": key,
        "image": image,
        "image_id": image_id,
        "config": str(config_path.resolve()),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(seconds, 1),
        "files_produced": produced,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True))
    temporary.replace(path)
    return path
