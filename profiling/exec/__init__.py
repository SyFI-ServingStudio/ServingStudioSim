"""Execution backend selection for L1 profiling."""

from __future__ import annotations

from profiling.exec.env import (
    ENV_REGISTRY,
    ProfileEnv,
    register_profile_env,
    resolve_profile_env,
)
from profiling.exec.local import LocalGpuPool
from profiling.exec.pool import (
    ChunkResult,
    GpuChunk,
    GpuPool,
)
from profiling.exec.remote import RemoteGpuPool

_default_pool: GpuPool | None = None


def set_default_pool(pool: GpuPool | None) -> None:
    global _default_pool
    _default_pool = pool


def get_default_pool() -> GpuPool:
    return _default_pool or LocalGpuPool()


__all__ = [
    "ChunkResult",
    "ENV_REGISTRY",
    "GpuChunk",
    "GpuPool",
    "LocalGpuPool",
    "ProfileEnv",
    "RemoteGpuPool",
    "get_default_pool",
    "register_profile_env",
    "resolve_profile_env",
    "set_default_pool",
]
