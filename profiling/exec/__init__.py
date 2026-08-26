"""Execution backend selection for L1 profiling."""

from __future__ import annotations

import os

from profiling.exec.env import (
    ENV_REGISTRY,
    ContainerProfileEnv,
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
    if _default_pool is not None:
        return _default_pool
    # Escape hatch: VIBESIM_PROFILE_GPUS=0,1,2,3 forces profiling onto an explicit
    # GPU set, bypassing the idle-GPU guard (find_idle_gpus). Use only when you
    # know those GPUs are yours to use — it will profile regardless of other jobs'
    # residency/util. Unset (the default) keeps the safe idle-GPU selection.
    forced = os.environ.get("VIBESIM_PROFILE_GPUS")
    if forced:
        gpus = [int(x) for x in forced.split(",") if x.strip()]
        if gpus:
            return LocalGpuPool(gpus=gpus)
    return LocalGpuPool()


__all__ = [
    "ChunkResult",
    "ContainerProfileEnv",
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
