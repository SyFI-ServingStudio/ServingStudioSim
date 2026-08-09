"""In-process concurrency limits; cross-process exclusion belongs to leases."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class ResourceScheduler:
    """Own independent concurrency budgets for simulation and analysis stages."""

    simulation_parallelism: int
    analysis_parallelism: int | None = None
    _simulation: asyncio.Semaphore = field(init=False, repr=False)
    _analysis: asyncio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.simulation_parallelism <= 0:
            raise ValueError("simulation_parallelism must be positive")
        analysis_parallelism = self.analysis_parallelism or self.simulation_parallelism
        if analysis_parallelism <= 0:
            raise ValueError("analysis_parallelism must be positive")
        self._simulation = asyncio.Semaphore(self.simulation_parallelism)
        self._analysis = asyncio.Semaphore(analysis_parallelism)

    @asynccontextmanager
    async def simulation_slot(self) -> AsyncIterator[None]:
        async with self._simulation:
            yield

    @asynccontextmanager
    async def analysis_slot(self) -> AsyncIterator[None]:
        async with self._analysis:
            yield
