"""Stage-2 outlier policy objects for L1b batch profiling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchOutlierPolicy:
    """Placeholder for the documented in-batch outlier policy.

    Concrete retry/sort behavior can grow here without changing runner or
    registry signatures.
    """

    enabled: bool = False
