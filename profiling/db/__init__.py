"""L1b profile-db: registry + schema + write-side scheduling.

Phase 0 ships only the abstract contracts:

- ``KernelArgs`` — frozen-dataclass base for per-kind args records.
- ``KernelKind`` — empty-bodied dispatch enum; variants land per-runner in
  Phase 1+.

Registry, Table, batch runner, and metadata wiring all land in Phase 1
alongside the first concrete L1 runner.
"""

from profiling.db.args import KernelArgs
from profiling.db.kind import KernelKind

__all__ = ["KernelArgs", "KernelKind"]
