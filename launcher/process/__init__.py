"""Launcher-owned child-process lifecycle primitives.

Every production subprocess enters through :class:`ProcessSupervisor`.  The
package deliberately does not know simulator, analyzer, or cache semantics;
those belong to workflow stages and artifact contracts.
"""

from .spec import ProcessResult, ProcessSpec
from .supervisor import ProcessSupervisor

__all__ = ["ProcessResult", "ProcessSpec", "ProcessSupervisor"]
