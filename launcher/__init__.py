"""MLSim launcher (L7-α) — Python orchestration for sim runs.

Public entry points for callers (skills, tests, researchers) that import the
launcher rather than shelling out to `python -m launcher`. Per design §1.4
INV-1, everyone constructing a run goes through `launcher.schema` rather than
rolling their own argv.
"""

from __future__ import annotations

from .schema import (
    build_cli_command,
    expand_sweep_params,
    normalize_params,
    validate_params,
)
from .schema.loader import Schema, SchemaNotFound, load_schema
from .sweep import run_single, run_sweep

__all__ = [
    "Schema",
    "SchemaNotFound",
    "load_schema",
    "validate_params",
    "normalize_params",
    "expand_sweep_params",
    "build_cli_command",
    "run_single",
    "run_sweep",
]
