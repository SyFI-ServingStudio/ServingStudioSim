"""Param schema operations: validate, normalize, sweep-expand, and build the
Rust binary's argv.

This is the **single source of param truth** on the Python side (design §1.4
INV-1): every caller — skills, CLI, tests — routes preset dicts through here
rather than rolling their own argv. Per-param *data* is Rust-authoritative
(loaded via `loader.load_schema`); this package holds only the *logic* that
operates on it, split by stage:

    loader.py             read the Rust-authoritative `deployment_schema.json`
    expr.py               sandboxed `derived` / `constraint` evaluation
    validate.py           static preset validation (V1–V6)
    expand.py             classify → sweep-expand → normalize → log_dir template
    argv.py               concrete params → Rust binary argv

This module re-exports the stable public surface so callers keep importing from
`launcher.schema`.
"""

from __future__ import annotations

from .argv import build_cli_command
from .expand import (
    _format_log_dir,
    expand_sweep_params,
    normalize_params,
    validate_unique_log_dirs,
)
from .validate import validate_params

__all__ = [
    "validate_params",
    "normalize_params",
    "expand_sweep_params",
    "build_cli_command",
    "_format_log_dir",
    "validate_unique_log_dirs",
]
