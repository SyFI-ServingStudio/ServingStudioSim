"""Param schema operations: validate, normalize, sweep-expand, and build the
Rust binary's argv.

This is the **single source of param truth** on the Python side (design §1.4
INV-1): every caller — skills, CLI, tests — routes preset dicts through here
rather than rolling their own argv. Per-param *data* is Rust-authoritative
(loaded via `loader.load_schema`); this package holds only the *logic* that
operates on it, split by stage:

    loader.py             read the Rust-authoritative `deployment_schema.json`
                          (the `Registry` + structural tree walk)
    expr.py               sandboxed `derived` / `constraint` evaluation
    validate.py           static preset validation (tree walk + V1–V6)
    expand.py             sweep-expand (${name}) → normalize → log_dir template
    argv.py               concrete config tree → config file + Rust binary argv

This module re-exports the stable public surface so callers keep importing from
`launcher.schema`.
"""

from __future__ import annotations

from .argv import build_cli_command, strip_internal, write_config
from .expand import (
    _format_log_dir,
    expand_sweep_params,
    normalize_params,
    validate_distinct_configs,
    validate_unique_log_dirs,
)
from .loader import log_dir_of
from .validate import validate_expanded, validate_params

__all__ = [
    "validate_params",
    "validate_expanded",
    "normalize_params",
    "expand_sweep_params",
    "build_cli_command",
    "write_config",
    "strip_internal",
    "log_dir_of",
    "_format_log_dir",
    "validate_unique_log_dirs",
    "validate_distinct_configs",
]
