"""Concrete config tree → Rust binary argv (new-interface-design §13.1).

Each run-like subcommand (`run` / `build-cache-only` / `dry-run`) takes a path
to ONE structured config file. This module strips the launcher-internal keys,
writes the concrete tree to `config_path` as block-style YAML, and returns
`[binary, subcommand, config_path]` — the single place a preset becomes a command
line, so skills / CLI / tests share the exact spelling (INV-1).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def strip_internal(config: dict) -> dict:
    """Drop launcher-internal `_`-prefixed keys (e.g. `_sweep_labels`, `_env`)
    so only the pure config tree the Rust binary parses is written."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


def write_config(config: dict, config_path: str | Path) -> Path:
    """Write the stripped config tree to `config_path` as block-style YAML — one
    param per line, no inline `{...}` flow maps. The binary's `load_config` reads
    `.yaml` via serde_yaml (YAML is a JSON superset, so nothing is lost). Returns
    the path."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(strip_internal(config), default_flow_style=False, sort_keys=False)
    )
    return path


def build_cli_command(
    config: dict, binary: str | Path, config_path: str | Path, subcommand: str = "run"
) -> list[str]:
    """Write `config` to `config_path` (block-style YAML) and emit `[binary,
    subcommand, path]`. `subcommand` is `run` (sim), `build-cache-only` (cache
    prebuild), or `dry-run` (coverage probe) — all take the same one-config-file
    surface."""
    path = write_config(config, config_path)
    return [str(binary), subcommand, str(path)]
