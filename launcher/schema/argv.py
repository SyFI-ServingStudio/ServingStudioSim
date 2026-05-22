"""Concrete params → Rust binary argv (design §1.8.2).

The one place a preset becomes a command line, so skills / CLI / tests share
the exact flag spelling and bool/list conventions (INV-1) instead of each
rolling their own argv.
"""

from __future__ import annotations

from pathlib import Path

from .loader import CONTROL_KEYS


def _cli_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_cli_command(
    params: dict, binary: str | Path, subcommand: str = "run"
) -> list[str]:
    """Emit `[binary, <subcommand>, <deployment>, *flags]`. `subcommand` is
    `"run"` for a sim or `"build-cache-only"` for the cache prebuild (both take
    the same deployment + flags). Only schema params present in `params` are
    emitted; bools are presence flags; list params repeat."""
    deployment = params["deployment"]
    argv: list[str] = [str(binary), subcommand, deployment]
    for key, value in params.items():
        if key in CONTROL_KEYS or key.startswith("_") or key == "deployment":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(_cli_flag(key))
        elif isinstance(value, (list, tuple)):
            for elem in value:
                argv.extend([_cli_flag(key), str(elem)])
        else:
            argv.extend([_cli_flag(key), str(value)])
    return argv
