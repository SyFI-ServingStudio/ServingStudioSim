"""Run a launcher command on another machine (`--remote NAME`)."""

from __future__ import annotations

import sys


def run_remote(name, argv, options) -> int:
    print("python -m launcher: --remote is not implemented yet", file=sys.stderr)
    return 2
