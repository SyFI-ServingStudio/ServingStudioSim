"""Run the simulator's existing ``kernel-query`` under launcher-owned PyO3 env.

The Rust Analyzer UI service uses this tiny transport adapter instead of
reconstructing PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH. Kernel semantics remain in
``simulator kernel-query``; this module only applies the same environment as all
other launcher-owned simulator subprocesses and forwards stdin/stdout/stderr.
"""

from __future__ import annotations

import argparse
import os

from .exec import _build_subprocess_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m launcher.kernel_query")
    parser.add_argument("--simulator", required=True)
    args = parser.parse_args(argv)
    # This transport adds no launcher stage of its own. Replacing the process
    # preserves stdin/stdout/stderr exactly and avoids creating an unsupervised
    # child solely to forward bytes.
    os.execve(
        args.simulator,
        [args.simulator, "kernel-query"],
        _build_subprocess_env(),
    )
    raise AssertionError("os.execve returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
