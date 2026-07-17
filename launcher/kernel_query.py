"""Run the simulator's existing ``kernel-query`` under launcher-owned PyO3 env.

The Rust Analyzer UI service uses this tiny transport adapter instead of
reconstructing PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH. Kernel semantics remain in
``simulator kernel-query``; this module only applies the same environment as all
other launcher-owned simulator subprocesses and forwards stdin/stdout/stderr.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .exec import _build_subprocess_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m launcher.kernel_query")
    parser.add_argument("--simulator", required=True)
    args = parser.parse_args(argv)
    completed = subprocess.run(
        [args.simulator, "kernel-query"],
        input=sys.stdin.buffer.read(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_build_subprocess_env(),
    )
    sys.stdout.buffer.write(completed.stdout)
    sys.stderr.buffer.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
