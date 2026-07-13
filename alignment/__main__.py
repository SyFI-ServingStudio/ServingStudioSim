"""Alignment-local artifact utilities — `python -m alignment <command>`.

    python -m alignment parse   --sqlite T.sqlite --metrics M.jsonl \
                                --iteration-start N --iteration-end M
    python -m alignment gpu-kernel-ratio --profile-dir profile/ \
                                --output gpu_kernel_ratio.json

Launching belongs to `python -m launcher alignment
{sim,profile,timing-predict,analyze}`. Configs may be YAML or JSON. `parse` is
the nsys ingestion boundary. Comparison, statistics, and rendering belong to
the repository-level analyzer.
"""

from __future__ import annotations

import sys

from .nsys import gpu_kernel_ratio
from .nsys import parse as nsys_parse


def _usage() -> int:
    print(__doc__)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _usage()
    cmd, rest = argv[0], argv[1:]

    if cmd == "parse":
        return nsys_parse.main(rest)
    if cmd == "gpu-kernel-ratio":
        return gpu_kernel_ratio.main(rest)

    print(f"unknown subcommand: {cmd!r}")
    return _usage()


if __name__ == "__main__":
    raise SystemExit(main())
