"""Alignment-local artifact utilities — `python -m alignment <command>`.

    python -m alignment parse   --sqlite T.sqlite --metrics M.jsonl \
                                --iteration-start N --iteration-end M
    python -m alignment overlap --sqlite T.sqlite --metrics M.jsonl \
                                --iteration-start N --iteration-end M
    python -m alignment ranges  T.sqlite --range-prefix framework.phase.
    python -m alignment label   {coverage,walk,slots,check,apply} ...

Launching belongs to `python -m launcher alignment
{sim,profile,timing-predict,analyze}`. Configs may be YAML or JSON. `parse` is
the nsys ingestion boundary. Comparison, statistics, and rendering belong to
the repository-level analyzer. The GPU-time duty-cycle multiplier is derived by
the analyzer's kernel-align pass (Σ measured_gpu_cycle_ms / Σ measured_ms), not
a standalone command. `label` holds the tools for the one step that is a
person's judgement rather than a command: mapping measured kernels onto modelled
operations.
"""

from __future__ import annotations

import sys

from .labeling import cli as labeling_cli
from .nsys import evidence as nsys_evidence
from .nsys import overlap as nsys_overlap
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

    if cmd == "ranges":
        return nsys_evidence.main(rest)

    if cmd == "overlap":
        return nsys_overlap.main(rest)

    if cmd == "label":
        return labeling_cli.main(rest)

    print(f"unknown subcommand: {cmd!r}")
    return _usage()


if __name__ == "__main__":
    raise SystemExit(main())
