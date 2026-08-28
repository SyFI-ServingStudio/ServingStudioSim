"""Reusable alignment campaigns: a generic engine over declarative packs.

An alignment round has always needed the same five things — generate the
workload matrix, run each phase across every case, label the kernel inventory,
read the numbers out of the Analyzer reports, and judge them. Each round so far
grew its own copy of all five inside a log directory that is not tracked, so the
next round started from nothing and the previous round's accepted numbers had
nothing to be compared against.

This package is the engine; a **pack** under `presets/alignment/<pack>/` is the
data. GLM-5.2 NVFP4 on B200 is the first pack. Adding a model means adding a
pack; adding DP+EP means adding a `variant` and some cases to an existing one.
Neither requires touching this code.

Layout:

    pack.py      the declarative schema — cases, variants, host profiles
    render.py    (case, variant, host) -> a runnable case directory
    check.py     the CPU self-check, shared by the CLI and the pytest gate
    execute.py   readiness, planning, and one-phase-across-all-cases execution
    label.py     the rule set, applied to a fixpoint
    metrics.py   the formula table and reading it off Analyzer reports
    extract.py   run directories -> one metrics document
    compare.py   tolerance judgement, golden drift, and recording
    cli.py       `python -m launcher alignment-campaign <verb>`
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
