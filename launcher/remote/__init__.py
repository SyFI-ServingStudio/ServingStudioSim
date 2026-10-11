"""Run a launcher command on another machine (`--remote NAME`).

`run.py` is the sending side and the entry point; `node.py` runs on the remote.
The remote is described in remote.yaml (`config.py`, `remote.example.yaml`).
Simulation and timing-predict code know nothing of this: the remote runs the
same command, unchanged but for ``--no-gpu --no-plot``.
"""

from __future__ import annotations


def run_remote(name, argv, options) -> int:
    from .run import run

    return run(name, argv, options)
