"""``python -m launcher mstar-experiment`` — shared manifest harness (POC).

Requires ``MSTAR_EXPERIMENT_LAB`` pointing at ``mstar-sim-tier-a-qwen3tts`` (or a
copy on the host). Does not import M* or run sim logic here; delegates to the lab.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    lab = Path(
        os.environ.get(
            "MSTAR_EXPERIMENT_LAB",
            Path.home() / "mstar-sim-tier-a-qwen3tts",
        )
    ).expanduser()
    if not lab.is_dir():
        sys.exit(
            f"MSTAR_EXPERIMENT_LAB not found: {lab}\n"
            "Set MSTAR_EXPERIMENT_LAB to the mstar-sim-tier-a-qwen3tts checkout."
        )
    sys.path.insert(0, str(lab))

    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: python -m launcher mstar-experiment run-experiment "
            "--manifest PATH [--mode full|simulate|validate|dry-run|run|full]\n"
            "       python -m launcher mstar-experiment run-campaign "
            "--campaign PATH [--skip-run]\n"
            "       python -m launcher mstar-experiment evaluate "
            "--report PATH --acceptance PATH\n"
            f"lab: {lab}"
        )
        return 0

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "run-experiment":
        from run_experiment import main as run_experiment_main

        old = sys.argv
        sys.argv = ["run_experiment", *rest]
        try:
            return run_experiment_main()
        finally:
            sys.argv = old
    if cmd == "run-campaign":
        from run_campaign import main as run_campaign_main

        # run_campaign uses argparse on sys.argv; fake it
        old = sys.argv
        sys.argv = ["run_campaign", *rest]
        try:
            return run_campaign_main()
        finally:
            sys.argv = old
    if cmd == "evaluate":
        from evaluate_acceptance import main as eval_main

        old = sys.argv
        sys.argv = ["evaluate_acceptance", *rest]
        try:
            return eval_main()
        finally:
            sys.argv = old

    sys.exit(f"unknown mstar-experiment subcommand: {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
