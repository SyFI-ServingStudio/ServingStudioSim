"""Small JSON subprocess contract for analyzer model overview resources.

Usage: ``uv run python -m model.work.parameter_counts <config.json>``.
Parameter counts are architecture properties, so the zero-token workload is
intentional: it avoids introducing a serving-batch assumption into the overview.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .core import Workload
from .registry import load_model


def compute_parameter_counts(config_path: str | Path) -> dict[str, int | str]:
    params = load_model(config_path).label(Workload(matmul_tokens=0, head_positions=0)).params
    activated = params["activated"]
    return {
        "total": int(params["total"]),
        "active": int(activated["with_embed_head"]),
        "active_layers": int(activated["layers"]),
        "active_definition": "with_embed_head",
    }


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: python -m model.work.parameter_counts <config.json>")
    json.dump(compute_parameter_counts(arguments[0]), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
