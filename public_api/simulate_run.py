"""``python -m public_api.simulate_run RUN_DIR BUILD_TYPE``: one queued simulation.

The service writes ``RUN_DIR/simulation.run.json`` (the concrete run config,
as the launcher's expansion leaves it) and ``RUN_DIR/simulation.preset.json``
(the config before expansion), then starts this in its own process so that it can stop the run at
the wall-clock limit. It runs the launcher's standard single run: cache check
(against profile.db, without a GPU), simulation, then analysis without plots.
It does not build the simulator: the service runs the binary it was started
with, as ``/predict`` does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from launcher.exec import _build_subprocess_env, binary_path
from launcher.schema.loader import schema_from_dict
from launcher.sweep import run_single
from public_api.simulate import RUN_CONFIG, RUN_PRESET


def main(argv: list[str]) -> int:
    run_dir, build_type = Path(argv[0]), argv[1]
    listed = subprocess.run(
        [str(binary_path(build_type)), "list-params"],
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
        check=True,
    )
    schema = schema_from_dict(json.loads(listed.stdout))
    config = json.loads((run_dir / RUN_CONFIG).read_text())
    preset = json.loads((run_dir / RUN_PRESET).read_text())
    ok = run_single(config, preset, schema, build_type, analyze=True, plot=False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
