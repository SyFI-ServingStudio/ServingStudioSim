"""Public serving process whose stock Neuron compiles are split into two regions.

Same workload, sampling and artifacts as ``vllm_forward_engine``. The compile
adapter is installed at import time when this module runs as ``-m`` main, so
vLLM's spawned engine-core and rank processes (which re-import the main module
as ``__mp_main__``) install it before they compile; forked children inherit it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from profiling.runners.neuron.vllm_region_partition import PARTITION_DIR_ENV, install

if __name__ in ("__main__", "__mp_main__"):
    install()


def main(root: Path, mode: str) -> None:
    from profiling.runners.neuron.vllm_forward_engine import main as engine_main

    # Inherited by every rank process; each compile writes its structural receipt here.
    os.environ[PARTITION_DIR_ENV] = str(root / f"partition-{mode}")
    engine_main(root, mode)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
