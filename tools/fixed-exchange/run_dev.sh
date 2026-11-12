#!/bin/bash
# Run dev_check.py in the vLLM profiler image on the job's GPUs with the IMEX channel
# (fabric-handle cuMemCreate), as profiling/exec/local.py does for MNNVL backends.
# Usage (inside a Slurm GPU allocation): tools/fixed-exchange/run_dev.sh <dev_check args>
set -euo pipefail
TREE=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=${FIXED_EXCHANGE_IMAGE:-vibesim-profiler-vllm:fe6a3d9a6af96e67}
CACHE=${FIXED_EXCHANGE_CACHE:-$TREE/tmp/devcache}
mkdir -p "$CACHE/home"
USER_ARG=$(cd "$TREE" && uv run --no-sync python -c 'from profiling.exec.container_user import container_user; print(container_user())')
exec docker run --rm --network none --ipc host --gpus all \
  --device nvidia.com/imex-channel=0 --user "$USER_ARG" \
  --env HOME=/cache/home --env USER=vibesim --env LOGNAME=vibesim \
  --env PYTHONUNBUFFERED=1 ${FX_DEBUG_FLAGS:+--env FX_DEBUG_FLAGS=1} \
  --volume "$CACHE:/cache" --volume "$TREE:/work:ro" --volume "$TREE/tmp:/work/tmp" \
  --workdir /work --entrypoint /usr/bin/python3 "$IMAGE" ${FIXED_EXCHANGE_SCRIPT:-tools/fixed-exchange/dev_check.py} "$@"
