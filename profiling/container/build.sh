#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
image=${VIBESIM_PROFILER_IMAGE:-vibesim-profiler-vllm:cu130-3f667d7e}
expected_vllm_revision=3f667d7eb4572583f6ead3b677a55086bd2ecf3b
vllm_revision=$(git -C "$repo_root/alignment/profiler/vllm" rev-parse HEAD)

if [[ "$vllm_revision" != "$expected_vllm_revision" ]]; then
    echo "expected vLLM $expected_vllm_revision, got $vllm_revision" >&2
    exit 1
fi

source_state=clean
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
    source_state=dirty
    if [[ ${VIBESIM_ALLOW_DIRTY_PROFILE_IMAGE:-0} != 1 ]]; then
        echo "refusing to build a versioned profiler image from a dirty worktree" >&2
        echo "set VIBESIM_ALLOW_DIRTY_PROFILE_IMAGE=1 for a development-only image" >&2
        exit 1
    fi
fi

docker build \
    --file "$repo_root/profiling/container/Dockerfile" \
    --build-arg "VIBESIM_REVISION=$(git -C "$repo_root" rev-parse HEAD)" \
    --build-arg "VIBESIM_SOURCE_STATE=$source_state" \
    --build-arg "VLLM_REVISION=$vllm_revision" \
    --tag "$image" \
    "$repo_root"
