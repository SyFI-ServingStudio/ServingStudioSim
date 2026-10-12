# syntax=docker/dockerfile:1.7
#
# `vllm_env`: the alignment vLLM fork at the superproject's gitlink
# (`servingstudio-alignment`, alignment/profiler/README.md), installed over the
# precompiled CUDA 13.0 wheel of its upstream base -- the fork changes only
# Python. That wheel carries vLLM's FlashMLA and DeepGEMM builds. Built and tagged
# by `profiling/container/images.py`, which resolves the revisions below.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG VLLM_REVISION
ARG VLLM_BASE_REVISION
ARG VLLM_VERSION
ARG VLLM_WHEEL_VARIANT=cu130

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PYTHONPATH=/opt/vibesim \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home \
    PATH=/opt/vibesim/alignment/profiler/vllm/.venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/vibesim
COPY alignment/profiler/vllm /opt/vibesim/alignment/profiler/vllm

RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "${VLLM_BASE_REVISION}" \
    && uv venv --python 3.12 alignment/profiler/vllm/.venv \
    && cd alignment/profiler/vllm \
    && VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_BASE_REVISION}" \
       VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_WHEEL_VARIANT}" \
       VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
       uv pip install --python .venv/bin/python -e . \
    && uv pip install --python .venv/bin/python nvtx numpy nvidia-ml-py \
    && uv pip check --python .venv/bin/python

# Workers run with --network none, so bake every FlashInfer cubin into the
# image. The default directory is under HOME, which the /cache bind mount
# shadows at run time. Workers create lock files next to the cubins, so the tree
# stays writable.
ENV FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubins
RUN python -m flashinfer --download-cubin \
    && test -f "${FLASHINFER_CUBIN_DIR}"/*/fmha/trtllm-gen/checksums.txt \
    && FLASHINFER_CUDA_ARCH_LIST=10.0 python -c \
       "from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module; gen_trtllm_gen_fused_moe_sm100_module()" \
    && chmod -R a+rwX "${FLASHINFER_CUBIN_DIR}"

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
ARG VIBESIM_PROFILE_ENVS
RUN python -c "import importlib.metadata, torch; assert torch.version.cuda == '13.0'; print(importlib.metadata.version('vllm'))" \
    && test -f alignment/profiler/vllm/vllm/_C_stable_libtorch.abi3.so \
    && python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
# What this image is, for a process running inside it
# (`VIBESIM_PROFILE_RUNTIME=in-image`, profiling/exec/env.py).
RUN /opt/vibesim/alignment/profiler/vllm/.venv/bin/python -c "import json, os; json.dump({'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/vibesim/alignment/profiler/vllm/.venv/bin/python', 'python_paths': [p for p in os.environ['PYTHONPATH'].split(':') if p and p != '/opt/vibesim']}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim vLLM kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.vllm.revision="${VLLM_REVISION}" \
      ai.vibesim.vllm.base-revision="${VLLM_BASE_REVISION}" \
      ai.vibesim.vllm.wheel-variant="${VLLM_WHEEL_VARIANT}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/vibesim/alignment/profiler/vllm/.venv/bin/python"

ENTRYPOINT ["/opt/vibesim/alignment/profiler/vllm/.venv/bin/python", "-m", "profiling.exec.local_worker"]
