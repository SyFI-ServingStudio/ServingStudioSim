# syntax=docker/dockerfile:1.7
#
# `flashinfer_kda_env`: the kda_chunk_prefill flashinfer_* backends. FlashInfer's
# Blackwell KDA prefill kernels (TIRx, CuTe DSL persistent) are only on its main
# branch and need Torch cu130, CuTe DSL >= 4.7 and a CUDA-enabled TVM with TIRx,
# which neither the project stack nor vllm_env can take. Installed exactly as
# frozen in requirements/flashinfer_kda.txt; TIRx JIT-compiles with this image's
# CUDA 13 nvcc. Built and tagged by `profiling/container/images.py`.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/opt/vibesim \
    CUDA_PATH=/usr/local/cuda \
    FLASHINFER_WORKSPACE_BASE=/cache/home \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

COPY profiling/container/requirements/flashinfer_kda.txt /opt/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-deps -r /opt/requirements.txt \
    && uv pip check --python /opt/venv/bin/python

WORKDIR /opt/vibesim
COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
ARG VIBESIM_PROFILE_ENVS
RUN python -c "import torch, flashinfer, tvm, tirx_kernels.tirx_lite; assert torch.version.cuda == '13.0'; print(torch.__version__, flashinfer.__version__)" \
    && python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
# What this image is, for a process running inside it
# (`VIBESIM_PROFILE_RUNTIME=in-image`, profiling/exec/env.py).
RUN /opt/venv/bin/python -c "import json, os; json.dump({'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/venv/bin/python', 'python_paths': [p for p in os.environ['PYTHONPATH'].split(':') if p and p != '/opt/vibesim']}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim FlashInfer KDA kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/venv/bin/python"

ENTRYPOINT ["/opt/venv/bin/python", "-m", "profiling.exec.local_worker"]
