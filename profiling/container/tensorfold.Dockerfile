# syntax=docker/dockerfile:1.7
#
# `tensorfold_env`: the TensorFold checkout (alignment/profiler/tensorfold) at the
# superproject's gitlink, installed editable on Torch 2.14.1 cu130 -- the stack the
# GB10 study serves from (perf_study envs/tf-dev). TensorFold's CUDA extensions
# JIT-compile on first use for the GPU present, so the image carries nvcc and no
# prebuilt kernels; workers keep the builds in the per-tag /cache. Built and
# tagged by `profiling/container/images.py`; `tensorfold.Dockerfile.dockerignore`
# lets the checkout into the build context.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG TENSORFOLD_REVISION
ARG TORCH_VERSION=2.14.1

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PYTHONPATH=/opt/vibesim:/opt/vibesim/alignment/profiler/tensorfold/src \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    MAX_JOBS=4 \
    HOME=/cache/home \
    PATH=/opt/tensorfold-venv/bin:${PATH}

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
COPY alignment/profiler/tensorfold /opt/vibesim/alignment/profiler/tensorfold

RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "${TENSORFOLD_REVISION}" \
    && uv venv --python 3.12 /opt/tensorfold-venv \
    && uv pip install --python /opt/tensorfold-venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match "torch==${TORCH_VERSION}" \
    && uv pip install --python /opt/tensorfold-venv/bin/python \
        -e alignment/profiler/tensorfold numpy nvidia-ml-py ninja
# No `uv pip check`: on aarch64 it rejects nvidia-cusparselt-cu13, whose wheel is
# tagged manylinux2014_sbsa (an NVIDIA packaging slip; Torch loads it fine). The
# import and self-check below are the gate.

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners). No GPU
# at build time, so no TensorFold extension compiles here.
ARG VIBESIM_PROFILE_ENVS
RUN python -c "import torch, triton, tensorfold; assert torch.version.cuda == '13.0'; print(torch.__version__, triton.__version__, tensorfold.__version__)" \
    && python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
# What this image is, for a process running inside it
# (`VIBESIM_PROFILE_RUNTIME=in-image`, profiling/exec/env.py).
RUN /opt/tensorfold-venv/bin/python -c "import json, os; json.dump({'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/tensorfold-venv/bin/python', 'python_paths': [p for p in os.environ['PYTHONPATH'].split(':') if p and p != '/opt/vibesim']}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim TensorFold kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.tensorfold.revision="${TENSORFOLD_REVISION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/tensorfold-venv/bin/python"

ENTRYPOINT ["/opt/tensorfold-venv/bin/python", "-m", "profiling.exec.local_worker"]
