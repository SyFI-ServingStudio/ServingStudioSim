# syntax=docker/dockerfile:1.7
#
# `default_env` and the envs that share its interpreter (`flashinfer_pip_env`,
# `causal_conv1d_env`): the project's own `profiling` dependency group --
# Torch 2.10 cu128, FlashInfer 0.6, Triton, NVSHMEM, DeepGEMM -- installed from
# uv.lock exactly as `just sync` installs it on a host. Built and tagged by
# `profiling/container/images.py`; see profiling/container/README.md.

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG CAUSAL_CONV1D_WHEEL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    DG_USE_LOCAL_VERSION=0 \
    TORCH_CUDA_ARCH_LIST="9.0;10.0" \
    PYTHONPATH=/opt/vibesim:/opt/causal_conv1d \
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

WORKDIR /opt/vibesim
COPY pyproject.toml uv.lock .python-version /opt/vibesim/

# DeepGEMM builds without isolation against the env's Torch, so everything
# else goes in first -- the same two steps as `just sync`.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group profiling --no-install-project \
        --no-install-package deep-gemm \
    && uv sync --frozen --no-default-groups --group profiling --no-install-project \
    && uv pip install --python /opt/venv/bin/python --no-deps \
        --target /opt/causal_conv1d "${CAUSAL_CONV1D_WHEEL}"

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

RUN python -c "import torch, flashinfer, deep_gemm, causal_conv1d; assert torch.version.cuda == '12.8'; print(torch.__version__, flashinfer.__version__)" \
    && python -m profiling list --json >/dev/null

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
LABEL org.opencontainers.image.title="VibeSim default kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.profile-envs="default_env flashinfer_pip_env causal_conv1d_env" \
      ai.vibesim.python="/opt/venv/bin/python"

ENTRYPOINT ["/opt/venv/bin/python", "-m", "profiling.exec.local_worker"]
