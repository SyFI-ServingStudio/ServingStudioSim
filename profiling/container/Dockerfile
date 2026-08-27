# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_SOURCE_STATE=unknown
ARG VLLM_REVISION=0b8edfb56f5df9cd512ce4ff586e7ac2c8f31921
ARG VLLM_BASE_REVISION=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
ARG VLLM_WHEEL_VARIANT=cu130

LABEL org.opencontainers.image.title="VibeSim vLLM kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.source.state="${VIBESIM_SOURCE_STATE}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.vllm.revision="${VLLM_REVISION}" \
      ai.vibesim.vllm.base-revision="${VLLM_BASE_REVISION}" \
      ai.vibesim.vllm.wheel-variant="${VLLM_WHEEL_VARIANT}"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    PYTHONPATH=/opt/vibesim \
    VIBESIM_SOURCE_REVISION=${VIBESIM_REVISION} \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/opt/vibesim/alignment/profiler/vllm/.venv/bin:${PATH}

WORKDIR /opt/vibesim
COPY alignment/profiler/vllm /opt/vibesim/alignment/profiler/vllm

RUN UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv venv --python "${PYTHON_VERSION}" alignment/profiler/vllm/.venv \
    && cd alignment/profiler/vllm \
    && VLLM_USE_PRECOMPILED=1 \
       VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_BASE_REVISION}" \
       VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_WHEEL_VARIANT}" \
       VLLM_VERSION_OVERRIDE="0.23.0+vibesim.${VLLM_REVISION}" \
       uv pip install --python .venv/bin/python -e . \
    && uv pip install --python .venv/bin/python nvtx \
    && uv pip check --python .venv/bin/python \
    && mkdir -p .venv/lib/cuda-compat

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

RUN python -c "import importlib.metadata, torch; assert torch.version.cuda == '13.0'; print(importlib.metadata.version('vllm'))" \
    && test -f alignment/profiler/vllm/vllm/_C.abi3.so \
    && python -m profiling list --json >/dev/null

ENTRYPOINT ["/opt/vibesim/alignment/profiler/vllm/.venv/bin/python", "-m", "profiling.exec.local_worker"]
