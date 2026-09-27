# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_SOURCE_STATE=unknown
ARG VLLM_REVISION=3f667d7eb4572583f6ead3b677a55086bd2ecf3b
ARG VLLM_BASE_REVISION=8369affa5428ca29a30f378d660fbffbad12e240
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
       VLLM_VERSION_OVERRIDE="0.28.1rc0+vibesim.${VLLM_REVISION}" \
       uv pip install --python .venv/bin/python -e . \
    && uv pip install --python .venv/bin/python nvtx \
    && uv pip check --python .venv/bin/python \
    && mkdir -p .venv/lib/cuda-compat

# FlashInfer 0.6.18 has no flashinfer-cubin wheel, and workers run with
# --network none, so bake every cubin into the image. The default directory is
# under HOME, which the /cache bind mount shadows at run time. Workers create
# lock files next to the cubins, so the tree stays writable.
ENV FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubins
RUN python -m flashinfer --download-cubin \
    && test -f "${FLASHINFER_CUBIN_DIR}"/*/fmha/trtllm-gen/checksums.txt \
    && FLASHINFER_CUDA_ARCH_LIST=10.0 python -c \
       "from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module; gen_trtllm_gen_fused_moe_sm100_module()" \
    && chmod -R a+rwX "${FLASHINFER_CUBIN_DIR}"

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

RUN python -c "import importlib.metadata, torch; assert torch.version.cuda == '13.0'; print(importlib.metadata.version('vllm'))" \
    && test -f alignment/profiler/vllm/vllm/_C_stable_libtorch.abi3.so \
    && python -m profiling list --json >/dev/null

ENTRYPOINT ["/opt/vibesim/alignment/profiler/vllm/.venv/bin/python", "-m", "profiling.exec.local_worker"]
