# syntax=docker/dockerfile:1.7
#
# `sglang_env`: the alignment SGLang fork at the superproject's gitlink, installed
# editable as alignment/profiler/README.md "SGLang environment" installs it
# (Torch 2.13 cu130, sglang-kernel, FlashInfer 0.6.15.post1 cu13), plus the
# FlashInfer cubin and cu130 JIT-cache wheels the README's runtime contract pins,
# since workers run with --network none. Its Rust extensions need a Rust
# toolchain at build time. Built and tagged by `profiling/container/images.py`;
# `sglang.Dockerfile.dockerignore` lets the checkout into the build context.

ARG CUDA_VERSION=13.0.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION
ARG SGLANG_REVISION
ARG SGLANG_VERSION
ARG FLASHINFER_VERSION=0.6.15.post1

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PYTHONPATH=/opt/vibesim:/opt/vibesim/alignment/profiler/sglang/python \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home \
    PATH=/opt/sglang-venv/bin:/opt/cargo/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libnuma1 \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.92

WORKDIR /opt/vibesim
COPY alignment/profiler/sglang /opt/vibesim/alignment/profiler/sglang

RUN --mount=type=cache,target=/root/.cache/uv \
    test -n "${SGLANG_VERSION}" \
    && uv venv --python 3.12 /opt/sglang-venv \
    && cd alignment/profiler/sglang/python \
    && SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_VERSION}" \
       uv pip install --python /opt/sglang-venv/bin/python \
           --extra-index-url https://download.pytorch.org/whl/cu130 \
           --index-strategy unsafe-best-match -e . \
    && uv pip install --python /opt/sglang-venv/bin/python \
        nvtx==0.2.16 numpy nvidia-ml-py \
    && uv pip install --python /opt/sglang-venv/bin/python --no-deps \
        --index-url https://flashinfer.ai/whl "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && uv pip install --python /opt/sglang-venv/bin/python --no-deps \
        --index-url https://flashinfer.ai/whl/cu130 "flashinfer-jit-cache==${FLASHINFER_VERSION}+cu130" \
    && uv pip check --python /opt/sglang-venv/bin/python \
    && rm -rf /opt/vibesim/alignment/profiler/sglang/rust/target

# FlashInfer reads its cubins from the flashinfer-cubin package and creates lock
# files next to them, as the worker's user; keep that tree writable.
RUN chmod -R a+rwX /opt/sglang-venv/lib/python3.12/site-packages/flashinfer_cubin/cubins

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
# sgl_kernel picks its compiled ops by the GPU's compute capability at import,
# and a build has no GPU; the first profiled sglang_env kernel imports it.
ARG VIBESIM_PROFILE_ENVS
RUN python -c "import torch, sglang, flashinfer; assert torch.version.cuda == '13.0'; print(torch.__version__, sglang.__version__, flashinfer.__version__)" \
    && python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
# What this image is, for a process running inside it
# (`VIBESIM_PROFILE_RUNTIME=in-image`, profiling/exec/env.py).
RUN /opt/sglang-venv/bin/python -c "import json, os; json.dump({'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/sglang-venv/bin/python', 'python_paths': [p for p in os.environ['PYTHONPATH'].split(':') if p and p != '/opt/vibesim']}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim SGLang kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.sglang.revision="${SGLANG_REVISION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/sglang-venv/bin/python"

ENTRYPOINT ["/opt/sglang-venv/bin/python", "-m", "profiling.exec.local_worker"]
