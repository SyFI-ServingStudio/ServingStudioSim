# syntax=docker/dockerfile:1.7
#
# `sgl_kernel_env`: the flashinfer_attn_decode `sgl_fa3` backend. sgl-kernel
# 0.3.17's prebuilt extension is ABI-bound to Torch 2.8 (cu128), which neither
# the project stack nor the SGLang image holds, so this image is those two
# packages, installed exactly as frozen in requirements/sgl_kernel.txt. Built
# and tagged by `profiling/container/images.py`.

ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG CUDA_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/opt/vibesim \
    TORCH_CUDA_ARCH_LIST="9.0" \
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
        libnuma1 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# `--torch-backend cu128` sends the Torch-family pins to the PyTorch index, as
# when the file was resolved.
COPY profiling/container/requirements/sgl_kernel.txt /opt/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python 3.12 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --torch-backend cu128 --no-deps \
        -r /opt/requirements.txt \
    && uv pip check --python /opt/venv/bin/python

WORKDIR /opt/vibesim
COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
# `docker build` has no GPU and so no driver: sgl_kernel's extension links
# libcuda.so.1, which the toolkit ships only as a link stub (its other system
# dependency, libnuma.so.1, is installed above). The stub is put on the loader
# path for these import checks alone; at run time the container runtime mounts
# the real driver.
ARG VIBESIM_PROFILE_ENVS
RUN mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && LD_LIBRARY_PATH=/tmp/cuda-stub python -c "import torch, sgl_kernel.flash_attn; assert torch.version.cuda == '12.8'; print(torch.__version__)" \
    && LD_LIBRARY_PATH=/tmp/cuda-stub python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS} \
    && rm -rf /tmp/cuda-stub

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
# What this image is, for a process running inside it
# (`VIBESIM_PROFILE_RUNTIME=in-image`, profiling/exec/env.py).
RUN /opt/venv/bin/python -c "import json, os; json.dump({'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/venv/bin/python', 'python_paths': [p for p in os.environ['PYTHONPATH'].split(':') if p and p != '/opt/vibesim']}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim sgl-kernel profiler" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.cuda.version="${CUDA_VERSION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/venv/bin/python"

ENTRYPOINT ["/opt/venv/bin/python", "-m", "profiling.exec.local_worker"]
