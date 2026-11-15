# syntax=docker/dockerfile:1.7
#
# The DeepInfer reference engine (deepseek-ai/DeepInfer, NVIDIA SM100): BASE.yaml's
# upstream commit with patches/ applied, built from source. Built and tagged by
# `profiling/container/images.py` (`just ref build-image deepinfer`), which passes the
# CUDA image pinned by this platform's digest as BASE_IMAGE.
#
# DeepInfer publishes no image (its README builds from source: scripts/build.sh), so
# unlike vllm/ and sglang/ this starts from the CUDA devel image and builds the engine,
# following build.sh: requirements-nvidia.txt, NCCL 2.32.3, scripts/build_vendor.sh's
# per-package `setup.py bdist_wheel` for the four NVIDIA vendor operators, then
# scripts/build.sh's in-tree `setup.py build`. README: Python 3.12, GCC >= 13, CUDA 13;
# torch is the cu132 wheel. Devel, because DeepEP and DeepGEMM JIT-compile their kernels
# at run time (nvcc, libdw, nvidia-smi/ibstat probes).

ARG BASE_IMAGE

# The source: BASE.yaml's upstream commit, its NVIDIA vendor submodules at the commits it
# pins (checked against BASE.yaml's `submodules`), and patches/ on top. A patch that does
# not apply means the tree is not BASE.yaml's commit.
FROM ${BASE_IMAGE} AS source
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY reference_engines/deepinfer/BASE.yaml /pins/BASE.yaml
RUN set -eu; \
    repo="$(sed -n 's/^upstream_repo: //p' /pins/BASE.yaml)"; \
    commit="$(sed -n 's/^upstream_commit: //p' /pins/BASE.yaml)"; \
    git init --quiet /src && cd /src \
    && git fetch --quiet --depth=1 "$repo" "$commit" && git checkout --quiet --detach "$commit"; \
    sed -n '/^submodules:/,$p' /pins/BASE.yaml | sed -n 's/^  \([^:]*\): \([0-9a-f]\{40\}\)$/\1 \2/p' > /pins/submodules; \
    test -s /pins/submodules; \
    while read -r path pinned; do \
        git -c protocol.file.allow=never submodule update --init --recursive --depth 1 -- "$path"; \
        head="$(git -C "$path" rev-parse HEAD)"; \
        [ "$head" = "$pinned" ] || { echo "$path is at $head, BASE.yaml pins $pinned" >&2; exit 1; }; \
    done < /pins/submodules
COPY reference_engines/deepinfer/patches /pins/patches
RUN cd /src && git -c safe.directory='*' apply /pins/patches/*.patch && git status --short

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# Nsight Systems at the version the host captures use, from the CUDA apt repository the
# base configures; gdb/procps for hang stacks. pciutils: DeepSelect/FlashMLA setup.py
# import tests/kernelkit, which probes the platform with lspci. libdw-dev: DeepJIT
# (DeepEP/DeepGEMM) includes elfutils/libdwfl.h. infiniband-diags: DeepEP's bandwidth
# detection calls ibstat.
ARG NSYS_PACKAGE=nsight-systems-2025.5.2
ARG NSYS_VERSION=2025.5.2.266-255236693005v0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        git build-essential ninja-build numactl gdb procps curl ca-certificates \
        pciutils libdw-dev infiniband-diags \
        "${NSYS_PACKAGE}=${NSYS_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && gcc --version | head -1 \
    && /opt/nvidia/nsight-systems/2025.5.2/bin/nsys --version

ENV VIRTUAL_ENV=/opt/deepinfer-venv \
    PATH=/opt/deepinfer-venv/bin:${PATH} \
    DEEPINFER_ACCELERATOR=nvidia \
    TORCH_CUDA_ARCH_LIST=10.0a \
    MAX_JOBS=64
RUN python3.12 -m venv "${VIRTUAL_ENV}" && pip install --no-cache-dir -U pip setuptools wheel

WORKDIR /opt/deepinfer
# Dependencies first so a source-only change reuses this layer.
COPY --from=source /src/requirements.txt /src/requirements-dev.txt /src/requirements-nvidia.txt ./
RUN pip install --no-cache-dir -r requirements-nvidia.txt \
    && pip install --no-cache-dir --no-deps --force-reinstall nvidia-nccl-cu13==2.32.3 \
    && python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Vendor operators (scripts/build_vendor.sh, NVIDIA branch), one layer each.
# FlashMLA's post-build register-spill lint rejects the build on nvcc 13.2.1 and 13.3.1:
# only the SM100 head-64 sparse FP8 *decode* kernel (flash_fwd_splitkv_mla_fp8_sparse_kernel)
# spills (20 B stores / 28 B loads). FLASH_MLA_SKIP_REG_SPILL_CHECK is the lint's own
# opt-out; the compiled code is the same. Whether the prefill path launches that kernel is
# checked in the nsys capture (wt-dsv41-deepinfer notes.md).
RUN mkdir -p /opt/vendor-wheels
COPY --from=source /src/vendor/DeepSelect vendor/DeepSelect
RUN cd vendor/DeepSelect && DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
    python setup.py bdist_wheel --dist-dir /opt/vendor-wheels
COPY --from=source /src/vendor/FlashMLA vendor/FlashMLA
RUN cd vendor/FlashMLA && FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA FLASH_MLA_SKIP_REG_SPILL_CHECK=1 \
    python setup.py bdist_wheel --dist-dir /opt/vendor-wheels
COPY --from=source /src/vendor/DeepEP vendor/DeepEP
RUN cd vendor/DeepEP && python setup.py bdist_wheel --dist-dir /opt/vendor-wheels
COPY --from=source /src/vendor/DeepGEMM vendor/DeepGEMM
RUN cd vendor/DeepGEMM && DG_FORCE_BUILD=1 python setup.py bdist_wheel --dist-dir /opt/vendor-wheels
RUN pip install --no-cache-dir --no-deps --no-index --find-links /opt/vendor-wheels \
        deep_select flash_mla deep_ep deep_gemm \
    && ls -la /opt/vendor-wheels

# RDMA userspace (rdma-core): DeepEP's EngramBuffer (the Engram tables in CPU memory) runs on
# NCCL GIN's GDAKI backend, which needs libibverbs and the mlx5 provider; without them NCCL
# reports no GIN and the engine stops at model load. A container also needs the host's
# /dev/infiniband/uverbs* devices. ibverbs-utils (ibv_devinfo) checks them in a job.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libibverbs1 ibverbs-providers librdmacm1 ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# DeepInfer itself, patched, built in tree (scripts/build.sh build). No import check: the
# extensions link libcuda, which only a GPU job provides.
COPY --from=source /src /opt/deepinfer
RUN bash ./scripts/build.sh build && ls deepinfer/_so/

# Autotuning is a hard error unless the parent set VIBESIM_AUTOTUNE=allow, in every process
# of the engine's interpreter (reference_engines/autotune_guard.py, loaded by a .pth at
# interpreter start; reference_engines/README.md#autotuning).
COPY reference_engines/autotune_guard.py /opt/vibesim-build/vibesim_autotune_guard.py
RUN SITE="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && cp /opt/vibesim-build/vibesim_autotune_guard.py "$SITE/" \
    && echo "import vibesim_autotune_guard; vibesim_autotune_guard.install()" > "$SITE/vibesim_autotune_guard.pth" \
    && python -c "import sys; assert any(getattr(f, 'vibesim_autotune_guard', False) for f in sys.meta_path)"

# DeepInfer imports from its tree (build.sh builds in place); the Sim code beside it.
ENV PYTHONPATH=/opt/deepinfer:/opt/vibesim \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home
COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu
COPY reference_engines/autotune_guard.py /opt/vibesim/reference_engines/autotune_guard.py

# The envs images.py assigns this image (`selfcheck` loads their runners). None yet: the
# image runs DeepInfer's own benchmark until DeepInfer L1 kernels are registered.
ARG VIBESIM_PROFILE_ENVS
RUN if [ -n "${VIBESIM_PROFILE_ENVS}" ]; then \
        cd /opt/vibesim && python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}; \
    fi

# Last, so a new commit only rewrites the manifest and labels.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
ARG VIBESIM_BASE_IMAGE=unknown
ARG VIBESIM_UPSTREAM_COMMIT=unknown
ARG VIBESIM_CACHE_KEY=unknown
# What this image is, for a process running inside it (profiling/exec/env.py).
RUN python -c "import json; json.dump({'image': 'deepinfer', 'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/deepinfer-venv/bin/python', 'python_paths': [], 'capture_python': None, 'nsys': '/opt/nvidia/nsight-systems/2025.5.2/bin/nsys', 'base_image': '${VIBESIM_BASE_IMAGE}', 'upstream_commit': '${VIBESIM_UPSTREAM_COMMIT}'}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="ServingStudio DeepInfer reference engine" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.base-image="${VIBESIM_BASE_IMAGE}" \
      ai.vibesim.upstream-commit="${VIBESIM_UPSTREAM_COMMIT}" \
      ai.vibesim.cache-key="${VIBESIM_CACHE_KEY}" \
      ai.vibesim.nsys.version="${NSYS_VERSION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/deepinfer-venv/bin/python"

ENTRYPOINT []
CMD []
