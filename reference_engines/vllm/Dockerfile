# syntax=docker/dockerfile:1.7
#
# The vLLM reference engine: the official image of BASE's upstream commit with
# patches/ applied, used by both L1 kernel workers (`vllm_env`) and alignment
# captures. Built and tagged by `profiling/container/images.py`, which passes
# the official image pinned by this platform's digest as BASE_IMAGE.
#
# Layers go from what changes least to what changes most: the official image,
# system tools, the capture interpreter, the req-frontend binary, then the
# patches and the Sim code. A patch-only or Sim-only change reuses everything
# above them. The official image already carries vLLM's wheel at that commit
# (system Python, /usr/local/lib/python3.12/dist-packages), FlashInfer with
# every cubin and its AOT JIT cache, and nvtx.

ARG BASE_IMAGE

# req-frontend on the base's own userland, so the binary links the glibc it
# runs against. OpenSSL is vendored by the crate, hence perl.
FROM ${BASE_IMAGE} AS req-frontend
ARG RUST_TOOLCHAIN=1.92.0
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
RUN apt-get update \
    && apt-get install -y --no-install-recommends perl pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path
COPY alignment/load_generator/req-frontend /src/req-frontend
RUN cd /src/req-frontend \
    && /opt/cargo/bin/cargo build --release --locked --bin session_runner \
    && ./target/release/session_runner --help > /dev/null

FROM ${BASE_IMAGE}

# Nsight Systems at the version the host captures use, from the CUDA apt
# repository the base already configures (/opt/nvidia/nsight-systems/2025.5.2);
# gdb and ps for a watchdog's hang stacks.
ARG NSYS_PACKAGE=nsight-systems-2025.5.2
ARG NSYS_VERSION=2025.5.2.266-255236693005v0
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        "${NSYS_PACKAGE}=${NSYS_VERSION}" gdb procps \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/nvidia/nsight-systems/2025.5.2/bin/nsys --version

# The capture runner's interpreter, separate from the engine's, with py-spy
# for `py-spy dump --native` of the engine's processes.
ARG PY_SPY_VERSION=0.4.2
COPY reference_engines/capture-requirements.txt /opt/vibesim-capture/requirements.txt
RUN uv venv --python /usr/bin/python3.12 /opt/vibesim-capture/.venv \
    && uv pip install --no-cache --python /opt/vibesim-capture/.venv/bin/python \
        -r /opt/vibesim-capture/requirements.txt "py-spy==${PY_SPY_VERSION}" \
    && uv pip check --python /opt/vibesim-capture/.venv/bin/python

# The official image is vLLM's CI image, which turns on its GPU-sync checker
# (wrappers on every tensor copy) for tests. A release image does not, and a
# measurement must not: drop it at interpreter start, since a Dockerfile cannot
# unset an inherited ENV and vLLM rejects an empty value. FlashInfer creates lock
# files beside its cubins as whichever user runs the worker, so their
# directories are writable (the files themselves are not copied).
RUN python3 -c "import site; open(site.getsitepackages()[0] + '/vibesim_drop_ci_env.pth', 'w').write('import os; os.environ.pop(\"VLLM_GPU_SYNC_CHECK\", None)\n')" \
    && python3 -c "import os; assert 'VLLM_GPU_SYNC_CHECK' not in os.environ" \
    && find "$(python3 -c 'import flashinfer_cubin; print(flashinfer_cubin.get_cubin_dir())')" \
        -type d -exec chmod a+rwx {} + \
    && python3 -c "import importlib.metadata as m; assert m.version('nvtx') == '0.2.15', m.version('nvtx')"

ENV PYTHONPATH=/opt/vibesim \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home
WORKDIR /opt/vibesim

# Where `alignment/load_generator/runner.py` looks for it.
COPY --from=req-frontend /src/req-frontend/target/release/session_runner \
    /opt/vibesim/alignment/load_generator/req-frontend/target/release/session_runner

# The patches, onto vLLM as the official image installed it. Only the package's
# own files apply (the fork's tests and notes are not installed); a patch that
# does not apply means the image is not BASE's commit.
COPY reference_engines/vllm/patches /opt/vibesim/reference_engines/vllm/patches
RUN cd /usr/local/lib/python3.12/dist-packages \
    && git apply --include='vllm/*' /opt/vibesim/reference_engines/vllm/patches/*.patch \
    && python3 -c "import vllm.v1.worker.alignment_trace"

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
ARG VIBESIM_PROFILE_ENVS
RUN python3 -c "import torch; assert torch.version.cuda == '13.0'" \
    && python3 -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the manifest and labels.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
ARG VIBESIM_BASE_IMAGE=unknown
ARG VIBESIM_UPSTREAM_COMMIT=unknown
ARG VIBESIM_REQ_FRONTEND_REVISION=unknown
ENV VIBESIM_CAPTURE_RUNTIME=in-image \
    VIBESIM_CAPTURE_PYTHON=/opt/vibesim-capture/.venv/bin/python
# What this image is, for a process running inside it (profiling/exec/env.py,
# alignment/profiler/capture_facts.py).
RUN python3 -c "import json; json.dump({'image': 'vllm', 'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/usr/bin/python3', 'python_paths': [], 'capture_python': '/opt/vibesim-capture/.venv/bin/python', 'nsys': '/opt/nvidia/nsight-systems/2025.5.2/bin/nsys', 'base_image': '${VIBESIM_BASE_IMAGE}', 'upstream_commit': '${VIBESIM_UPSTREAM_COMMIT}', 'req_frontend_revision': '${VIBESIM_REQ_FRONTEND_REVISION}'}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="ServingStudio vLLM reference engine" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.base-image="${VIBESIM_BASE_IMAGE}" \
      ai.vibesim.upstream-commit="${VIBESIM_UPSTREAM_COMMIT}" \
      ai.vibesim.req-frontend.revision="${VIBESIM_REQ_FRONTEND_REVISION}" \
      ai.vibesim.nsys.version="${NSYS_VERSION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/usr/bin/python3"

# Callers name the interpreter and module (`--entrypoint`): an L1 worker runs
# `profiling.exec.local_worker` on /usr/bin/python3, a capture runs `launcher`
# on the capture interpreter.
ENTRYPOINT []
CMD []
