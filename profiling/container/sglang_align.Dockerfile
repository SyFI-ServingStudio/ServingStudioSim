# syntax=docker/dockerfile:1.7
#
# The SGLang profiler image plus what an alignment capture runs beside the
# server, exactly as `vllm_align.Dockerfile` does for vLLM: Nsight Systems at the
# version the host captures used, the req-frontend load generator at the
# superproject's gitlink, and an interpreter for the capture runner itself.
# Built and tagged by `profiling/container/images.py`, which passes the SGLang
# image's tag as BASE_IMAGE. The checkout's `alignment/` (minus its submodules),
# `launcher/`, `profiling/` and `gpu/` are overlaid read-only at run time
# (`alignment/image_runtime.py`), so no capture code is baked in; the SGLang
# source the server imports (`alignment/profiler/sglang/python`) is a submodule
# and so stays the base image's.
#
# Nothing is added to the SGLang venv: the base installs the wheels an SGLang
# capture's `python_runtime` contract pins (presets/alignment/*sglang*/
# campaign.yaml: nvtx, the FlashInfer cubin and JIT-cache wheels). The runner
# verifies them before the server starts and, with --network none, refuses a
# missing one instead of installing it.

ARG BASE_IMAGE

# req-frontend on the base image's own userland, so the binary links the glibc
# it runs against. The base already has the Rust toolchain (CARGO_HOME=/opt/cargo)
# its SGLang build needed. OpenSSL is vendored by the crate, hence perl.
FROM ${BASE_IMAGE} AS req-frontend
RUN apt-get update \
    && apt-get install -y --no-install-recommends perl \
    && rm -rf /var/lib/apt/lists/*
COPY alignment/load_generator/req-frontend /src/req-frontend
RUN cd /src/req-frontend \
    && /opt/cargo/bin/cargo build --release --locked --bin session_runner \
    && ./target/release/session_runner --help > /dev/null

FROM ${BASE_IMAGE}

# Nsight Systems from NVIDIA's CUDA apt repository (already configured in the
# CUDA base image), pinned to the exact package version. It installs under
# /opt/nvidia/nsight-systems/2025.5.2, the same path as on the capture hosts.
ARG NSYS_PACKAGE=nsight-systems-2025.5.2
ARG NSYS_VERSION=2025.5.2.266-255236693005v0
RUN apt-get update \
    && apt-get install -y --no-install-recommends "${NSYS_PACKAGE}=${NSYS_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/nvidia/nsight-systems/2025.5.2/bin/nsys --version

# The capture runner's interpreter, separate from the server's venv.
COPY profiling/container/requirements/capture.txt /opt/vibesim-capture/requirements.txt
RUN uv venv --python 3.12 /opt/vibesim-capture/.venv \
    && uv pip install --python /opt/vibesim-capture/.venv/bin/python \
        -r /opt/vibesim-capture/requirements.txt \
    && uv pip check --python /opt/vibesim-capture/.venv/bin/python

# Where `alignment/load_generator/runner.py` looks for it.
COPY --from=req-frontend /src/req-frontend/target/release/session_runner \
    /opt/vibesim/alignment/load_generator/req-frontend/target/release/session_runner

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
ARG VIBESIM_REQ_FRONTEND_REVISION=unknown
ENV VIBESIM_CAPTURE_RUNTIME=in-image \
    VIBESIM_CAPTURE_PYTHON=/opt/vibesim-capture/.venv/bin/python
RUN /opt/vibesim-capture/.venv/bin/python -c "import json; json.dump({'image': 'sglang-align', 'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': [], 'python': '/opt/sglang-venv/bin/python', 'capture_python': '/opt/vibesim-capture/.venv/bin/python', 'nsys': '/opt/nvidia/nsight-systems/2025.5.2/bin/nsys', 'req_frontend_revision': '${VIBESIM_REQ_FRONTEND_REVISION}', 'python_paths': []}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim SGLang alignment capture" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.req-frontend.revision="${VIBESIM_REQ_FRONTEND_REVISION}" \
      ai.vibesim.nsys.version="${NSYS_VERSION}"

WORKDIR /opt/vibesim
ENTRYPOINT ["/opt/vibesim-capture/.venv/bin/python", "-m", "launcher"]
