# syntax=docker/dockerfile:1.7
#
# The vLLM profiler image plus what an alignment capture runs beside the server
# (alignment/README.md, "In-image capture"): Nsight Systems at the version the
# host captures used, the req-frontend load generator at the superproject's
# gitlink, and an interpreter for the capture runner itself. Built and tagged by
# `profiling/container/images.py`, which passes the vLLM image's tag as
# BASE_IMAGE. The checkout's `alignment/` (minus its submodules), `launcher/`,
# `profiling/` and `gpu/` are overlaid read-only at run time
# (`alignment/image_runtime.py`), so no capture code is baked in.

ARG BASE_IMAGE

# req-frontend on the base image's own userland, so the binary links the glibc
# it runs against. OpenSSL is vendored by the crate (Cargo.toml), hence perl.
FROM ${BASE_IMAGE} AS req-frontend
ARG RUST_TOOLCHAIN=1.92.0
# The base image sets HOME=/cache/home (a run-time mount), so pin the toolchain's homes.
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

# The capture's `python_runtime` contract pins NVTX in the server venv
# (presets/alignment/*/campaign.yaml); a container runs with --network none,
# so the pinned version is installed here rather than at run time.
ARG NVTX_VERSION=0.2.15
RUN uv pip install --python /opt/vibesim/alignment/profiler/vllm/.venv/bin/python \
        "nvtx==${NVTX_VERSION}" \
    && uv pip check --python /opt/vibesim/alignment/profiler/vllm/.venv/bin/python

# Hang diagnostics: `py-spy dump --native` for the server's Python processes and
# gdb for native ones (nsys). A job's watchdog runs them through `docker exec`
# when the engine stops iterating; the container has no network to fetch them.
ARG PY_SPY_VERSION=0.4.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends gdb procps \
    && rm -rf /var/lib/apt/lists/* \
    && uv pip install --python /opt/vibesim-capture/.venv/bin/python "py-spy==${PY_SPY_VERSION}" \
    && /opt/vibesim-capture/.venv/bin/py-spy --version \
    && gdb --version | head -1

# Where `alignment/load_generator/runner.py` looks for it.
COPY --from=req-frontend /src/req-frontend/target/release/session_runner \
    /opt/vibesim/alignment/load_generator/req-frontend/target/release/session_runner

# Last, so a new commit only rewrites the labels, not the dependency layers.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
ARG VIBESIM_REQ_FRONTEND_REVISION=unknown
ENV VIBESIM_CAPTURE_RUNTIME=in-image \
    VIBESIM_CAPTURE_PYTHON=/opt/vibesim-capture/.venv/bin/python
RUN /opt/vibesim-capture/.venv/bin/python -c "import json; json.dump({'image': 'vllm-align', 'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': [], 'python': '/opt/vibesim/alignment/profiler/vllm/.venv/bin/python', 'capture_python': '/opt/vibesim-capture/.venv/bin/python', 'nsys': '/opt/nvidia/nsight-systems/2025.5.2/bin/nsys', 'req_frontend_revision': '${VIBESIM_REQ_FRONTEND_REVISION}', 'python_paths': []}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="VibeSim vLLM alignment capture" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.req-frontend.revision="${VIBESIM_REQ_FRONTEND_REVISION}" \
      ai.vibesim.nsys.version="${NSYS_VERSION}"

WORKDIR /opt/vibesim
ENTRYPOINT ["/opt/vibesim-capture/.venv/bin/python", "-m", "launcher"]
