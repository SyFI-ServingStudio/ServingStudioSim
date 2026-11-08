# syntax=docker/dockerfile:1.7
#
# The SGLang reference engine: the official image of BASE's upstream commit with
# patches/ applied, used by both L1 kernel workers (`sglang_env`) and alignment
# captures. Built and tagged by `profiling/container/images.py`, which passes
# the official image pinned by this platform's digest as BASE_IMAGE.
#
# Same layer order as the vLLM engine: official image, system tools, the capture
# interpreter, pip pins, the req-frontend binary, then the patches and the Sim
# code. The official image installs SGLang editable from its git checkout at
# /sgl-workspace/sglang into the /opt/sglang venv, with sglang-kernel and
# FlashInfer (every cubin and its AOT JIT cache), so the patches apply to that
# checkout.

ARG BASE_IMAGE

# req-frontend on the base's own userland; the base has a Rust toolchain.
FROM ${BASE_IMAGE} AS req-frontend
COPY alignment/load_generator/req-frontend /src/req-frontend
RUN cd /src/req-frontend \
    && /root/.cargo/bin/cargo build --release --locked --bin session_runner \
    && ./target/release/session_runner --help > /dev/null

FROM ${BASE_IMAGE}

# Nsight Systems at the version the host captures use, beside the base's own
# newer CLI (/opt/nvidia/nsight-systems/2025.5.2); the base has gdb.
ARG NSYS_PACKAGE=nsight-systems-2025.5.2
ARG NSYS_VERSION=2025.5.2.266-255236693005v0
RUN apt-get update \
    && apt-get install -y --no-install-recommends "${NSYS_PACKAGE}=${NSYS_VERSION}" procps \
    && rm -rf /var/lib/apt/lists/* \
    && /opt/nvidia/nsight-systems/2025.5.2/bin/nsys --version \
    && gdb --version | head -1

# The capture runner's interpreter, separate from the engine's, with py-spy.
ARG PY_SPY_VERSION=0.4.2
COPY reference_engines/capture-requirements.txt /opt/vibesim-capture/requirements.txt
RUN uv venv --python 3.12 /opt/vibesim-capture/.venv \
    && uv pip install --no-cache --python /opt/vibesim-capture/.venv/bin/python \
        -r /opt/vibesim-capture/requirements.txt "py-spy==${PY_SPY_VERSION}" \
    && uv pip check --python /opt/vibesim-capture/.venv/bin/python

# NVTX for the patches' ranges, which the base does not install; workers and
# captures run with --network none. FlashInfer creates lock files beside its
# cubins as whichever user runs the worker, so their directories are writable.
ARG NVTX_VERSION=0.2.16
RUN uv pip install --no-cache --python /opt/sglang/bin/python "nvtx==${NVTX_VERSION}" \
    && find "$(/opt/sglang/bin/python -c 'import flashinfer_cubin; print(flashinfer_cubin.get_cubin_dir())')" \
        -type d -exec chmod a+rwx {} +

# Captures run with FLASHINFER_DISABLE_JIT=1, so every FlashInfer module SGLang
# loads must be in the jit-cache. The 0.7.3 wheel lacks tinygemm2 (SGLang's BF16
# low-M GEMM: unquant._bf16_low_m_autotune -> flashinfer mm_bf16); compile it for
# the Blackwell datacenter providers (no GPU needed) and register it there.
ARG FLASHINFER_AOT_PROVIDERS=sm100a,sm103a
ARG FLASHINFER_AOT_MODULES="tinygemm2=flashinfer.gemm.routergemm:gen_tinygemm2_module"
COPY reference_engines/complete_flashinfer_aot.py /opt/vibesim-build/complete_flashinfer_aot.py
RUN /opt/sglang/bin/python /opt/vibesim-build/complete_flashinfer_aot.py \
        "${FLASHINFER_AOT_PROVIDERS}" ${FLASHINFER_AOT_MODULES}

# Autotuning is a hard error unless the parent set VIBESIM_AUTOTUNE=allow, in
# every process of the engine's interpreter (reference_engines/autotune_guard.py,
# loaded by a .pth at interpreter start; reference_engines/README.md#autotuning).
COPY reference_engines/autotune_guard.py /opt/vibesim-build/vibesim_autotune_guard.py
RUN SITE="$(/opt/sglang/bin/python -c 'import site; print(site.getsitepackages()[0])')" \
    && cp /opt/vibesim-build/vibesim_autotune_guard.py "$SITE/" \
    && echo "import vibesim_autotune_guard; vibesim_autotune_guard.install()" > "$SITE/vibesim_autotune_guard.pth" \
    && /opt/sglang/bin/python -c "import sys; assert any(getattr(f, 'vibesim_autotune_guard', False) for f in sys.meta_path)"

ENV PYTHONPATH=/opt/vibesim \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    TRITON_CACHE_DIR=/cache/triton \
    HOME=/cache/home
WORKDIR /opt/vibesim

# Where `alignment/load_generator/runner.py` looks for it.
COPY --from=req-frontend /src/req-frontend/target/release/session_runner \
    /opt/vibesim/alignment/load_generator/req-frontend/target/release/session_runner

# The patches, onto the official image's checkout (and so onto its editable
# install). A patch that does not apply means the image is not BASE's commit.
COPY reference_engines/sglang/patches /opt/vibesim/reference_engines/sglang/patches
RUN cd /sgl-workspace/sglang \
    && git -c safe.directory='*' apply /opt/vibesim/reference_engines/sglang/patches/*.patch \
    && /opt/sglang/bin/python -c "import sglang.srt.observability"

COPY profiling /opt/vibesim/profiling
COPY launcher /opt/vibesim/launcher
COPY gpu /opt/vibesim/gpu

# The envs images.py assigns this image (`selfcheck` loads their runners).
# sgl_kernel picks its compiled ops by the GPU's compute capability at import,
# and a build has no GPU; the first profiled sglang_env kernel imports it.
ARG VIBESIM_PROFILE_ENVS
RUN /opt/sglang/bin/python -c "import torch; assert torch.version.cuda == '13.0'" \
    && /opt/sglang/bin/python -m profiling.container.selfcheck ${VIBESIM_PROFILE_ENVS}

# Last, so a new commit only rewrites the manifest and labels.
ARG VIBESIM_REVISION=unknown
ARG VIBESIM_IMAGE_INPUTS=unknown
ARG VIBESIM_BASE_IMAGE=unknown
ARG VIBESIM_UPSTREAM_COMMIT=unknown
ARG VIBESIM_REQ_FRONTEND_REVISION=unknown
ARG VIBESIM_CACHE_KEY=unknown
ENV VIBESIM_CAPTURE_RUNTIME=in-image \
    VIBESIM_CAPTURE_PYTHON=/opt/vibesim-capture/.venv/bin/python
# What this image is, for a process running inside it (profiling/exec/env.py,
# alignment/profiler/capture_facts.py).
RUN /opt/sglang/bin/python -c "import json; json.dump({'image': 'sglang', 'inputs': '${VIBESIM_IMAGE_INPUTS}', 'envs': '${VIBESIM_PROFILE_ENVS}'.split(), 'python': '/opt/sglang/bin/python', 'python_paths': [], 'capture_python': '/opt/vibesim-capture/.venv/bin/python', 'nsys': '/opt/nvidia/nsight-systems/2025.5.2/bin/nsys', 'base_image': '${VIBESIM_BASE_IMAGE}', 'upstream_commit': '${VIBESIM_UPSTREAM_COMMIT}', 'req_frontend_revision': '${VIBESIM_REQ_FRONTEND_REVISION}'}, open('/etc/vibesim-profiler-image.json', 'w'))"
LABEL org.opencontainers.image.title="ServingStudio SGLang reference engine" \
      org.opencontainers.image.revision="${VIBESIM_REVISION}" \
      ai.vibesim.image.inputs="${VIBESIM_IMAGE_INPUTS}" \
      ai.vibesim.base-image="${VIBESIM_BASE_IMAGE}" \
      ai.vibesim.upstream-commit="${VIBESIM_UPSTREAM_COMMIT}" \
      ai.vibesim.req-frontend.revision="${VIBESIM_REQ_FRONTEND_REVISION}" \
      ai.vibesim.cache-key="${VIBESIM_CACHE_KEY}" \
      ai.vibesim.nsys.version="${NSYS_VERSION}" \
      ai.vibesim.profile-envs="${VIBESIM_PROFILE_ENVS}" \
      ai.vibesim.python="/opt/sglang/bin/python"

# Callers name the interpreter and module (`--entrypoint`), as for vLLM; the
# base's nvidia_entrypoint.sh banner is not wanted in a worker's output.
ENTRYPOINT []
CMD []
