#!/usr/bin/env bash
# Build the `flashinfer_kda_env` profiling venv (profiling/exec/env.py).
#
# The env serves the kda_chunk_prefill flashinfer_* backends. It pins:
# - FlashInfer nightly 0.7.2.dev20261008, built from main at 73b63f88, which
#   carries the TIRx (#5613), CuTe DSL persistent and small-BH (#5032) KDA
#   prefill backends. The 0.7.1 release branch lacks TIRx.
# - Torch 2.11.0 (cu130) and CuTe DSL >= 4.7, which those backends need.
# - The TVM / TIRx Lite pins from FlashInfer's docs/api/kda.rst. TIRx JIT
#   compiles with the CUDA 13 `nvcc` found through CUDA_PATH or PATH.
#
# Usage: profiling/exec/flashinfer_kda_env.sh [VENV_DIR]
# VENV_DIR defaults to ~/profile_envs/flashinfer_kda. Any other directory is
# symlinked there, which is where the registry looks; put large venvs on the
# disk that holds UV_CACHE_DIR so uv can hardlink instead of copying.
set -euo pipefail

FLASHINFER_WHEEL="flashinfer_python-0.7.2.dev20261008-py3-none-any.whl"
FLASHINFER_URL="https://github.com/flashinfer-ai/whl/releases/download/nightly-v0.7.2-20261008/${FLASHINFER_WHEEL}"
FLASHINFER_SHA256="35b41bb54be1531de35b41a4668c97c9955fb8676eb66ac857ec44334fc60b84"
TIRX_KERNELS="tirx-kernels @ git+https://github.com/mlc-ai/tirx-kernels.git@c4b700e7e8c390f069b369b588ecfe20215e5850"

link="${HOME}/profile_envs/flashinfer_kda"
target="${1:-${link}}"

download_dir="$(mktemp -d "${TMPDIR:-/tmp}/flashinfer_kda_env.XXXXXX")"
trap 'rm -rf "${download_dir}"' EXIT
curl -fsSL -o "${download_dir}/${FLASHINFER_WHEEL}" "${FLASHINFER_URL}"
echo "${FLASHINFER_SHA256}  ${download_dir}/${FLASHINFER_WHEEL}" | sha256sum -c -

uv venv --python 3.12 "${target}"
uv pip install --python "${target}/bin/python" \
    "torch==2.11.0" \
    "${download_dir}/${FLASHINFER_WHEEL}" \
    "nvidia-cutlass-dsl[cu13]>=4.7.0" \
    "apache-tvm==0.27.0" \
    "apache-tvm-ffi==0.1.14.post1" \
    "${TIRX_KERNELS}" \
    "numpy>=2.0" \
    "nvidia-ml-py>=12.0.0"

if [[ "$(realpath -m "${target}")" != "$(realpath -m "${link}")" ]]; then
    mkdir -p "$(dirname "${link}")"
    ln -sfn "$(realpath "${target}")" "${link}"
fi
"${link}/bin/python" -c "import flashinfer, tvm, tirx_kernels.tirx_lite; print(flashinfer.__version__, flashinfer._build_meta.__git_commit__)"
