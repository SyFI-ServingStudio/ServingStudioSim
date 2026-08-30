# VibeSim test-tier runner. One recipe per capability tier (see tests/conftest.py
# + skill `dev-run-tests`). Install just: `cargo install just` (or your package mgr).
#
#   just            # = test-cpu (the fast default gate)
#   just test-gpu   # GPU tier (throughput regression etc.)
#   just test-all   # cpu + gpu
#   just test-bench # separate perf step; run after test-all for full validation

# libpython dir — the Rust test binary embeds PyO3 and crashes on
# `libpython3.12.so` without it on LD_LIBRARY_PATH (uv pins the 3.12 interp).
libdir := `uv run --no-sync python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`

# DeepGEMM (grouped_gemm `deepgemm` backend) builds JIT-only and skips its
# git-clean version assert when this is 0; default is 1, which forces an
# install-time CUDA compile AND asserts a clean tree (fails in uv's build clone).
# Exported to every recipe so a cold `just sync` builds deep_gemm correctly
# without anyone passing it by hand. (Once built, uv caches the wheel by commit.)
export DG_USE_LOCAL_VERSION := "0"

# Default: the cpu gate.
default: test-cpu

# Sync the uv env (builds the pinned deep_gemm wheel on a cold cache). Use this
# as the install entrypoint so DG_USE_LOCAL_VERSION above is in effect.
sync:
    # no-build-isolation requires DeepGEMM's setup.py imports to be installed
    # before DeepGEMM itself. A single cold sync has no such install ordering.
    uv sync --inexact --no-install-package deep-gemm
    uv sync

# Reproducible CUDA-13 environment for vLLM-backed L1 kernels.
profile-container-build:
    profiling/container/build.sh

# cpu tier — Rust unit tests + deterministic mocked pytest. No GPU/binary.
# `workers` is the xdist worker count. Not `auto`: that means one worker per core,
# and each worker pays ~9 s importing torch + flashinfer, so a big host spends
# more on startup than it saves. 16 is the measured knee here (17 s, against 19 s
# at 8 and 24); override on a smaller machine with `just test-cpu workers=4`.
test-cpu workers="16": sync
    LD_LIBRARY_PATH="{{libdir}}:${LD_LIBRARY_PATH:-}" uv run --no-sync cargo test -p simulator --lib
    uv run --no-sync pytest -m "not gpu and not agent and not bench" -n {{workers}}

# gpu tier — needs a CUDA device. Auto-includes needs_binary/needs_db tests when
# present; the perf_api bridge / launcher set their own subprocess env.
test-gpu: sync
    uv run --no-sync pytest -m gpu

# agent tier — Codex runner+judge skill cases (expensive; opt-in).
test-agent: sync
    uv run --no-sync python tests/skill_tests/run_codex_skill_tests.py

# bench tier — separate perf step: sim-speed (warn-only) + Rust release microbenches (ignored).
test-bench: sync
    uv run --no-sync pytest -m bench
    LD_LIBRARY_PATH="{{libdir}}:${LD_LIBRARY_PATH:-}" uv run --no-sync cargo test -p simulator --release -- --ignored --nocapture

# Main correctness gate (cpu + gpu). Full validation also runs `just test-bench`.
test-all: test-cpu test-gpu

# (Re)record per-GPU goldens for throughput + sim-speed on this device.
update-golden: sync
    uv run --no-sync pytest -m "gpu or bench" --update-golden

# ── alignment campaigns (skill: operate-run-alignment) ───────────────────────
# The campaign layer drives the five alignment phases across a whole case matrix.
# `check` is pure CPU and is also what `just test-cpu` runs; the rest are the
# daily commands. `run` is deliberately absent here: it takes a --phase and a
# --out-root that only the operator knows, and there is no --all.

# Validate every pack under presets/alignment/ (no GPU, no build). Pass extra
# flags through, e.g. `just alignment-check --pack presets/alignment/<name>`.
alignment-check *flags:
    uv run --no-sync python -m launcher alignment-campaign check {{flags}}

# Read a completed campaign's Analyzer reports into a metrics document, then
# judge it against the pack's tolerances. `runs` is colon-separated.
alignment-compare pack runs out="/tmp/alignment_metrics.json":
    uv run --no-sync python -m launcher alignment-campaign extract --pack {{pack}} --runs {{runs}} --out {{out}}
    uv run --no-sync python -m launcher alignment-campaign compare --pack {{pack}} --measured {{out}}

# Accept the current numbers as the new baseline. Review the compare output
# first: this is the only writer of tests/golden/alignment_<pack>/.
alignment-record pack out="/tmp/alignment_metrics.json" *flags:
    uv run --no-sync python -m launcher alignment-campaign compare --pack {{pack}} --measured {{out}} --record {{flags}}
