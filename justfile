# VibeSim test-tier runner. One recipe per capability tier (see tests/conftest.py
# + skill `dev-run-tests`). Install just: `cargo install just` (or your package mgr).
#
#   just            # = test-cpu (the fast default gate)
#   just test-gpu   # GPU tier (throughput regression etc.)
#   just test-all   # cpu + gpu
#   just test-bench # separate perf step; run after test-all for full validation

# libpython dir — the Rust test binary embeds PyO3 and crashes on
# `libpython3.12.so` without it on LD_LIBRARY_PATH (uv pins the 3.12 interp).
libdir := `uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`

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
    uv sync

# cpu tier — Rust unit tests + deterministic mocked pytest. No GPU/binary.
test-cpu:
    LD_LIBRARY_PATH="{{libdir}}:${LD_LIBRARY_PATH:-}" uv run cargo test -p simulator --lib
    uv run pytest -m "not gpu and not agent and not bench"

# gpu tier — needs a CUDA device. Auto-includes needs_binary/needs_db tests when
# present; the perf_api bridge / launcher set their own subprocess env.
test-gpu:
    uv run pytest -m gpu

# agent tier — Codex runner+judge skill cases (expensive; opt-in).
test-agent:
    uv run python tests/skill_tests/run_codex_skill_tests.py

# bench tier — separate perf step: sim-speed (warn-only) + Rust release microbenches (ignored).
test-bench:
    uv run pytest -m bench
    LD_LIBRARY_PATH="{{libdir}}:${LD_LIBRARY_PATH:-}" uv run cargo test -p simulator --release -- --ignored --nocapture

# Main correctness gate (cpu + gpu). Full validation also runs `just test-bench`.
test-all: test-cpu test-gpu

# (Re)record per-GPU goldens for throughput + sim-speed on this device.
update-golden:
    uv run pytest -m "gpu or bench" --update-golden
