# ServingStudioSim test-tier runner. One recipe per capability tier (see tests/conftest.py
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
# at 8 and 24); override on a smaller machine with `just test-cpu 4`.
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

# binaryen release `setup-wasm` installs and `build-wasm` requires.
binaryen_version := "132"

# Writes simulator_wasm.js, simulator_wasm_bg.wasm and version.json, which
# ServingStudioIntro's build-predict-payload copies into its static site. Rebuild
# after every simulator change: the site checks version.json's sim_commit against
# the public API's. WASM_BINDGEN / WASM_OPT override the setup-wasm tools.
# Build the simulator for a browser (simulator/wasm) into target/wasm-pkg/.
build-wasm:
    #!/usr/bin/env bash
    set -euo pipefail
    tools="${WASM_TOOLS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/servingstudio-wasm}"
    want="$(sed -n 's/^wasm-bindgen = "=\(.*\)"$/\1/p' simulator/wasm/Cargo.toml)"
    bindgen="${WASM_BINDGEN:-$tools/wasm-bindgen-$want/bin/wasm-bindgen}"
    wasm_opt="${WASM_OPT:-$tools/binaryen-version_{{binaryen_version}}/bin/wasm-opt}"
    for tool in "$bindgen" "$wasm_opt"; do
        command -v "$tool" >/dev/null || { echo "$tool not found: run just setup-wasm" >&2; exit 1; }
    done
    have="$("$bindgen" --version | awk '{print $2}')"
    [ "$have" = "$want" ] || { echo "wasm-bindgen-cli $have != crate's $want: run just setup-wasm" >&2; exit 1; }
    "$wasm_opt" --version | grep -q "(version_{{binaryen_version}})" \
        || { echo "$("$wasm_opt" --version) != binaryen version_{{binaryen_version}}: run just setup-wasm" >&2; exit 1; }
    commit="$(git rev-parse HEAD)"
    if ! git diff --quiet HEAD -- simulator Cargo.toml Cargo.lock; then commit="$commit-dirty"; fi
    SERVINGSTUDIO_SIM_COMMIT="$commit" cargo build -p simulator-wasm --target wasm32-unknown-unknown --profile wasm
    out=target/wasm-pkg
    rm -rf "$out" && mkdir -p "$out"
    "$bindgen" --target web --no-typescript --out-dir "$out" target/wasm32-unknown-unknown/wasm/simulator_wasm.wasm
    "$wasm_opt" -Oz --enable-bulk-memory --enable-multivalue --enable-mutable-globals --enable-nontrapping-float-to-int --enable-reference-types --enable-sign-ext \
        "$out/simulator_wasm_bg.wasm" -o "$out/simulator_wasm_bg.wasm"
    printf '{"sim_commit": "%s", "kernel_data_format": %s}\n' "$commit" \
        "$(sed -n 's/^pub const KERNEL_DATA_FORMAT: u32 = \([0-9]*\);$/\1/p' simulator/src/timing/bridge/kernel_data.rs)" > "$out/version.json"
    ls -l "$out"

# One directory every checkout shares (WASM_TOOLS_DIR, default
# ~/.cache/servingstudio-wasm) holds the wasm32 target's tools: wasm-bindgen-cli
# at the crate's wasm-bindgen pin and binaryen's wasm-opt at `binaryen_version`.
# Rerun when either pin moves.
# Install the pinned tools build-wasm uses.
setup-wasm:
    #!/usr/bin/env bash
    set -euo pipefail
    tools="${WASM_TOOLS_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/servingstudio-wasm}"
    rustup target add wasm32-unknown-unknown
    want="$(sed -n 's/^wasm-bindgen = "=\(.*\)"$/\1/p' simulator/wasm/Cargo.toml)"
    bindgen="$tools/wasm-bindgen-$want"
    [ -x "$bindgen/bin/wasm-bindgen" ] \
        || cargo install --locked --root "$bindgen" wasm-bindgen-cli --version "$want"
    binaryen="$tools/binaryen-version_{{binaryen_version}}"
    if [ ! -x "$binaryen/bin/wasm-opt" ]; then
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64) platform=x86_64-linux ;;
            Linux-aarch64) platform=aarch64-linux ;;
            Darwin-arm64) platform=arm64-macos ;;
            Darwin-x86_64) platform=x86_64-macos ;;
            *) echo "no binaryen release for $(uname -sm)" >&2; exit 1 ;;
        esac
        release="binaryen-version_{{binaryen_version}}"
        url="https://github.com/WebAssembly/binaryen/releases/download/version_{{binaryen_version}}/$release-$platform.tar.gz"
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' EXIT
        curl -fsSL -o "$tmp/binaryen.tar.gz" "$url"
        expected="$(curl -fsSL "$url.sha256" | awk '{print $1}')"
        actual="$({ sha256sum || shasum -a 256; } < "$tmp/binaryen.tar.gz" 2>/dev/null | awk '{print $1}')"
        [ "$actual" = "$expected" ] || { echo "$url: sha256 $actual != $expected" >&2; exit 1; }
        tar xzf "$tmp/binaryen.tar.gz" -C "$tmp"
        # wasm-opt is static on Linux; the macOS build links lib/libbinaryen.dylib.
        mkdir -p "$binaryen/bin" "$binaryen/lib"
        cp "$tmp/$release/bin/wasm-opt" "$binaryen/bin/"
        cp "$tmp/$release"/lib/*.dylib "$binaryen/lib/" 2>/dev/null || true
    fi
    "$bindgen/bin/wasm-bindgen" --version
    "$binaryen/bin/wasm-opt" --version
