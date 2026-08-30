---
name: dev-run-tests
description: Use when asked to run, select, or understand VibeSim's tests — which tier needs a GPU / the built binary / an agent runtime / is just CPU, how the auto-skip works, and how the per-GPU throughput & sim-speed goldens are recorded. Covers the `just test-*` recipes, pytest markers, and the GPU-type-tagged golden store. NOT for writing a new kernel's cache-fidelity check (that is impl-validate-kernel-cache).
---

# Run VibeSim Tests (capability tiers)

VibeSim tests span three runtimes (Rust `cargo test`, `pytest`, and a Codex
skill-test harness) and several **capability tiers**. A test declares its tier
with a pytest marker; `tests/conftest.py` **auto-skips** the tiers the host can't
run, so bare `uv run pytest` "just works" anywhere. Prefer the `just` recipes —
they encode the env gotchas (notably the libpython `LD_LIBRARY_PATH` the Rust
test binary needs) so you don't re-derive them.

Run everything under `uv` (pins the 3.12 venv; bare python/cargo links system 3.9
and crashes — see memory `vibesim_pyo3_build_python_pin`).

## Tiers

| Tier | Marker | Needs | Gate (auto-skip when…) |
|---|---|---|---|
| cpu | *(unmarked)* | nothing — deterministic | always runs |
| binary | `needs_binary` | release `simulator` built | binary absent |
| gpu | `gpu` | a CUDA device | `torch.cuda.is_available()` false |
| db | `needs_db` | warm `profile.db` rows | *(test asserts/skips itself)* |
| agent | `agent` | Codex runner+judge | not `--run-agent` (opt-in) |
| bench | `bench` | — (perf, non-deterministic) | not `-m bench` (opt-in) |

cpu/binary/gpu auto-skip on *absence*; agent/bench are **opt-in** (expensive /
flaky), never collected unless explicitly requested.

## Recipes (`just`)

Install once: `cargo install just`. Then from the repo root:

```bash
just              # = test-cpu (the fast default gate)
just test-cpu     # Rust --lib tests + mocked pytest (no GPU/binary)
just test-gpu     # gpu tier (throughput regression, cupti, …) on a CUDA box
just test-agent   # Codex skill-test harness (tests/skill_tests/*.md)
just test-all     # cpu + gpu
just test-bench   # separate perf step: sim-speed (warn-only) + Rust release microbenches (--ignored)
just update-golden # (re)record per-GPU goldens on THIS device
```

For a full GPU-box validation, run it as two explicit steps:

```bash
just test-all
just test-bench
```

Raw pytest equivalents (when you need `-k` / `-x`): `uv run pytest -m gpu`,
`uv run pytest -m "not gpu and not agent and not bench"`, etc. Rust tests need
the libpython dir on `LD_LIBRARY_PATH` (the recipes set it; do the same by hand:
`` LD_LIBRARY_PATH="$(uv run python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH" uv run cargo test -p simulator --lib ``).

## Per-GPU goldens (hardware-dependent metrics)

Modeled throughput and sim-speed depend on the GPU type (the cost model reads a
per-GPU `profile.db`), so their goldens are tagged by device:

```
tests/golden/<metric>/<gpu_name>.json     # e.g. throughput/NVIDIA_H200.json
```

`<gpu_name>` is the exact CUDA device name = the `profile.db` key (e.g.
`"NVIDIA H200"`, not `"H200"`). The `golden` fixture (conftest) loads the file
for the *detected* GPU; a test **skips** if there's no golden for this device
(never asserts a wrong-hardware value). Record/advance with `--update-golden`:

```bash
uv run pytest -m "gpu or bench" --update-golden    # seed throughput + sim-speed
```

These checks **warn** on drift (never hard-fail) — they're monitors, so an
intended cost-model change surfaces a signal instead of breaking the run;
re-record with `--update-golden` once confirmed. Modeled throughput is
**bit-identical** across runs (a pure function of trace + config + cost model),
so its warn bar is tight (±1% — trips on any cost-model change). Sim-speed
(`realtime_x`) is host-load dependent, so `just test-bench` is a separate full-
validation step: it runs the sim several times and warns on >10% drift of the
**median**. The golden key encodes the trace size `n`; changing
`tests/fixtures/gen_throughput_trace.py` invalidates it (re-record). The worked
example is `tests/test_throughput_regression.py`.

### Alignment goldens: recorded by the launcher, not by pytest

`tests/golden/alignment_<pack>/<gpu_name>.json` has the same shape as the store
above (flat `{key: float}`, keys `<variant>/<case>@<metric>`), but **`pytest
--update-golden` does not write it.** Its values come from a completed GPU
alignment matrix, which no test can run, so the recorder is the launcher:

```bash
uv run python -m launcher alignment-campaign extract \
  --pack presets/alignment/<pack> --runs <case_root>[:<case_root>] --out /tmp/metrics.json
uv run python -m launcher alignment-campaign compare \
  --pack presets/alignment/<pack> --measured /tmp/metrics.json --record
# or: just alignment-record <pack> out=/tmp/metrics.json
```

Both paths share one storage implementation (`launcher/golden.py`), which is why
a production command writes under `tests/` — a second store would drift from the
first.

`--record` refuses to write when any case is unavailable, when the reports'
`schema_version` disagrees with the pack's declared `analyzer_schema`, or when a
recorded case depends on a `provisional` calibrated input. That last one is
overridable with `--accept-provisional`, and the provisional field names are then
written into the `.provenance.json` sidecar next to the golden, so a baseline can
never quietly depend on an uncalibrated value.

Tolerances are **not** in the golden store. Re-recording measured values is
routine; moving the standard they are judged against belongs in a reviewed edit
to the pack's `acceptance.yaml`. Sharing one store would let `--record` change
the goalposts along with the numbers.

The CPU-tier `tests/test_alignment_campaign.py` guards the *inputs* to that
recording — pack structure, byte-identical trace regeneration, tolerance
coverage, and that every golden key still names a live variant×case×metric, so a
rename surfaces as a zombie key rather than a silently orphaned number.

## Adding a tiered test

First state the observable behavior and the real defect the test prevents. Keep
independent numerical expectations, actual argument forwarding, invalid-input
rejection, stable logical launch boundaries, and cache interpolation checks.
Do not add per-kernel copies of registry/facade metadata tests or tests that
merely repeat the implementation formula. Keep DB/GPU integration out of the
default CPU tier: mark genuine integration tests with the tier below, and use
the public profiling smoke workflow for per-kernel DB/GPU evidence. Put generic
registry behavior in shared infra tests.

- Pure logic → no marker (cpu). Keep it deterministic and mock the GPU.
- Needs the binary / a CUDA device / warm db → add `@pytest.mark.needs_binary` /
  `gpu` / `needs_db` (module-level `pytestmark` if the whole file is one tier).
- A hardware-dependent number → compare to `golden("<metric>")[key]`, skip if
  unrecorded, record with `--update-golden` (copy the `_warn_vs_golden` pattern;
  warn vs hard-fail is a per-metric call).
- A new GPU-only Rust test → `simulator/tests/` with `#[ignore = "gpu"]` so
  `cargo test --lib` stays fast; the `bench` recipe runs `--ignored`.

## Reference

- Gating brain + fixtures: `tests/conftest.py`. Markers also in
  `pyproject.toml [tool.pytest.ini_options]`.
- Recipes: `justfile`. Worked example: `tests/test_throughput_regression.py`
  (+ `tests/fixtures/gen_throughput_trace.py`).
- Structured run metrics: the simulator writes `<log_dir>/summary.json`
  (`RunSummary` in `simulator/src/sim/run.rs`) — what the throughput test reads.
- Agent tier details: `tests/skill_tests/README.md`.
