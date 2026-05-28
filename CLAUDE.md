# MLSim — agent notes

Discrete-event simulator for ML serving/training workloads. Rust core
(`simulator/`) + Python L1 profiling/launcher, bridged via PyO3. Answer user questions using Chinese. Draft plans also in Chinese. Write code comments in English.

## Environment (read first)

- **Initialize the env with `just sync`** (install `just` via `cargo install just`).
  This is the canonical bootstrap: it runs `uv sync` with `DG_USE_LOCAL_VERSION=0`
  exported (set in the `justfile`), which the pinned `deep_gemm` git dependency
  needs to build JIT-only and skip its git-clean version assert. A bare `uv sync`
  on a **cold** cache fails to build `deep_gemm` without that var; once built, uv
  caches the wheel by commit so later plain `uv sync`/`uv run` are fine.
- **Always run under `uv`** (`uv run cargo …`, `uv run python …`, `uv run pytest`).
  It pins the 3.12 venv; a bare `python`/`cargo` links the system 3.9 and crashes
  PyO3 init. The Rust test binary also needs libpython on `LD_LIBRARY_PATH`
  (the `just` recipes set it; do it by hand as
  `` LD_LIBRARY_PATH="$(uv run python -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH" ``).

## Running a simulation

The launcher is the run interface — it builds the release binary + the PyO3 env
itself (no manual `LD_LIBRARY_PATH`/`PYTHONPATH`) and prewarms the kernel cache:

```bash
uv run python -m launcher <preset>.json [--dry-run] [--override k=v ...]
```

See skill `run-simulation` for preset/sweep conventions and dated log dirs.

## Testing

Tests are split into **capability tiers** by pytest marker; `tests/conftest.py`
auto-skips what the host can't run. Use the `just` recipes (install: `cargo
install just`) — they encode the env gotchas:

```bash
just test-cpu     # Rust --lib + mocked pytest. Fast, no GPU. The default gate.
just test-gpu     # gpu tier (throughput regression, cupti). Needs a CUDA device;
                  #   the launcher builds the binary + warms profile.db itself.
just test-all     # cpu + gpu — the usual "did my refactor break anything".
just test-bench   # opt-in: sim-speed median + Rust --ignored microbenches.
just update-golden # re-record per-GPU goldens after an INTENTIONAL cost change.
```

After a refactor: `just test-all`. Modeled throughput is bit-identical run-to-run,
so the throughput regression test **warns at ±1%** — if it warns, the refactor
changed the cost model. If that change was intended, confirm the numbers and
`just update-golden`. Hardware-dependent goldens are tagged per GPU under
`tests/golden/<metric>/<gpu_name>.json`; the test skips on a GPU with no recorded
golden.

Tiers, markers, the golden store, and how to add a tiered test: skill `run-tests`.
Other test/quality skills: `validate-kernel-cache`, `profile-run-existing-kernel`,
`profile-add-kernel`, `profile-sim-speed`.
