---
name: impl-wire-kernel-to-rust
description: >-
  Use when implementing the Rust timing/cache wiring for an MLSim L1 kernel
  after the orchestrator has validated the Python profiling handoff and selected
  the Rust Config/Input/cache shape. Covers KernelSpec implementation,
  sweep/enumerate/cache wiring, mod.rs exports, slot_input logging, Rust tests,
  and kernel-query grid smoke. Does not implement Python runners or choose the
  kernel semantics.
---

# Impl Wire Kernel To Rust

You are the implementer for Rust timing/cache wiring under
`simulator/src/timing/`. The orchestrator should provide the Python-to-Rust
handoff and the chosen Rust shape: `KIND`, backends, args fields, Config/Input
split, dtype axes, sweep grid, cache kind, and any infeasible-mask decision.

Do not change Python profiling code in this skill.

## Read First

Read the local implementation before editing:

- `simulator/src/timing/README.md`
- the closest `simulator/src/timing/kernels/<kind>.rs`
- `simulator/src/timing/kernels/engine.rs`
- `simulator/src/timing/kernels/mod.rs`
- `simulator/src/timing/slot_input.rs`
- `simulator/src/timing/sweep.rs`
- `simulator/src/timing/cache/mod.rs`

## Expected Write Scope

Normally edit only:

- `simulator/src/timing/kernels/<kind>.rs`;
- `simulator/src/timing/kernels/mod.rs`;
- `simulator/src/timing/slot_input.rs` if the input can reach a cost-log leaf.

Stop and ask before adding a new cache variant, new derive behavior, bridge
core behavior, central kind registry, or L2/L3/L4 consumer.

## Kernel File Contract

The new kernel file should follow the closest existing kind and define:

- `Config` with `#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]`;
- `Input` with `#[derive(SweepCoords)]` when cache axes equal public query
  fields, or a manual `SweepCoords` implementation for re-axis/domain
  projections;
- `serde::Serialize` and `serde::Deserialize` on `Input` when it participates in
  cost logs or `kernel-query`;
- unit `Spec` implementing `KernelSpec`;
- `const KIND` exactly equal to the Python `KIND`;
- `sweep_grid`;
- `cache_kind`;
- `enumerate`;
- optional `profile_kind()` only when a cache variant reuses another Python
  profile table;
- optional `infeasible_mask()` for physically unreachable grid cells;
- `register_kernel!(<Name>Kernel, <Name>Spec)`.

## Field And Dtype Contract

`enumerate` must emit one `ArgsPayload` per profiled grid point and backend:

- always include `.with("backend", backend)`;
- include every Python `KernelArgs` field with the exact same snake_case name;
- do not emit fields that are not in Python `KernelArgs`;
- use Config fields for static identity and Input/grid fields for runtime sweep
  dimensions;
- tag dtype-bearing Config fields with `#[compute_dtype]` and `#[kv_dtype]`
  where relevant so launcher backend selection sees the same axes as Python
  `BackendSupport`.

If a Python args field cannot be derived from Config/Input/grid/backend, stop
and return to the orchestrator.

## Wiring

Wire only the required Rust surfaces:

- add `pub mod <kind>;` and `pub use ...` exports in
  `simulator/src/timing/kernels/mod.rs`;
- add `SlotInput` import and `log_inputs!` entry in
  `simulator/src/timing/slot_input.rs` when the kernel's Input can reach an
  evaluated leaf;
- do not add a central kind enum or match for `kernel-query`; `register_kernel!`
  and `inventory` own discovery.

## Tests And Smoke

Add focused tests in the new kernel file:

- Config identity and `describe_config`;
- Input `coords()` / `coord_field_names()`;
- `sweep_grid` dimensionality and representative axes;
- `cache_kind`;
- `enumerate` emits `backend` plus all Python args fields and no extras;
- dtype tags where applicable.

Run:

```bash
uv run cargo test --lib timing
```

If a simulator binary is needed, build it:

```bash
uv run cargo build --release -p simulator
```

Then run a `kernel-query` grid smoke with a representative Config:

```bash
printf '%s' '{"op":"grid","kind":"<kind>","config":{...}}' \
  | target/release/simulator kernel-query
```

The grid smoke should return JSON with `input_fields` and `grid_axes` and should
not require GPU profiling.

## Report Back

Return files changed, Config fields, Input fields, emitted ArgsPayload field
list, dtype tags, selected cache kind, sweep grid summary, infeasible-mask
decision, `slot_input` decision, commands run, and any blocker for cache
fidelity or upper-layer consumers.
