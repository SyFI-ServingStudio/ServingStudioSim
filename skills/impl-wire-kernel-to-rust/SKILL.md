---
name: impl-wire-kernel-to-rust
description: >-
  Use when implementing the Rust timing/cache wiring for a ServingStudio Sim L1 kernel
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

Before implementing or profiling the grid, calculate its feasible-coordinate
count for a representative resolved `KernelConfig`: expand the Cartesian grid,
remove cells selected by `infeasible_mask()`, and do not multiply by the number
of backends. The count must be at most **500**. If it is larger, stop and return
the proposed axes, count, and likely cause to the orchestrator; splitting the
same grid across profiling calls does not satisfy the ceiling.

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

Only cache dimensions that affect physical execution. Preserve exact ragged
topology in `Input` when totals cannot determine planner, page lookup, or a
branch, but do not mechanically copy upstream popularity into every helper.
Treat runtime capacity and checkpoint capability as separate identities. Before
adding an axis, record its timing impact, expected fidelity gain, grid cost, and
DB migration consequence.

Profile rows do not own runtime semantics. If source-backed workload derivation
changes, update the payload contract and re-profile; do not add a model-specific
legacy conversion merely to hit old keys. Reusing another profile table through
`profile_kind()` requires identical semantics, args meaning, dtype/layout, and
logical callable boundary.

One Rust L1 kind represents one inseparable semantic callable even if that
callable contains a fixed launch sequence or a runtime chunk loop. Do not expose
chunks, ranks, layers, or capacity as permanent upper-layer slots. Truly
independent production operations still need distinct kinds/slots.

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

Add the smallest focused tests that protect the physical mapping: a
kernel-specific Config rejection, exact Input-to-coordinate projection,
infeasible-domain boundary, or exact Python payload forwarding. Shared derive,
registry, cache-kind, and metadata behavior belongs in shared timing tests; do
not restate it once per kernel. Every test should identify the production defect
it catches.

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
decision, feasible-coordinate count, `slot_input` decision, commands run, and
any blocker for cache fidelity or upper-layer consumers.
