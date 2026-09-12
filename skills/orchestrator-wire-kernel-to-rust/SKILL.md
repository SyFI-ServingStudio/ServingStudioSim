---
name: orchestrator-wire-kernel-to-rust
description: >-
  Use when orchestrating the Rust timing/cache wiring for a ServingStudio Sim L1 kernel
  after the Python profiling side is defined. This skill is for the coordinator
  who turns a Python-to-Rust handoff into Rust implementation tasks, verifies
  KernelSpec/cache/bridge behavior, and decides when cache fidelity is good
  enough. It does not implement Python runners or profile new backend sources.
---

# Orchestrator Wire Kernel To Rust

You are the orchestrator, not the implementer. Your job is to take the
Python-side facts for a kernel and turn them into Rust timing/cache wiring tasks
under `simulator/src/timing/`, with completion criteria you can verify.

This skill starts after the Python profiling side has a working registry entry,
runner, public CLI smoke, and a Python-to-Rust handoff. If those facts are
missing, return to `orchestrator-add-kernel-to-python-profile`.

## Grounding

Before issuing tasks, read the current Rust timing implementation:

- `simulator/src/timing/README.md`
- the closest `simulator/src/timing/kernels/<kind>.rs`
- `simulator/src/timing/kernels/engine.rs`
- `simulator/src/timing/kernels/mod.rs`
- `simulator/src/timing/slot_input.rs`
- `simulator/src/timing/cache/mod.rs`
- `skills/impl-validate-kernel-cache/SKILL.md` when interpolation or quantization
  fidelity matters.

Use docs as background, but let current code decide the exact API shape.

## Verification Ownership

The implementer may be another agent, but the verifier is the orchestrator. Ask
for evidence, then independently inspect the Rust diff and rerun the relevant
commands when feasible.

## Task R.1 — Validate The Python-To-Rust Handoff

Confirm the Python side gives enough facts to wire Rust:

- `KIND` / facade stem;
- backend strings;
- `KernelArgs` fields and order;
- metric family;
- dtype/GPU capability axes;
- representative smoke spec;
- shape and dtype constraints discovered during Python validation.

Things to verify:

- `uv run python -m profiling list --json` shows the expected `(KIND, backend)`
  rows;
- the generated Python facade resolves, e.g.
  `uv run python -c "from profiling import perf_api; assert hasattr(perf_api, 'get_<kind>_times')"`;
- the Python profiling smoke succeeds through the public CLI with a temporary
  DB, e.g.
  `uv run python -m profiling run <kind> --backend <backend> --db /tmp/<kind>_<backend>_handoff.db --spec '<json spec>' --json`;
- `KIND` matches the Python table/facade stem exactly;
- every Python `KernelArgs` field has a Rust source: either a Config field or an
  Input field emitted by `enumerate`;
- `backend` remains routing metadata and is emitted separately by Rust;
- dtype axes are known well enough to choose `#[compute_dtype]` and `#[kv_dtype]`
  tags, or the kernel is explicitly dtype-agnostic.

If the Config/Input split is ambiguous, ask the user or return to the Python
orchestrator before assigning implementation.

## Task R.2 — Decide Rust Kernel Shape

Ask the implementer to propose the Rust `KernelSpec` shape before writing code:

- new `simulator/src/timing/kernels/<kind>.rs` or reuse/variant of an existing
  kind via `profile_kind()`;
- `Config` fields: static identity, including `backends` and `gpu_name`;
- `Input` fields: runtime query shape;
- `SweepCoords`: physical axes or a re-axis projection;
- `sweep_grid`;
- `cache_kind`;
- `infeasible_mask()` if some grid cells cannot correspond to real shapes.

A deliberately designed cache grid may contain at most **500 feasible
coordinates** for one resolved `KernelConfig`, counted after
`infeasible_mask()` and before multiplying by candidate backends. This is a
design ceiling, not a profiling-batch limit. Do not split a larger grid across
calls to evade it. If a proposal exceeds 500, reject it and inspect whether it
cross-products axes that do not independently affect timing, retains
unreachable shapes, or compensates for an inadequate cache or extrapolation
policy. Redesign the axes, projection, cache policy, or supported domain before
implementation.

Things to verify:

- the proposed `enumerate` emits exactly Python args fields plus `backend`;
- Config fields and Input fields explain the model/kernel shape in stable terms;
- cache axes are not confused with public query fields when a re-axis is used;
- the proposal reports its feasible-coordinate count and it is at most 500;
- `CacheKind` is one currently built in `timing/cache/mod.rs`, unless the user
  explicitly approved adding a new cache variant;
- if multiple cache/grid choices are plausible, summarize tradeoffs and ask the
  user which one to implement first.

## Task R.3 — Implement, Compile, And Introspect

Ask the implementer to add the Rust timing kernel by using
`impl-wire-kernel-to-rust`. The brief should include the validated handoff from
R.1 and the Rust shape decision from R.2.

Things to verify:

- the implementer followed `impl-wire-kernel-to-rust`;
- the Rust diff is limited to the agreed timing files unless a blocker was
  escalated;
- `Spec::KIND` equals the Python `KIND`;
- emitted `ArgsPayload` fields match Python args plus `backend`;
- dtype tags match Python `BackendSupport`;
- cache kind, sweep grid, infeasible mask, and `slot_input.rs` behavior match the
  R.2 decision;
- focused Rust tests in the new kernel file cover Config identity,
  `describe_config`, Input coords, `sweep_grid`, `cache_kind`, and `enumerate`
  field shape;
- `uv run cargo test --lib timing` passes;
- `uv run cargo build --release -p simulator` succeeds when a binary is needed;
- `simulator kernel-query` `grid` works for a representative Config and returns
  `input_fields` / `grid_axes` without GPU profiling.

Example `kernel-query` grid smoke:

```bash
printf '%s' '{"op":"grid","kind":"<kind>","config":{...}}' \
  | target/release/simulator kernel-query
```

## Task R.4 — Cache Fidelity Gate

Do this after Rust wiring compiles and `kernel-query grid` works. Ask the
implementer to validate interpolation or quantization error through
`impl-validate-kernel-cache`.

Things to verify:

- the fidelity run compares Rust cache eval against Python `perf_api` ground
  truth, not a reimplemented Python interpolation;
- simple kernels can use the generic cache-fidelity CLI;
- re-axis/domain kernels use a thin caller with physical probes;
- report includes median ratio, within-rate, worst ratio, regions, and CSV/log
  path;
- the orchestrator inspected the final CSV/log summary, not just accepted that
  the command ran;
- the reported error is acceptable under the user-specified threshold, or the
  default fidelity bar from `impl-validate-kernel-cache`;
- deviations are localized by region and failure mode: interior interpolation,
  extrapolation, quantization bucket boundaries, re-axis mismatch, or noisy
  backend measurements;
- unacceptable error leads to a concrete remediation plan, not silent
  acceptance. Consider a denser sweep grid, a different built-in cache kind, a
  better physical re-axis, a fitted variant if the codebase supports it, or a
  narrower supported domain;
- if the remediation would increase profiling cost materially, add a new cache
  or fitting mechanism, change the public query/input contract, or accept a
  known inaccurate region, ask the user before continuing;
- after remediation, rerun fidelity and validate the new final result.

Use `Cache1DDirect` quantization the same way: probe inside buckets and at
boundaries through the fidelity harness, then compare Rust cache output to
`perf_api` ground truth.

## Closeout

End with a short Rust wiring report:

- Rust kind and Python kind/facade matched;
- Config fields, Input fields, and emitted ArgsPayload fields;
- backends and dtype tags;
- chosen sweep grid, cache kind, and any infeasible mask;
- files changed;
- cargo/kernel-query/fidelity commands run;
- open blockers before L2/L3/L4 consumers should use this kernel.

Do not ask the implementer to modify Python runners, profile new backend
sources, or wire L2/L3/L4 users in this skill.
