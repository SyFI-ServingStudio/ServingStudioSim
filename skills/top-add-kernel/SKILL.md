---
name: top-add-kernel
description: >-
  Use as the entry point whenever the user asks to add an MLSim L1 kernel. This
  is a top-level router: it does not contain implementation details. First route
  Python profiling work to orchestrator-add-kernel-to-python-profile, then route
  Rust timing/cache wiring to orchestrator-wire-kernel-to-rust once the Python
  handoff is ready.
---

# Top Add Kernel

Use this skill as the entry point for adding an L1 kernel. Do not implement from
this file; issue the work to the two role-specific orchestrators.

## Steps

1. Fix it in Python.

Use `orchestrator-add-kernel-to-python-profile` to define and validate the
Python profiling side:

- decide whether this is a brand-new kernel kind or a new backend;
- ground the operation and dtype/shape contract;
- build or verify the Torch reference when needed;
- register Python profiling backends and verify `python -m profiling` smoke.

2. Wire it into Rust.

Use `orchestrator-wire-kernel-to-rust` after the Python side has a working
handoff:

- map Python `KIND`, backend strings, args fields, and dtype axes to Rust;
- assign `KernelSpec` / Config / Input / cache wiring;
- compile and introspect through `kernel-query`;
- validate cache fidelity and decide whether the cache/grid needs refinement.

## Completion

End-to-end kernel work is complete only when both orchestrators have produced
their closeout evidence, or when the user explicitly asks for only the Python
side or only the Rust side.
