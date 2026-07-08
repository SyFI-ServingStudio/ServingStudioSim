---
name: top-add-kernel
description: >-
  Use as the entry point whenever the user asks to add an VibeSim L1 kernel. This
  is a top-level orchestrator skill: it does not contain implementation details. Do not pass
  the whole end-to-end kernel request to an implementer; implementers only
  receive bounded impl-* subtasks issued from the relevant orchestrator phase.
---

# Top Add Kernel

Use this skill when you are the orchestrator for an end-to-end L1 kernel
request. Do not implement from this file, and do not treat this file as an
implementer brief. Use it to enter the two child orchestrator skills in order.

## Delegation Boundary

Stay in the orchestrator role. Read and follow
`orchestrator-add-kernel-to-python-profile` first. After that phase produces a
working Python-to-Rust handoff, read and follow
`orchestrator-wire-kernel-to-rust`.

Do not ask an implementer to "add the kernel end to end", use this top-level
skill, choose the orchestration sequence, or own both Python profiling and Rust
timing/cache wiring.

Only issue implementer work from inside the relevant child orchestrator phase.
Each implementer brief must be a narrow leaf task that names the exact `impl-*`
skill to use and the evidence required for you, the orchestrator, to verify
completion.

## Steps

1. Coordinate the Python profiling orchestrator.

Use `orchestrator-add-kernel-to-python-profile` to define and validate the
Python profiling side:

- decide whether this is a brand-new kernel kind or a new backend;
- ground the operation and dtype/shape contract;
- build or verify the Torch reference when needed;
- register Python profiling backends and verify `python -m profiling` smoke;
- produce the Python-to-Rust handoff for the Rust orchestrator.

2. Coordinate the Rust timing/cache orchestrator.

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
