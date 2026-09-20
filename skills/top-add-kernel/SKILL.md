---
name: top-add-kernel
description: >-
  Use as the entry point whenever the user asks to add a ServingStudio Sim L1 kernel. This
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

## Several kernels at once

A split table usually names more than one *new dedicated kernel*. They are
independent work — different kinds, different runner files, different
`profile.db` tables — so run them in parallel. Who fans out depends on the role
you are in:

- **Only agent** (nobody above you, nobody below). Launch one subagent per
  kernel, in parallel; each runs this skill end to end for its own kernel. Do
  not walk the list yourself.
- **Orchestrator.** Give the implementer the **whole set in one brief**. Issuing
  one kernel per brief serializes work that has no ordering constraint.
- **Implementer handed the set.** Fan out again — one subagent per kernel, in
  parallel. Receiving several kernels is not an instruction to do them in order.

Before any fan-out, take the reuse verdict below **once for the whole set**: two
agents must not independently mint the same new kind. Then sequence only what
genuinely depends — a new backend of a kind another agent is still creating.

Two or more agents writing one repo need `dev-orchestrate-parallel-subagents`:
a git worktree each, or they clobber each other on `mod.rs`, `__init__.py`, and
the registry. Read it before launching.

This does not override `top-add-new-arch`'s largest-share-first stop rule. That
rule decides **which** kernels are worth building; the ones that clear it go in
parallel, not one after another.

## Steps

Before choosing either path, require a reuse verdict. Compare the nearest
existing kind's operation semantics, `KernelArgs` meaning, dtype and storage
layout, production callable, and logical launch boundary. A new backend may
change the implementation but not those meanings. Create a new kind only when
this comparison proves that reuse would change the contract; when source review
is inconclusive, require a matched-shape A/B with identical logical I/O.

1. Coordinate the Python profiling orchestrator.

Use `orchestrator-add-kernel-to-python-profile` to define and validate the
Python profiling side:

- decide whether this is a brand-new kernel kind or a new backend;
- ground the operation and dtype/shape contract;
- identify the production public callable and build an independent Torch
  correctness oracle when needed;
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
