---
name: top-add-kernel
description: >-
  Use as the entry point whenever a ServingStudio Sim L1 kernel is to be added. It
  sequences two child skills for one kernel: Python profiling, then Rust
  timing/cache wiring. A kernel is the unit of delegation: an orchestrator hands
  each kernel over whole, and whoever owns it runs this skill end to end.
---

# Top Add Kernel

Use this skill to add one L1 kernel end to end. It holds no implementation
details: read and follow `orchestrator-add-kernel-to-python-profile` first, and
after it produces a working Python-to-Rust handoff, read and follow
`orchestrator-wire-kernel-to-rust`.

## One kernel is one task

Whoever owns a kernel runs this skill for it end to end, Python through Rust,
and works through the child skills' tasks as its own steps. Splitting one kernel
into a delegation per step turns each step into a round trip and a review, and
the kernel's owner loses the context the next step needs.

- **Orchestrator of a larger build** (such as `top-add-new-arch`). Take the
  reuse verdict below, then delegate each kernel whole. The brief names the
  kernel, the verdict and target kind/backend, its measured share and capture
  evidence, and the decisions that must come back to you. Review once, at the
  kernel's closeout: read the diff and the evidence and spot-check the claims
  that matter.
- **Kernel owner** (an implementer handed the kernel, or the only agent). Run
  this skill end to end. Check each step against the child skills' "Things to
  verify" lists before moving on, and stop at the first check that fails.
  Return before the closeout only for a decision that is not yours: one the child skills reserve for the user, or
  anything that would change the verdict you were given.

The child skills speak to "the orchestrator" and say "ask the implementer".
When you own the kernel, do those steps yourself.

## Several kernels at once

A split table usually names more than one *new dedicated kernel*. They are
independent work — different kinds, different runner files, different
`profile.db` tables — so run them in parallel, **in batches of 3**: launch three
subagents, wait for that batch to close out, then launch the next three. Who
fans out depends on the role you are in:

- **Only agent** (nobody above you, nobody below). Launch one subagent per
  kernel, three at a time; each runs this skill end to end for its own kernel.
  Do not walk the list yourself.
- **Orchestrator.** Hand over **three whole kernels per brief**, and issue the
  next brief after that batch closes out. One kernel per brief serializes work
  that has no ordering constraint; the whole table in one brief hands down a
  queue the implementer has to bound anyway.
- **Implementer handed a batch.** Fan out — one subagent per kernel, all three
  at once, each running this skill end to end. Receiving several kernels is not
  an instruction to do them in order.

Three, not all of them: each agent wants a GPU for its profiling submission and
its own worktree build, so an unbounded fan-out contends for both and you get
measurements taken on a busy card. Three also keeps the batch small enough to
verify before the next one starts.

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

End-to-end kernel work is complete only when both child skills have produced
their closeout evidence, or when the user explicitly asks for only the Python
side or only the Rust side.
