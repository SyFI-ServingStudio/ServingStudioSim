---
name: orchestrator-add-kernel-to-python-profile
description: Use when orchestrating Python-side ServingStudio Sim L1 profiling work for a new kernel kind or a new backend of an existing kind. This skill is for the coordinator who writes implementer briefs, defines completion checks, and sequences Torch reference work before framework backends. It does not cover Rust timing/cache wiring.
---

# Orchestrator Add Kernel To Python Profile

You are the orchestrator, not the implementer. Your job is to turn a user's
kernel request into one or more clear implementation tasks for the Python
profiling side under `profiling/`, with enough completion criteria that you can
verify whether the step is done.

This skill deliberately stops before Rust timing/cache wiring. Any Rust
`KernelSpec`, cache, bridge, or simulator integration belongs to a later skill.

## Verification Ownership

The implementer may be another agent, but the verifier is the orchestrator. Ask
the implementer for evidence, then independently check it before accepting the
step. Do not treat "the implementer says it works" as completion.

For each assigned task, state what evidence must come back and what you will
check yourself: file diffs, imports, registry entries, generated facade symbols,
focused tests, dry-run output, environment notes, and unresolved assumptions.

## Grounding

Before issuing tasks, anchor the request in the project rules:

- Current `profiling/` code first. In particular read `profiling/README.md`,
  `profiling/kernels/__init__.py`, the closest existing `profiling/kernels/*.py`
  file, and the closest runner under `profiling/runners/`.
- Use `doc/architecture.md` and `doc/detailed_design/L1.md` as
  background, but do not let stale doc wording override the implementation.
- For concrete registration/file rules, delegate to `impl-register-kernel`.

If the requested name, location, args shape, or behavior is not supported by the
docs or current code conventions, ask the user to decide before assigning
implementation.

## First Decide User Intent

Classify the request before doing anything else:

- **Brand-new kernel kind**: the user is asking for a new operation family, not
  just another implementation of an operation ServingStudio Sim already knows how to
  describe. For example, if ServingStudio Sim currently has `rmsnorm` and `single_gemm`, and
  the user asks to add convolution, convolution is a new kernel kind. It needs a
  new semantic contract, args shape, Torch reference, runner, and registry
  entry.
- **New backend for an existing kind**: the operation family already exists and
  the task is to add another implementation for the same semantic contract. For
  example, adding a TensorRT or FlashInfer backend for an existing attention or
  norm kind is a new backend, not a new kind, unless the requested behavior
  changes the operation contract.

If the user asks for both, split it into ordered tasks. The new kind lands first
with its production backend and an independent Torch correctness oracle, then
additional backends reuse that established kind and args schema. Register a
Torch backend only when the production operation is itself a public Torch
callable worth timing.

Before accepting “brand-new kind”, compare the nearest existing contract across
operation semantics, args meaning, dtype/layout, production callable, and
logical launch boundary. Source ambiguity requires a matched-shape A/B with
identical logical I/O. A backend may change implementation, never field meaning.

## Path A: Brand-New Kernel Kind

Start by grounding what operation the kernel actually represents.

**Task A.1 — Operation grounding.** Ask the user to point to a real usage of
the kernel in a particular model, model family, paper, framework, or repository
when possible. Use that source to understand what the kernel computes, which
inputs are semantic, and which dimensions are just implementation details.
Also determine the dtype contract the user expects: compute/activation dtype,
KV/cache dtype if relevant, output dtype, and which dtype combinations must be
supported in the first implementation versus deferred.

If the user already gives a detailed kernel description, do a source/search pass
against primary sources when needed, then restate the inferred operation
contract to the user before assigning implementation. Do not proceed to the
Torch reference until the user has confirmed that this is the operation they
want profiled.

Things to verify:

- there is a concrete model/framework/source anchor, or the user explicitly
  confirms the detailed operation contract;
- the intended inputs, outputs, dtype behavior, and unsupported cases are stated
  in model terms rather than backend implementation terms;
- expected dtype combinations are explicit, including compute dtype, KV/cache
  dtype when relevant, output dtype, and first-pass support vs deferred support;
- any ambiguity that could change the `KernelArgs` schema is resolved with the
  user before implementation.

**Task A.2 — Torch reference.** Ask the implementer to write a runnable Torch
reference for the exact operation and shape contract. It should be small,
direct, and independent of any specialized framework.

Pass the Task A.1 investigation result into this brief: model/source anchor,
operation contract, intended inputs/outputs, dtype behavior, unsupported cases,
and any schema-sensitive decisions. The implementer should use that context to
find or implement the Torch reference. If A.1 identified a real model or model
family, suggest searching the relevant transformer package or Hugging Face
modeling code for an existing PyTorch implementation. Use
`dev-lookup-transformers-model` when the source should come from the local
Transformers install, then extract the smallest runnable reference that
preserves the model semantics.

Things to verify:

- the reference runs with representative inputs;
- outputs have the expected shape and dtype behavior;
- numerical expectations and tolerance are stated;
- edge cases and unsupported cases are explicit.

**Task A.3 — Production profiling backend.** Ask the implementer to wrap the
production public callable by using `dev-explore-kernel` and
`impl-register-kernel`. The brief should name the kernel kind, backend, args
fields, closest existing runner to mirror, and the Torch oracle from Task A.2.
Do not vendor specialized kernel source into the profiler.

Things to verify:

- the implementer ran `impl-register-kernel` or followed its current
  `profiling/` contracts;
- `uv run python -m profiling list --json` shows the `(kind, backend)` entry;
- the generated perf API symbol resolves, e.g.
  `uv run python -c "from profiling import perf_api; assert hasattr(perf_api, 'get_<kind>_times')"`;
- a real profiling smoke succeeds through the public CLI on a representative
  spec, writing only a temporary DB, e.g.
  `uv run python -m profiling run <kind> --backend <backend> --db "$TMPDIR/<kind>_<backend>_smoke.db" --spec '<json spec>' --json`;
- the production callable is what the runner measures, while the Task A.2
  reference checks it outside timing;
- the implementer produced tests and smoke evidence, without writing shared
  `profiling/profile.db` unless the user explicitly authorized it.

Once the production backend is in place, ask whether additional implementations
are needed. Use Path B for each added backend; do not repeat Path A or change the
kind's established semantics.

## Path B: New Backend For An Existing Kind

First verify that the existing kind's args schema actually describes the backend
the user wants. Do not add backend-specific fields casually. If the backend needs
new semantic dimensions, decide whether the kind is underspecified and return to
Path A or ask the user for a schema decision.

**Task B.1 — Framework source and wrapper plan.** Ask the implementer to identify
the framework implementation to wrap by using `dev-explore-kernel`.

Things to verify:

- the source is specific enough to implement from: exact function/class path,
  required version or environment, and input mapping;
- you have read the pointed implementation code, not just the docs or the
  implementer's summary, and the call signature/input mapping appears plausible;
- the backend fits the existing kind's args schema without hidden backend-only
  fields;
- backend support constraints are explicit: compute dtype, KV/cache dtype if
  relevant, output dtype assumptions, GPU names or architecture gates, and any
  package-version constraints;
- unsupported dtype/GPU/shape cases are stated;
- the plan says how the wrapper will be smoke-tested through the public
  profiling CLI.

If Task B.1 finds multiple viable sources/backends, summarize the tradeoffs and
ask the user which one to implement first before assigning Task B.2.

**Task B.2 — Backend runnable and correctness check.** Before registration, ask
the implementer to run the selected implementation directly in its required
environment and compare it against the Torch reference from Task A.2 on
representative shapes.

Things to verify:

- the selected implementation actually runs outside the profiling registry;
- outputs align with the Torch reference under stated tolerances;
- dtype/layout conversions needed for comparison are explicit;
- the claimed backend support matrix is checked on available hardware/dtypes,
  or untested entries are clearly marked as assumptions;
- unsupported shapes or dtypes fail clearly;
- you independently review the correctness evidence and, when feasible, rerun
  the provided check before allowing registration work to proceed.

**Task B.3 — Backend registration.** Ask the implementer to add the backend
through `impl-register-kernel`, reusing the existing kind's args schema.

Things to verify:

- the framework kernel can be invoked through a thin runner wrapper;
- environment selection is declared through registry metadata, not ad hoc runner
  environment mutation;
- `BackendSupport` matches the verified support matrix: compute dtype, KV/cache
  dtype if relevant, and GPU/architecture gates;
- `uv run python -m profiling list --json` shows the new backend under the
  existing kind;
- the existing generated perf API symbol still resolves for that kind;
- a real profiling smoke succeeds through the public CLI on a representative
  spec for that backend, writing only a temporary DB, e.g.
  `uv run python -m profiling run <kind> --backend <backend> --db "$TMPDIR/<kind>_<backend>_smoke.db" --spec '<json spec>' --json`;
- focused tests and smoke evidence cover the new backend. If no compatible
  GPU/environment is available, treat the task as not fully verified and return
  the exact blocker instead of accepting it as done.

## How To Issue Implementer Tasks

Keep briefs short and concrete. A good task brief names:

- intent classification;
- implementation skill to use, normally `impl-register-kernel` after the Torch
  reference or framework source has been identified;
- exact kernel kind, backend name, and expected args fields;
- source of truth: docs anchors, existing reference files, and framework source
  or documentation to mirror;
- implementation files the agent is allowed to touch on the Python side;
- completion evidence the implementer must return: commands run, test output,
  dry-run result, environment notes, and unresolved assumptions;
- verification checks you will perform before accepting the task.

Do not ask the implementer to wire Rust timing, cache selection, simulator
bridge code, or L2/L3/L4 consumers. End the task with a **Python-to-Rust
handoff** section for the later role that will add the Rust `KernelSpec` /
bridge/cache wiring. This is not Rust implementation work; it is a short list
of facts the Python work established: `KIND` / facade stem, backend strings,
`KernelArgs` fields and order, metric family, dtype/GPU capability axes, a
representative smoke spec, and shape or dtype constraints discovered while
validating the runner.
