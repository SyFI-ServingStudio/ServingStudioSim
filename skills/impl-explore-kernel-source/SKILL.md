---
name: impl-explore-kernel-source
description: Use when researching and planning the source/wrapper for a new MLSim Python profiling backend before implementation. Finds a real framework/kernel implementation, maps it to an existing or proposed KernelArgs schema, records environment and capability constraints, and produces a wrapper plan. Does not edit code or register KernelProfilerSpec rows.
---

# Impl Explore Kernel Source

You are the implementer for source exploration only. Do not edit code. Produce a
plan that the orchestrator can verify and hand to `impl-register-kernel`.

## Inputs

The orchestrator should provide:

- kernel kind and whether this is a new backend or a possible new kind;
- intended backend name;
- expected args fields or the existing kind whose args schema must be reused;
- target framework preference if any.

If any of these are missing, state the missing decision instead of guessing.

## Search Targets

Find a real implementation source before recommending registration work. Prefer
primary docs and source for the framework. When the user has not named a source,
check likely serving/kernel frameworks such as:

- vLLM
- SGLang
- TensorRT / TensorRT-LLM
- FlashInfer

For each plausible source, record the exact module/function/class path, the
version or commit assumption if known, and why it does or does not fit the
requested kernel.

Do not stop at a framework name or high-level documentation. Read the pointed
implementation code closely enough to understand the callable signature, tensor
layout assumptions, setup/planning calls, required workspace/state, and failure
conditions. If the implementation code is not accessible, return that as a
blocker instead of proposing a wrapper.

## Wrapper Plan

The output plan must be concrete enough for `impl-register-kernel` to implement:

- selected framework source and exact call path to wrap;
- source file or URL plus symbol name that the orchestrator can open and verify;
- package or profiling environment required, including whether an existing
  `profiling.exec.env` entry is enough or a new env decision is needed;
- input mapping from `KernelArgs` fields to framework call arguments;
- dtype, GPU, architecture, and shape constraints;
- backend support matrix for registration: compute dtype, KV/cache dtype if
  relevant, output dtype assumptions, GPU names or architecture gates, and
  package-version constraints;
- unsupported cases and how the runner should fail them;
- expected `BackendSupport` axes: compute dtype, KV dtype if relevant, and GPU
  gates if relevant;
- suggested timing method by mirroring the closest current runner family;
- representative smoke spec for `uv run python -m profiling run ... --db /tmp/...`.

If the source requires changing the existing args schema, say so explicitly and
return to the orchestrator for a kind/schema decision. Do not hide
backend-specific fields inside a runner.

## Things To Verify

Before returning, verify the plan against the current `profiling/` implementation:

- read `profiling/README.md`;
- read the target kind's `profiling/kernels/<kind>.py` if it exists;
- read the closest runner family under `profiling/runners/`;
- read the pointed framework implementation code and confirm the proposed
  wrapper call matches the actual signature;
- confirm whether `backend` is only routing metadata and not a runner kwarg;
- confirm whether the intended smoke can use the public CLI:
  `uv run python -m profiling run <kind> --backend <backend> --db /tmp/<kind>_<backend>_smoke.db --spec '<json spec>' --json`.

Return blockers instead of a fake plan when no source, environment, compatible
GPU, or compatible schema is available.
