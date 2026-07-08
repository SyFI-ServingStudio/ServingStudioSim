---
name: dev-explore-kernel
description: Use to search the kernel/serving ecosystem (vLLM, SGLang, TensorRT-LLM, FlashInfer, flash-attn, cutlass) for how an operation is really implemented — the exact call path, the launch granularity (one fused kernel or several), and the constraints. Read-only. Reports findings and, when a source is chosen to wrap, a concrete wrapper plan for impl-register-kernel. Does NOT decide the op's VibeSim kernel home (that is top-split-model-into-kernels) and does not edit code or register KernelProfilerSpec rows.
---

# Dev Explore Kernel

Read-only search. Given an operation, find how it is really issued as GPU kernels
in the serving/kernel ecosystem and report the facts. You do not decide where the
op lands in VibeSim's vocabulary — `top-split-model-into-kernels` makes that
boundary verdict from your findings, and the add-kernel orchestrator uses your
wrapper plan. Do not edit code or register anything.

Kernel-side sibling of `dev-lookup-transformers-model`: that skill reports what
math an op computes; you report how that math runs as kernels.

## Inputs

Whoever calls you should provide:

- the operation and where it is used (model / family / paper / framework);
- whether a kernel kind / args schema already exists that a wrap must reuse;
- intended backend name and framework preference, if any;
- dtype / GPU / shape context that constrains the search.

If a needed decision is missing, state it instead of guessing.

## Search Targets

Prefer primary source and docs. When no source is named, check the likely
serving/kernel frameworks:

- vLLM, SGLang, TensorRT / TensorRT-LLM
- FlashInfer, FlashAttention, cutlass / CUTLASS-based libs

Read the pointed code closely enough to know the callable signature, tensor
layout, planning/setup calls, required workspace/state, and failure conditions —
not just the framework name or high-level docs. Record the exact
module/function/class path and the version/commit assumption. If the
implementation code is not accessible, return that as a blocker.

## What To Report

For each found implementation:

- exact call path (module/function/class) + source file/URL + symbol to open;
- **launch granularity** — is the op issued as one fused kernel, or several
  separate launches? Which neighboring math is fused in (RoPE, bias, scale,
  activation)? This is the key fact the boundary decision needs — report it, do
  not judge it;
- fit to the existing or proposed args schema, and any schema-sensitive fields;
- dtype / GPU / architecture / shape constraints, and unsupported cases;
- the package or `profiling.exec.env` the implementation needs.

If no implementation is found, say so plainly — that absence is itself a finding
the boundary decision uses (it usually points to an `elementwise` placeholder).

## Wrapper Plan (only when a source is chosen to wrap)

When the caller has decided to wrap a specific source (add-kernel Task B.1), the
plan must be concrete enough for `impl-register-kernel`:

- selected framework source and exact call path;
- package / `profiling.exec.env` requirement (an existing entry is enough, or a
  new env decision is needed);
- input mapping from `KernelArgs` fields to framework call arguments;
- backend support matrix: compute dtype, KV/cache dtype if relevant, output
  dtype, GPU/architecture gates, package-version constraints;
- unsupported cases and how the runner should fail them;
- suggested timing method by mirroring the closest current runner family;
- a representative smoke spec for
  `uv run python -m profiling run <kind> --backend <backend> --db /tmp/<kind>_<backend>_smoke.db --spec '<json spec>' --json`.

If the source needs a changed args schema, say so and return for a kind/schema
decision. Do not hide backend-specific fields inside a runner.

## Things To Verify

- read `profiling/README.md` and, if it exists, `profiling/kernels/<kind>.py`;
- read the closest runner family under `profiling/runners/`;
- read the pointed framework code and confirm the call signature/input mapping;
- confirm whether `backend` is only routing metadata and not a runner kwarg.

Return blockers instead of a fake plan when no source, environment, compatible
GPU, or compatible schema is available.
