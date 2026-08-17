---
name: dev-build-serving-repetitive-unit
description: >-
  Build a reduced-depth model that keeps the production execution path for
  kernel and layer-timing work. Not for scheduling, capacity, or E2E results.
---

# Build A Complete Reduced Serving Model

Build the smallest complete model variant that can exercise representative
blocks through the production serving path. Reduction changes selected blocks,
materialized weights, and proportional runtime state—not the surrounding
implementation.

## Scope

Use it for:

- kernel and immediate producer/consumer-boundary correctness and performance;
- layer-level attention, FFN, routing, and communication investigation;
- backend, layout, graph, fallback, and partial-loading integration.

Do not use it to evaluate scheduling, admission, continuous batching,
concurrency, cache policy/capacity, queueing, TTFT/TPOT, throughput, saturation,
or workload speedup. Those require the full model.

## Preserve the production path

Full and reduced variants must share the production model/registry, server and
request protocol, engine and batch entry, distributed ownership, loading
infrastructure, runtime-state implementation, backends, graph policy, sampling,
profiling schema, and lifecycle.

Only explicit model-execution choices may differ. The request frontend,
readiness checks, profiler, evidence parser, and artifact schema must not branch
on “repetitive unit.” Copied model code, a parallel loader, a manually recreated
runtime, or an experiment-only canonical runner invalidates the evidence. A
debug probe may use such a path only when clearly labeled and never for
promotion.

## Represent reduction without changing model identity

Keep checkpoint and architecture identity immutable. Represent the selected
blocks in the framework's native configuration and type system; this skill does
not prescribe an API or schema.

Preserve an explicit mapping between compact runtime block identity and original
checkpoint block identity. Runtime identity owns execution order and compact
state; source identity owns checkpoint lookup, provenance, and source-indexed
metadata. Persist enough information to reconstruct that mapping.

A homogeneous model needs its real boundaries plus one representative block. A
heterogeneous model needs at least one block from every distinct class.

## Share construction and partial loading

The production construction and loading planners must consume the reduced plan.
Both variants use the same manifest validation, shard reading, slicing, tensor
assignment, and readiness path. Partial loading selects fewer tensors and avoids
irrelevant shards; it is not a second loader.

Record selected sources, opened shards, materialized bytes, checkpoint revision,
runtime/source mapping, and cold/warm cache state. Separate startup, distributed
initialization, planning, checkpoint I/O, backend preparation, graph capture,
and measured execution time.

## Run and validate

Launch through the normal server entry and native-token request protocol. Use
the standard launcher-managed capture. Measure production graph paths with graph
replay; eager and synthetic inputs are debugging-only unless production uses
them.

Before accepting timing, verify:

1. normal startup, readiness, request, sampling, and teardown;
2. tensor shape/layout/aliasing and distributed ownership;
3. complete weight and auxiliary-metadata mapping;
4. correctly sized runtime state;
5. finite outputs and applicable routing invariants;
6. production backend/fallback selection and repeated graph replay;
7. several post-warmup executions, not one cold invocation.

Use a deterministic sequence pack as a regression sentinel and compare against
a reference execution of the same reduced plan—not the full model's sequence.

## Interpret only layer-level evidence

The reduced model validates production-context kernel, boundary, loading,
backend, and graph behavior. Its outputs and data-dependent routing need not
represent deep full-model activations; use a separately labeled captured-input
probe when that fidelity is required.

Never scale one block's timing across model depth without proving block-class,
shape, routing, backend, and communication equivalence. Never scale model-boundary
work by repeated-block count. Promote candidates through the parent workflow's
full-model integration and canonical workload gates.
