---
name: top-compose-real-framework-from-sim
description: >-
  Use by the orchestrator as the top entry point when VibeSim should actively
  build a real LLM-serving framework from scratch or optimize an existing one
  from simulation evidence, using trusted correctness and benchmark utilities.
---

THIS SKILL IS MAINLY FOR ORCHESTRATOR

# Top Compose Real Framework From Sim

## Overall Goal

Turn simulation into a real framework implementation. The user may give you an
optimization goal. Explore the design space inside the simulator, then use its
results to guide the real-world implementation.

Advance the simulation and real framework in a tick-tock way: improve the
simulated deployment first, then update the real framework to catch up. Return
to the simulator afterward to explore whether it can do even better.

The user may provide accuracy-checking and benchmark utilities. Do not modify
them; treat them as trusted evaluation contracts.

Implement framework code under `/framework/name`. That directory may initially
be empty: do not assume that a repository, runnable server, or baseline
implementation already exists. When it is empty, compose the first
implementation from VibeSim evidence and the trusted evaluation contracts. When
it already contains a runnable implementation, improve it in place.

VibeSim is the optimization reference; the real framework is the implementation
being improved. Keep simulation predictions, measured framework results, and
comparisons derived from named artifacts separate. Never fill an evidence gap
with an estimate, and do not use `top-align-with-framework` or
`operate-run-alignment` for this direction.

## Operating model

The **Orchestrator** guides the whole process, including planning and evaluating
results. It may delegate one frozen code-writing brief to the Implementer, but
it keeps every decision and every evidence-producing action.

| Work | Owner |
| --- | --- |
| Lock comparison contract | Orchestrator |
| Run simulation/timing-predict and inspect artifacts | Orchestrator |
| Select exactly one trial | Orchestrator |
| Write the real-framework code for that trial | Implementer |
| Review the diff and run trusted gates | Orchestrator |
| Run the canonical benchmark/profile | Orchestrator |
| Explain the result and accept/reject the trial | Orchestrator |
| Record progress periodically in Markdown files | Orchestrator |

If no code-writing agent is available, stop and ask for one. Do not silently
collapse the role boundary by having the Orchestrator write the implementation.

## Concrete Steps

### Step 1 — Lock the real comparison contract

Make sure you are confident with the following:

- model/checkpoint revision, dtype, quantization, and KV dtype;
- target path and whether it is empty or contains an existing implementation;
- repository revision and dirty-tree state only when a repository exists;
- GPU type/count/topology and TP/EP/CP/PP;
- scheduler, batching, KV layout, and kernel/backend choices;
- arrival process, request-rate or concurrency policy, input/output
  distributions, warmup, duration, and completion policy;
- exact throughput, TTFT, TPOT, and correctness definitions;
- trusted accuracy, benchmark, and profiler commands.

Do not compare runs whose workloads or metric semantics differ. Make the
simulation reproduce the trusted benchmark contract rather than choosing a
convenient preset. Ask the user only when a missing choice materially changes
the experiment. An empty target directory is a valid starting condition, not a
missing input.

### Step 2 — Establish the VibeSim target

Work directly through existing repo-local VibeSim skills when necessary:

- use `top-explore-models` when checkpoint facts are unclear;
- use `top-add-new-arch` when VibeSim lacks the model architecture;
- use `top-add-kernel` when a required kernel/backend is missing;
- use `operate-run-simulation` for a serving-workload throughput target;
- use `operate-run-timing-predict` only for explicit fixed-shape building-block
  comparisons.

First use the simulator to find a strong solution and establish a performance
target. Confirm that VibeSim supports the exact architecture, kernel semantics,
deployment, and GPU. Do not force a nearby model, omit unsupported work, or
replace missing measurements with guessed costs.

Inspect throughput, kernel-time-share, kernel-throughput, optimality, memory/KV
capacity, iteration breakdown, and the actual CostTree. Require a completed
simulation and its analyzer artifacts; a dry run, roofline, or offline
calculation is not a throughput result. Cite the preset snapshot, launcher
command, log directory, and reports used.

When the simulator itself is wrong or incomplete, pause real-framework work,
repair VibeSim through its owning top/orchestrator/impl skill, rerun the target,
and only then resume.

### Step 3 — Build or advance the real framework

After obtaining a simulated result, inspect `/framework/name`.

- **Empty or non-runnable target** — select one coherent bootstrap trial that
  creates a real, runnable implementation for the exact trusted correctness and
  benchmark contracts. Do not request a pre-existing codebase, commit,
  benchmark result, or profiler trace, and do not use stubs or
  benchmark-specific shortcuts. The first correctness-passing, benchmarkable
  implementation becomes the real-framework baseline.
- **Existing runnable target** — before changing code, run the trusted
  correctness check, benchmark, and profiler on the unchanged implementation.
  Preserve the command, revision, configuration, raw output,
  completed/requested counts, throughput, TTFT/TPOT, and profiler provenance.

Use simulator analyzer artifacts for both cases. Use measured framework
profiling to diagnose a performance gap only after a runnable baseline exists.

Check GPU idleness before launching the benchmark and ensure that no other
process is using the target GPU. Otherwise, the measured gap may be a scheduling
artifact rather than a cost-model gap.

Consider the following aspects when aligning the performance:

- **Fundamental improvement** — the *modeled work itself* is off: a real kernel/op is
  slower than the sim's slot cost (different backend/implementation, a shape/quant
  edge), or the batch shapes / parallelism the server actually runs differ from what
  the sim assumed. Close it by lowering the server's fundamental cost to match the
  sim.

- **Overhead reduction** — the kernels already match the sim's slot costs, but the
  measured wallclock is larger because of **implementation-dependent overhead** the
  sim does not model: CPU launch overhead, gaps between kernels, no CUDA-graph
  capture, no async-scheduler overlap, Python overhead. Close it with
  implementation-time techniques that raise the GPU **duty cycle**. These do **not**
  move the sim breakdown, so alignment is the only place they show up.

For an existing baseline, diagnose the gap in order: verify workload and
batch-shape equivalence; match measured operations to CostTree slots; separate
localized kernel/backend gaps from clock or thermal shifts; compare kernel-busy
time with GPU-cycle time; then explain remaining scheduling, queueing, batching,
memory, and framework overhead. Leave uncertain operation matches unresolved
rather than forcing them.

Select exactly one carefully scoped trial: either the bootstrap trial for an
empty target or one optimization trial for an existing baseline. Freeze its
hypothesis, code scope, invariants, expected observable effect, and trusted
validation commands. Delegate only that implementation work to the Implementer,
using `dev-llm-serving` for relevant techniques and source maps. The Orchestrator
must inspect the diff and run the trusted checks itself.

Correctness is mandatory. Evaluate the trial not only by absolute speedup, but
also by whether its proposal was technically sound, whether the implementation
tested that proposal, and whether the explanation fits the measured result.
Preserve the VibeSim target and every available real-framework baseline, and
record the decision before starting another trial.

## Step 4 — Explore more simulation possibilities

After the real result is close enough to the simulation, or the remaining
fundamental gap is explained, return to the simulator to explore more of the
design space.

When several directions are viable, prefer the one with the best
**effort-to-leverage ratio** — cheap to try and easy to attribute. The levels below
are ordered by *scope* (local kernel → global topology), and attribution generally
gets harder as scope widens — but **effort is not monotonic in the level number**: a
scheduling or batching *policy* knob (often just a config flag) can be the cheapest
change of all, easier than swapping a kernel, even though it sits at a higher level.
So weigh difficulty against leverage per candidate; don't march rigidly up the
numbering. Each level names the matching `dev-llm-serving` tier for the technique
menu: **this skill names the level, `dev-llm-serving` names the techniques.**

1. **Kernel** — a single hot kernel dominates. Swap it for a faster implementation or
   backend. First **look for a public solution** (a faster attention / GEMM / norm
   kernel, a backend such as FlashInfer / CUTLASS, a fused variant); if one exists,
   **point the simulator at it** — ask VibeSim to model it, or to implement it with
   your guidance. You can also **ask VibeSim directly for a kernel suggestion** — it
   knows its own kernel catalog and what is fast on the target GPU. →
   `dev-llm-serving`: backends / hardware / algorithms.
2. **Operation** — the cost is in *how ops are wired*, not one kernel. Fuse operations
   (norm+rope+attention, GEMM+bias+activation), drop redundant work, cut memory
   round-trips so several small kernels become one. → `dev-llm-serving`: algorithms /
   engines.
3. **Architecture / parallelism** — the compute↔communication balance is off. Change
   the parallelism layout (TP / EP / CP / PP) and how the model shards across GPUs. →
   `dev-llm-serving`: frameworks / hardware.
4. **Worker scheduling / KV management** — the kernels are fine but the *sequence* of
   work isn't. Scheduler policy (continuous batching, chunked prefill), KV-cache
   management (paged, block size, eviction), batch composition. → `dev-llm-serving`:
   engines / frameworks.
5. **Deployment** — the single-worker shape is exhausted; restructure the topology.
   Prefill/decode (PD) disaggregation, disaggregated / multi-node serving, replica
   routing. → `dev-llm-serving`: frameworks.

Inside the level the breakdown points at, rank candidates by **leverage** = (fraction
of predicted time the change targets) × (fraction of that it could plausibly remove):
a change on a 5% slice caps at a 5% win; the headroom is on the dominant slice. Pick
the single highest-leverage change per iteration (one attributable change keeps the
later alignment's gap explainable), and confirm in the grounded sim that it is
**feasible across the *whole* analyzer output** — throughput **and** KV capacity, iter
breakdown, optimality — not just the headline number. A candidate that improves the
headline but that the breakdown shows is infeasible (e.g. exceeds KV capacity) is not
a valid pick.

When the simulation finds another feasible improvement, return to Step 3 to
advance the real framework with one new trial.

## Completion

Report:

- the real comparison contract and starting state (empty or existing revision);
- VibeSim target and CostTree/analyzer provenance;
- each frozen trial and implementation diff;
- trusted correctness, benchmark, and profiler artifacts;
- predicted versus measured results with diagnosed residuals;
- retained/rejected changes and the remaining evidence gap.

## Neighboring workflows

- Real serving implementation techniques and source maps:
  `dev-llm-serving`.
