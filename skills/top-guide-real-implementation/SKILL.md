---
name: top-guide-real-implementation
description: >-
  Use as the top entry point when the user wants to guide or optimize a real ML
  serving implementation against a VibeSim-grounded performance target. Runs a
  comparable deployment simulation to obtain predicted throughput and kernel
  breakdown artifacts, first adding fresh model/architecture support through
  the model-exploration and new-architecture workflows when necessary. Then
  asks the user for benchmark and profiler evidence from another serving
  framework, diagnoses where that implementation loses time relative to the
  simulation, and recommends validated experiments to close the gap. This is
  reverse alignment: hold the grounded simulation target as the optimization
  reference and guide the external implementation toward it. It is not the
  VibeSim-to-vLLM validation workflow.
---

# Top Guide Real Implementation

## Core direction — reverse alignment

**VibeSim is the optimization reference; the external serving framework is the
implementation being improved.** The direction is always:

`grounded VibeSim simulation → user-provided framework measurements → framework fixes`

This is not VibeSim-to-vLLM alignment. Do not invoke `operate-run-alignment` or
`top-align-with-framework`, and do not start by calibrating VibeSim to reproduce
framework overhead. First execute VibeSim to establish the target. Then ask the
user for real benchmark and profiler artifacts, explain why the external
framework falls short, and guide its implementation toward the target.

Keep simulated predictions, measured implementation results, and derived
comparisons separate throughout the workflow. Never fill an evidence gap with
an estimate.

## Step 1 — Lock the comparison contract

Record the exact comparison tuple before running anything:

- model/checkpoint, dtype and quantization;
- GPU model, count, topology, and parallelism;
- serving framework and implementation revision;
- scheduler, batching, KV-cache, and backend settings;
- arrival process or request rate, prompt/output-length distributions, warmup,
  duration, and completion policy;
- throughput and latency definitions.

Do not compare runs whose workload or metric semantics differ. Ask for a missing
choice when it materially changes the result. If the real implementation already
has a fixed benchmark, make the simulation reproduce that benchmark rather than
silently choosing a convenient preset.

## Step 2 — Establish the simulation-grounded target

First determine whether VibeSim already supports the exact model architecture,
kernel semantics, deployment, and target GPU required by the comparison
contract. Do not force a nearby model tag, silently drop an unsupported
operation, or approximate a new architecture with a superficially similar one.

If model facts or the support decision are unclear, route to
`top-explore-models` to resolve the exact checkpoint and architecture from
grounded evidence. If VibeSim lacks the architecture, enter
`top-add-new-arch`. That umbrella workflow reuses the exploration evidence,
splits the forward pass, adds or reuses measured L1 kernels, composes L2–L4,
wires timing-predict and deployment dispatch, and validates representative
CostTrees before attempting a DES run. Do not skip its timing-predict validation
or replace missing L1 measurements with guessed costs.

Only after the requested model and deployment are fully supported, route the
deployment run to `operate-run-simulation`. A dry run, source reading, hardware
roofline, or offline arithmetic is not a throughput result: require a completed
discrete-event simulation and its analyzer artifacts.

Report the target as **simulation-predicted throughput**, even if the user calls
it theoretical throughput. Read the value from the run's throughput report; do
not recompute it from guessed iteration latency. Report its exact workload and
configuration beside it.

Read the applicable analyzer outputs rather than inventing a breakdown:

- `throughput` for the deployment throughput target;
- `kernel-time-share` for the simulated kernel-time breakdown;
- `kernel-throughput` for achieved modeled TFLOP/s or GB/s by CostTree location;
- `optimality` for artifact-backed idle, imbalance, batching, communication, and
  profiled-to-hardware gaps;
- `iter_breakdown.ans` or exact iteration details when representative
  per-iteration CostTree evidence is needed.

The `optimality` subject reports lower-bound GPU-time rungs and an optimality
ratio. Do not relabel a GPU-time lower bound or a hardware roofline as achieved
throughput, and do not convert it into a throughput number unless an analyzer
artifact explicitly provides that metric.

If the run still cannot complete because the architecture, kernel backend, GPU
profile rows, deployment wiring, or analyzer subject is unsupported, return to
the owning implementation workflow and finish that coverage before reporting a
target. Do not substitute an estimate. Cite the preset snapshot, launcher
command, log directory, and every report used.

## Step 3 — Obtain real implementation evidence

After establishing the simulation target, inventory any measured artifacts the
user has already supplied. If they are absent or incomplete, stop and ask the
user for a comparable profiling bundle from the external framework. Do not
launch `operate-run-alignment`; this skill consumes user-provided measurements
and does not run a VibeSim-to-vLLM validation experiment.

Require enough evidence to reproduce and interpret the measurement:

- benchmark command, implementation commit, full serving configuration, and
  request trace or workload generator settings;
- measured throughput, TTFT/TPOT, completed/requested counts, warmup interval,
  and raw benchmark output;
- an NSYS capture (`.nsys-rep`) plus any exported SQLite, CSV, or `nsys stats`
  reports needed to inspect it, including CUDA graph and NVTX information when
  applicable;
- GPU model/count/topology plus clock, power, thermal, and competing-process
  context for the captured interval.

Treat reported benchmark numbers and profiler timelines as **measured results**.
An NSYS file without the matching benchmark configuration is not comparable
evidence. If the bundle is incomplete, return the simulation baseline plus an
exact artifact request; do not claim where the implementation is slow.

## Step 4 — Diagnose the gap

First verify the Step 1 tuple on both sides. Then compute the headline headroom
only from artifact-backed values:

`headroom_fraction = (simulated_throughput - measured_throughput) / simulated_throughput`

Label this as derived from the named simulation and benchmark artifacts. It is a
comparison, not a new measurement and not proof that all headroom is attainable.

Treat the simulation as the optimization reference and diagnose the external
framework from the user-provided benchmark and profiler artifacts. Do not route
to `top-align-with-framework`; that skill judges simulator fidelity rather than
guiding another framework toward a simulation target. Diagnose in this order:

1. Check per-iteration workload and batch-shape equivalence.
2. Match measured framework operations to simulated CostTree slots by phase,
   batch shape, ordered neighbors, and kernel semantics. Mark uncertain matches
   unresolved rather than forcing them.
3. Separate concentrated kernel/backend/shape gaps from broad clock or thermal
   shifts.
4. Compare measured kernel-busy time with GPU-cycle time to expose launch,
   synchronization, CPU scheduling, or other host bubbles.
5. Explain remaining end-to-end throughput, TTFT, and TPOT gaps from scheduling,
   queueing, batching, memory pressure, or unmodeled framework work.

Do not tune VibeSim to absorb avoidable framework overhead. A grounded simulation
plus a slower measured run is framework optimization headroom. If the evidence
instead exposes a missing or mismodeled simulator operation, pause reverse
alignment, fix and rerun the simulation target, then resume; never hide a
simulator defect by attributing it to the framework.

## Step 5 — Recommend and validate fixes

Prioritize recommendations by observed time share and demonstrated gap. For each
recommendation, provide:

- the measured and simulated artifact evidence;
- the diagnosed cause and confidence level;
- the concrete implementation change or controlled experiment;
- the metric that should move and the validation command or capture to rerun.

Keep hypotheses explicitly labeled. Do not promise a throughput gain from a
time-share calculation. Quantify an expected gain only by rerunning VibeSim with
a supported configuration change or by measuring the changed implementation.

After each implementation change, rerun the same real benchmark and profiler
capture. Rerun VibeSim when the modeled deployment, scheduler policy,
parallelism, kernel/backend choice, measured kernel-cost coverage, or workload
changed. Preserve both artifact sets and update the gap table instead of
overwriting the baseline.

## Completion report

Return:

- the comparison tuple and any caveats;
- simulation-predicted throughput with source paths;
- simulated kernel breakdown and optimality findings with source paths;
- measured throughput/latency and NSYS provenance;
- the artifact-derived gap, diagnosed causes, and unmapped coverage;
- prioritized fixes with confidence and a validation plan.

If either side is missing, clearly mark the workflow incomplete and request the
specific next artifact. Never present an estimate as the missing side.
