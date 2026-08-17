---
name: top-compose-real-framework-from-sim
description: >-
  Build or optimize a real serving framework through repeated simulation, one measured trial, and the Align framework to VibeSim side of operate-run-alignment.
---

THIS SKILL IS MAINLY FOR ORCHESTRATOR

# Top Compose Real Framework From Sim

## Overall Goal

Turn simulation into a real framework implementation. The user may give you an
optimization goal. Explore the design space inside the simulator, then use its
results to guide the real-world implementation.

The user may provide accuracy-checking and benchmark utilities. Do not modify
them; treat them as trusted evaluation contracts.

Before planning implementation work, identify and record the user's intended
starting mode:

- **Fresh build** — compose a new implementation from VibeSim evidence and the
  trusted evaluation contracts. Do not assume an existing serving framework,
  repository, runnable server, or measured baseline.
- **Existing-code optimization** — use the user-selected codebase as the real
  baseline, preserve its history and local changes, and improve it in place.

Use explicit user wording and the named target to determine the mode. Directory
contents alone do not determine intent: a fresh-build workspace may contain
evaluators or reference assets, while an existing-code task may point to a
codebase elsewhere. If the mode remains ambiguous and choosing one would change
the implementation strategy, ask the user before proceeding.

The starting mode says **where the real baseline comes from**. It does *not* say
what the implementation may be built out of — that is a second, independent
decision made in Step 1 (*Implementation mode*). An empty target implies neither.

Implement framework code under `/framework/name` unless the user selected
another target. Inspect the target only after identifying the mode, then verify
that its actual state is compatible with that mode.

## Direction invariant

**VibeSim is the optimization reference; the real framework is the
implementation being improved.** Keep simulation predictions, measured framework
results, and comparisons derived from named artifacts separate. Never fill an
evidence gap with an estimate. Alignment evidence is reusable in both directions:
it can expose simulator fidelity gaps and real-framework implementation gaps.
Using it here must not reverse the target by copying an accidental real-engine
limitation into VibeSim.

The hard guard, applied every time a real measurement seems to contradict the
simulated target:

> Before changing a simulation because of a real measurement, ask whether an
> **external contract fact** changed or whether the **current implementation
> merely differs**. Only the former may update the comparison contract. A real
> limitation must be fixed in reality, not normalized into the simulation.

The direction is fixed once, at the start of the task, and holds for the whole
loop. **A new measurement never re-opens it.** The realistic failure mode is not
forgetting the rule — it is learning something new mid-loop and quietly
re-deciding, usually in the name of making the comparison "fair". The more a
real-engine limitation looks like an unfair experimental condition (an active
slot cap, an allocated KV budget, missing graph coverage), the more likely it is
precisely the gap this workflow exists to expose. Copying it back into the
simulation destroys the target that justified the work.

## Governing loop — Tick, Tock, Probe

This workflow is a repeated three-beat loop, not a one-pass linear plan:

1. **Tick — VibeSim advances first.** Establish the grounded target, inspect the
   full analyzer evidence, and **freeze exactly one hypothesis** as a trial brief.
2. **Tock — the real framework catches up.** Implement that single change, pass
   trusted correctness, and run the canonical benchmark. **A Tock ends at
   measurement** — no interpretation, no acceptance, no rejection.
3. **Probe — attribute the outcome and decide.** Explain the result from
   artifacts, profile when artifacts are insufficient, add temporary
   instrumentation only when the trace cannot attribute the gap, and close with
   an explicit **retain / reject / inconclusive**.

Then Tick again: feed the Probe's evidence and the remaining headroom into the
next simulation-backed search.

Every plan, progress log, and handoff must name the current Tick, Tock, or Probe
and the cycle number. Never compress multiple cycles into one numbered
implementation plan. A Tock contains exactly one attributable code change, and
**a cycle is complete only once its Probe has recorded a decision**. The next
framework change cannot begin before that.

## Operating model

The **Orchestrator** guides the whole process, including planning and evaluating
results. It may delegate one frozen code-writing brief to the Implementer, but
it keeps every decision and every evidence-producing action.

| Work | Owner |
| --- | --- |
| Lock comparison contract and implementation mode | Orchestrator |
| Run simulation/timing-predict and inspect artifacts | Orchestrator |
| Freeze exactly one trial | Orchestrator |
| Write the real-framework code for that trial | Implementer |
| Review the diff, run trusted gates, run source hygiene | Orchestrator |
| Run the canonical benchmark (end of Tock) | Orchestrator |
| Probe: attribute the outcome from artifacts and profiles | Orchestrator |
| Define a Probe's diagnostic question and required ranges | Orchestrator |
| Write a Probe's behavior-neutral diagnostic instrumentation | Implementer |
| Probe: retain / reject / inconclusive | Orchestrator |
| Record progress periodically in Markdown files | Orchestrator |

If no code-writing agent is available, stop and ask for one. Do not silently
collapse the role boundary by having the Orchestrator write the implementation.
That boundary also applies to temporary Probe instrumentation: the Orchestrator
defines the diagnostic question and stable ranges, delegates a bounded patch to
the Implementer, then runs the evidence-producing capture, analyzes its
artifacts, and decides. The Orchestrator never writes the patch itself.

## Concrete Steps

### Step 1 — Lock the real comparison contract

Every field that could differ between the simulation and the real framework
belongs to exactly one of three classes. Classify before comparing — the class
decides what may change, and in which direction.

| Class | Fields | Rule |
| --- | --- | --- |
| **Exogenous — must match** | every value fixed by the user, trusted evaluator/benchmark, or environment; commonly model/checkpoint revision, dtype, quantization, KV dtype, GPU type/count/topology, TP/PP/EP/CP, arrival process, request-rate or concurrency policy, input/output length distributions, warmup, duration, request cap, cutoff, SLO, and throughput/TTFT/TPOT/correctness definitions | Identical on both sides. Classification follows **who fixed the value**, not the field's intrinsic name. **Only a change in this class may update the comparison contract.** |
| **Simulated target design** | only choices left unfixed for VibeSim to select; these may include scheduler, batching policy, KV allocation/layout, kernel/backend, parallelism, or deployment topology | Chosen by VibeSim. The real implementation **chases** it. Never copied backward from the real engine. A field is not a target-design field merely because it is usually tunable. |
| **Current real-engine limitations** | active slot count; allocated KV memory; synchronization points; CUDA-graph coverage; current backend behavior; host/Python overhead | **Diagnostic gaps.** Never an experimental control, never copied into the simulation target. Fixed in reality or recorded as an open gap. |

**Comparison table gate.** Before *every* sim/real comparison, write each field
into one of the three columns above. A field that has not been classified does
not enter the comparison. This gate is what catches a real-engine limitation
being smuggled in as an experimental control.

#### Two comparison perspectives

These are evidence perspectives, not launcher modes or CLI flags. **Baseline
reproduction** asks whether VibeSim explains the real implementation under the
same contract, specialization, layout, precision, topology, and workload;
residuals are fidelity or attribution gaps. **Target exploration** changes
declared design axes within the external contract and physical feasibility; its
gap from the reproduced baseline is an implementation opportunity. Label each
relaxed axis as implemented-and-measured, implementable-but-unbuilt,
model-supported counterfactual, or speculative/unsupported. Report both views
and never slow the target merely to improve alignment.

Trusted command semantics belong to the real evidence protocol, not to an
imaginary simulation command parity rule. Exact profiler executable, capture
mode, and flag parity applies between comparable **real baseline and real
trial Probe captures**. VibeSim need not and cannot run the real profiler
command.

Do not compare runs whose workloads or metric semantics differ. Make the
simulation reproduce the trusted benchmark contract **for the exogenous class
only** — never reshape the simulated target design to match what the real engine
currently happens to do. Ask the user only when a missing choice materially
changes the experiment. An empty target directory is a valid condition for a
fresh build, not a missing input. For existing-code optimization, do not silently
replace a missing or non-runnable selected codebase with a fresh implementation.

#### Implementation mode — engine provenance

A second, independent decision, and it is **mandatory before the first Tock**.
The starting mode answers *where the real baseline comes from*; this answers
*what the implementation is allowed to be made of*. An empty target implies
neither.

| Mode | When it applies | What it allows |
| --- | --- | --- |
| **Clean-room composition** | **Default** whenever the target holds no runnable implementation | Existing serving engines (vLLM, SGLang, TensorRT-LLM) are **read-only architectural references**: they may not be imported, executed, linked, vendored, copied, adapted, or mechanically translated. Public primitive libraries — PyTorch, Transformers weight loading, FlashInfer, FastAPI, Uvicorn — are allowed **when declared**. |
| **Extend an existing framework** | The target **itself** contains that runnable framework, or the user names it | Change it in place |
| **Assemble around an existing runtime** | **Explicit user authorization only** | vLLM/SGLang/TensorRT-LLM may execute underneath |

Three guards, each closing a path that makes the wrong choice look legitimate:

- **Evaluator directories do not make a target an existing implementation.** A
  target holding only evaluators, datasets, or reference assets is still empty.
- **A reference checkout elsewhere in the workspace does not authorize a runtime
  dependency.** Being able to read vLLM is not permission to depend on vLLM.
- **"Use as reference" means** learning the architecture and public API
  behavior — **not** copying source and not translating it line by line.

If the user has not selected a mode: default to clean-room for an empty target,
and state the provenance declaration (Step 3) so the choice is visible rather
than silent. Ask first when the signals conflict — the user named an engine, or
the target already contains a runnable framework.

#### Trusted-metric precedence

When a trusted evaluator defines the score, **its semantics win**. If the
evaluator mandates arithmetic-mean TPOT, arithmetic-mean TPOT is the acceptance
number. The general guidance in
[`dev-llm-serving/references/tooling/serving-benchmark.md`](../dev-llm-serving/references/tooling/serving-benchmark.md)
to prefer percentiles over means is sound as *diagnosis*, and percentiles remain
useful as supplementary evidence — but they never replace the trusted score.
Report both; accept on the trusted one.

#### Variance policy

Interpret a delta only against a known noise level.

- Large, unambiguous delta → a single run may carry the conclusion, but record
  it as a single run.
- Small delta, **or** a result near the target / an acceptance threshold /
  an evaluator threshold → establish a noise band first: repeat the baseline
  under the same contract at least three times, and record the band in the
  contract.
- A delta inside the noise band is **inconclusive** — not accepted, not
  rejected. Say so plainly instead of picking the flattering reading.
- A change to the contract or the hardware voids the band; re-measure it.

#### Finite-workload ceiling audit

When the benchmark caps requests, the evaluator can bound the score below the
server's real capacity. Before reading any result as saturation:

- compute the reachable ceiling — `request_cap × max_output_tokens ÷ window`;
- check completed vs requested counts, and whether the trace was exhausted
  before the window closed;
- raise the request rate and see whether throughput **repeats the same number**
  rather than rising.

A result at that ceiling is **evaluator-limited**. Label it so. It is not server
saturation, and it is not evidence that the real engine has reached its limit.

### Step 2 — Tick: establish and advance the VibeSim target

Work directly through existing repo-local VibeSim skills when necessary:

- use `top-explore-models` when checkpoint facts are unclear;
- use `top-add-new-arch` when VibeSim lacks the model architecture;
- use `top-add-kernel` when a required kernel/backend is missing;
- use `operate-run-simulation` for a serving-workload throughput target;
- use `operate-run-timing-predict` for one-iteration timing to get kernel time;
- use the **Align framework to VibeSim** side of `operate-run-alignment` to align the
  target with the measured baseline or trial through the same production entry
  point used by the full model.

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

Search beyond the current framework's slots, graph coverage, scheduler,
preallocated KV, cache policy, decomposition, and backend shape support; stop at
an external contract or physical capacity boundary. In concurrency sweeps use
enough requests for multiple complete admission waves, label an observed
maximum rather than a theoretical bound, and report output-token throughput,
input processing, completions, KV/cache state, and latency separately.

**A Tick ends by freezing exactly one trial brief:** the hypothesis, the code
scope, the invariants that must hold, the expected observable effect, and the
trusted validation commands. The Tock starts from that brief and nothing else.

### Step 3 — Tock: build or advance the real framework

After obtaining a simulated result, inspect `/framework/name`.

- **Fresh build** — select one coherent bootstrap trial that creates a real,
  runnable implementation for the exact trusted correctness and benchmark
  contracts, within the implementation mode locked in Step 1. Do not request a
  pre-existing codebase, commit, benchmark result, or profiler trace, and do not
  use stubs or benchmark-specific shortcuts. The first correctness-passing,
  benchmarkable implementation becomes the real-framework baseline.
- **Existing-code optimization** — before changing code, run the trusted
  correctness check, benchmark, and profiler on the selected unchanged
  implementation. Preserve the command, revision, configuration, raw output,
  completed/requested counts, throughput, TTFT/TPOT, and profiler provenance.

Use simulator analyzer artifacts for both cases. Use measured framework
profiling to diagnose a performance gap only after a runnable baseline exists.

Use `dev-build-serving-repetitive-unit` only for kernel/boundary/layer-timing
loops. It must share the production path; only its execution plan and weights
may shrink. Promote through the reduced model, bounded full-model integration,
then canonical workload. Scheduler, capacity, latency, throughput, and workload
claims require the full model; never extrapolate across layers or requests.

Keep sequence consistency, downstream model accuracy, and performance as three
separate evidence lanes. Declare whether a trial is numerically preserving or
an intentional quality/performance tradeoff. Unexplained sequence drift blocks
a numerically preserving trial; expected drift under an intentional numeric
change is judged against a declared downstream-quality budget and an appropriate
reference. Matching length or a second framework is not an accuracy verdict.

Freeze numeric provenance with the performance contract: checkpoint revision,
compute/weight/KV dtype, scale strategy, page/layout contract, backend,
topology, and sampling policy. A numeric mismatch qualifies attribution even
when the measurement remains useful.

Use a checked-in deterministic sequence pack in the inner loop, versioned by
token digest, invariants, numeric policy, and coverage intent. Cover only the
active seam: representative phase shapes, graph replay, routing/skew, unusual
layouts, and saturation/outliers. Escalate immediately on non-finite output,
corruption, routing collapse, graph instability, or over-budget quality loss;
otherwise defer the full suite until the win is credible and attributable.

Before launching any benchmark or capture, run the preflight in
`operate-profile-serving-run` — GPU idleness, process cleanup, provenance.
A measured gap taken on a contended GPU is a scheduling artifact, not evidence.

**Declare provenance before delegating.** The brief does not go to the
Implementer until these five are written down:

1. engine provenance — which implementation mode from Step 1;
2. runtime dependencies — the declared list, nothing outside it;
3. who owns model execution — which code actually runs the forward pass;
4. the attention backend;
5. whether **any** existing serving engine participates in execution.

A missing declaration blocks delegation. Delegate only the frozen trial to the
Implementer, using `dev-llm-serving` for relevant techniques and source maps.

**Review the diff yourself, and run source hygiene under clean-room mode.**
The Orchestrator inspects the diff and runs the trusted checks itself. Under
clean-room, also check for the forbidden forms: engine imports
(`rg -n '^\s*(import|from)\s+(vllm|sglang|tensorrt_llm)' <target>`), `sys.path`
injection pointing at an engine checkout, launching an engine as a subprocess,
linking its binaries, and vendored or line-by-line-translated source. **A hit
voids the Tock** — it does not proceed to Probe.

**A Tock ends at measurement.** Record the trusted accuracy result, the
canonical benchmark result, and the artifact paths — then stop. Do not
interpret, retain, or reject here; that is Step 4.

### Step 4 — Probe: attribute the outcome and decide

Run these in order. The order matters: the direction check is cheap and
disqualifies whole classes of wrong conclusion before any profiling effort.

**1. Direction checkpoint — first, always.** If the measurement appears to
contradict the simulated target, return to the *Direction invariant* and the
Step 1 three-way table before anything else. Did an **exogenous fact** change,
or does the **current implementation merely differ**? Only the former may touch
the comparison contract. A real-engine limitation discovered here is a finding
to be fixed in reality — never a reason to adjust the simulation, and never an
experimental control to be held constant.

**2. Explain from existing artifacts first.** Use `operate-run-alignment` to
produce or resume the shared labeled comparison against the frozen VibeSim
target. Two causes need separating,
because they have different fixes and only one moves the simulated breakdown:

- **Fundamental improvement** — the *modeled work itself* is off: a real
  kernel/op is slower than the sim's slot cost (different backend/implementation,
  a shape/quant edge), or the batch shapes / parallelism the server actually runs
  differ from what the sim assumed. Close it by lowering the server's fundamental
  cost to match the sim.
- **Overhead reduction** — the kernels already match the sim's slot costs, but
  measured wallclock is larger because of **implementation-dependent overhead**
  the sim does not model: CPU launch overhead, gaps between kernels, no
  CUDA-graph capture, no async-scheduler overlap, Python overhead. Close it with
  implementation-time techniques that raise the GPU **duty cycle**. These do
  **not** move the sim breakdown, so this comparison is the only place they show.

Diagnose the gap in order: verify workload and batch-shape equivalence; match
measured operations to CostTree slots; separate localized kernel/backend gaps
from clock or thermal shifts; compare kernel-busy time with GPU-cycle time; then
explain remaining scheduling, queueing, batching, memory, and framework overhead.
Leave uncertain operation matches unresolved rather than forcing them.

**3. Observability gate.** Before selecting any **CPU / launch-overhead** class
optimization, you must already hold evidence attributing time to specific engine
phases. An opaque host gap is not such evidence — it is equally compatible with
scheduling, input preparation, attention planning, a hidden device sync, and
plain Python overhead, which are different bugs with different fixes. If the
trace cannot name the gap, add observability *before* choosing the optimization.
Route to `operate-profile-serving-run` for the capture, the NVTX readiness test,
the instrumentation contract, and the SQLite aggregation that turns ranges into
per-phase GPU-busy and host-gap numbers. For any engine that replays CUDA
graphs, that skill's node-level tracing requirement is part of the gate: at
graph level the graph-covered iterations contain no kernel rows at all, so the
phase is unattributable no matter how good the NVTX coverage is.

**4. Probe boundaries.** A Probe is diagnosis, not a performance trial:

- only behavior-neutral, opt-in, default-off, exception-safe instrumentation;
- the Orchestrator specifies the diagnostic question and stable ranges; the
  Implementer alone writes the bounded instrumentation patch;
- instrumentation lands in a **separate diagnostic commit**, never mixed with
  the trial commit;
- a Probe **must not** introduce the next optimization;
- comparable real baseline/trial captures use the same workload, profiler
  executable, capture mode, and capture flags — a change invalidates that real
  capture pair, not a simulation/real comparison;
- revert the instrumentation when the Probe ends, unless it is deliberately kept
  as dormant, default-off observability support.

**5. Decide.** Close the cycle with an explicit **retain / reject /
inconclusive**, applying the variance policy from Step 1. Judge the trial not
only by absolute speedup, but also by whether the proposal was technically
sound, whether the implementation actually tested that proposal, and whether the
explanation fits the measured result. A change whose delta sits inside the noise
band is *inconclusive* — record it that way rather than resolving it by
preference. Preserve the VibeSim target and every available real-framework
baseline, and record the decision before starting another trial.

Do not infer a causal speedup from aggregate throughput, phase occupancy, or
iteration counts. Require a controlled A/B or lossless occurrence-weighted
accounting; otherwise decide `inconclusive` and name the missing experiment.

### Step 5 — Tick again: explore more simulation possibilities

After every completed **Probe**, return to the simulator before choosing another
framework change. Feed the measured outcome and remaining headroom into the next
search. A successful cycle raises the real baseline; a rejected one still
constrains the next hypothesis; an inconclusive one usually means the next step
is a better measurement, not a new optimization. Do not skip this Tick merely
because another implementation idea already looks promising.

When several directions are viable, prefer the one with the best
**effort-to-leverage ratio** — cheap to try and easy to attribute. The levels below
are ordered by *scope* (local kernel → global topology), and attribution generally
gets harder as scope widens — but **effort is not monotonic in the level number**: a
scheduling or batching *policy* knob (often just a config flag) can be the cheapest
change of all, easier than swapping a kernel, even though it sits at a higher level.
So weigh difficulty against leverage per candidate; don't march rigidly up the
numbering. This skill chooses the level; `dev-compose-kernel` owns kernel and
fused-boundary trials, while `dev-llm-serving` supplies broader references.

1. **Kernel** — a single hot kernel dominates. Swap it for a faster implementation or
   backend. First **look for a public solution** (a faster attention / GEMM / norm
   kernel, a backend such as FlashInfer / CUTLASS, a fused variant); if one exists,
   **point the simulator at it** — ask VibeSim to model it, or to implement it with
   your guidance. You can also **ask VibeSim directly for a kernel suggestion** — it
   knows its own kernel catalog and what is fast on the target GPU. →
   `dev-compose-kernel`, with backend references from `dev-llm-serving`.
2. **Operation** — the cost is in *how ops are wired*, not one kernel. Fuse operations
   (norm+rope+attention, GEMM+bias+activation), drop redundant work, cut memory
   round-trips so several small kernels become one. → `dev-compose-kernel`, with
   engine references from `dev-llm-serving`.
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

Candidates at level 4, and any candidate justified by host-side overhead rather
than modeled work, require a passed **observability gate** (Step 4.3) before they
may be frozen. Picking a launch-overhead fix from an unattributed gap is
guesswork wearing an optimization's clothes.

Inside the level the breakdown points at, rank candidates by **leverage** = (fraction
of predicted time the change targets) × (fraction of that it could plausibly remove):
a change on a 5% slice caps at a 5% win; the headroom is on the dominant slice. Pick
the single highest-leverage change per iteration (one attributable change keeps the
later attribution explainable), and confirm in the grounded sim that it is
**feasible across the *whole* analyzer output** — throughput **and** KV capacity, iter
breakdown, optimality — not just the headline number. A candidate that improves the
headline but that the breakdown shows is infeasible (e.g. exceeds KV capacity) is not
a valid pick.

When this Tick finds another feasible improvement, freeze only the
highest-leverage candidate and return to Step 3 for the next Tock.

## Completion

Report:

- the real comparison contract, its three-way field classification, and the
  starting state (empty or existing revision);
- the implementation mode and the five-field provenance declaration;
- VibeSim target and CostTree/analyzer provenance;
- each frozen trial and implementation diff;
- trusted correctness, benchmark, and profiler artifacts;
- predicted versus measured results with diagnosed residuals;
- per cycle: the Probe's attribution and its retain / reject / inconclusive
  decision;
- the measured noise band, where one was established;
- any result labeled evaluator-limited;
- the sequence-consistency, downstream-accuracy, and performance gates used;
- the numeric provenance and every intentionally relaxed design axis;
- retained/rejected changes and the remaining evidence gap.

## Neighboring workflows

- Shared VibeSim/framework alignment: `operate-run-alignment`; techniques:
  `dev-llm-serving`; kernels: `dev-compose-kernel`; phase-level Probe evidence:
  `operate-profile-serving-run`.
- Complete reduced model: `dev-build-serving-repetitive-unit`; second-engine comparison: `operate-compare-serving-performance`.
