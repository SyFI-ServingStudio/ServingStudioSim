# Skill Update Plan From The DeepSeek L1 Rebuild

This file records reusable lessons discovered while rebuilding DeepSeek V4 L1
support from `master`. It is a backlog for later skill edits, not an extension
of the current skill contract.

## How To Use This File

- Add an item only when a concrete implementation, source inspection, or
  validation result motivates it.
- Mark the evidence and the skill that would own the rule.
- Keep model-specific constants out of generic skills unless they illustrate a
  general decision rule.
- Promote an item into `skills/` only after the rebuilt kernel validates it.
- Remove or revise an item when later evidence disproves it.

## Candidate Standards

### 0. Compute Timing Means CUPTI Kernel Sum

**State:** accepted and validated across single- and multi-launch operations.

**Observation:** CUDA-event elapsed includes host/capture gaps and cannot be
compared directly with NSYS kernel time. Conversely, filtering a compound
public callable to one symbol drops legitimate internal launches.

**Proposed rule:** ordinary L1 compute rows use `Timer.cupti`. A single-kernel
call may use a stable target filter; a public compound operation uses
`kernel_name=None` so the row is the sum of all launches in one logical call.
CUDA events are diagnostics only and must not populate the compute timing DB.
Correctness, compilation, allocation, and synchronization stay outside timing.

**Likely owners:** `impl-register-kernel`, `operate-profile-existing-kernel`.

**Evidence:** fused inverse-RoPE is one CUPTI launch; MHC and split-K GEMM are
valid multi-launch public calls; sparse MLA is one CUDA-graph replay whose
internal production launches must remain one semantic cost while being summed.

### 1. Profile The Production Callable

**State:** validated by the first clean implementation.

**Observation:** `moe_align_block_size` has a public vLLM library callable. A
large Torch program that reconstructs the operation would measure a different
implementation and duplicate production code.

**Proposed rule:** when a source-faithful library callable is available, the
timed backend must invoke it directly. Torch code should remain an independent
correctness oracle and should normally be tens of lines, not a second timed
implementation.

**Likely owners:** `dev-explore-kernel`,
`orchestrator-add-kernel-to-python-profile`, `impl-register-kernel`.

**Evidence:** `moe_align_block_size:vllm_cuda` directly invokes
`vllm._custom_ops.moe_align_block_size`; its public H200 smoke wrote one row at
`0.0059809 ms`. The independent oracle checks padded work, block owners, and
the complete local route multiset outside timing.

### 2. Do Not Vendor Native Kernel Implementations

**State:** accepted project constraint; implementation audit pending.

**Observation:** Marlin, CUTLASS, FlashMLA, and similar implementations belong
to their upstream libraries. Copying their source into VibeSim creates an
unreviewable fork and obscures which production code was measured.

**Proposed rule:** call a stable library API or use a version-pinned submodule.
Record the source commit and callable. Do not copy CUDA/CUTLASS/DSL kernel
source into an L1 runner.

**Likely owners:** `top-add-kernel`, `dev-explore-kernel`,
`orchestrator-add-kernel-to-python-profile`.

**Evidence needed before promotion:** every retained DeepSeek L1 kind maps to a
library/submodule callable or has an explicit reviewed exception.

### 3. Tests Must Protect Observable Behavior

**State:** accepted for implementation; suite audit pending.

**Observation:** a test that restates registry metadata or copies the same
formula as the implementation can pass after the production path is broken.

**Proposed rule:** retain a test only when an intentional production defect
would make it fail. Prefer independently derived output, callable argument
capture, launch-count evidence, profile-DB/cache behavior, and invalid-input
rejection. Generic registry behavior belongs in shared registry tests, not in
every kernel test file.

**Likely owners:** `impl-register-kernel`, `impl-wire-kernel-to-rust`,
`impl-validate-kernel-cache`, `dev-run-tests`.

**Evidence needed before promotion:** for each retained test, document the
production regression it detects; delete tests without such a regression.

### 4. Set A Line Budget Before Implementation

**State:** first data point recorded; thresholds still need calibration.

**Observation:** wrappers grew into thousands of lines when they mixed source
reimplementation, repeated validation, registry assertions, and generated
shape cases.

**Proposed rule:** inventory the production callable and set an expected line
budget before writing. Exceeding it triggers a design review for duplicated
logic, generated data in source, or an incorrectly chosen kernel boundary. A
budget is a review signal, not a mechanical limit.

**Likely owners:** `top-add-kernel`,
`orchestrator-add-kernel-to-python-profile`,
`orchestrator-wire-kernel-to-rust`.

**Evidence needed before promotion:** record actual LOC and responsibilities for
several clean kernels, then choose useful ranges by kernel class.

**First data point:** `moe_align_block_size` uses 32 lines of registration, 132
lines of framework-independent validation/oracle/input construction, 155 lines
of production runner, and 72 lines of behavioral tests: 391 lines total. The
deleted Torch timing implementation and duplicated reference suite accounted
for most of the removed code.

### 5. Keep Semantic Slots Separate From Physical Launch Accounting

**State:** proposed; must be revalidated during Rust wiring.

**Observation:** expanding bounded physical chunks into many permanent
CostTree slots made model trees enormous. Conversely, hiding unrelated kernels
inside one timing proxy destroys attribution.

**Proposed rule:** choose one semantic slot for one modeled operation. Let its
L1 runner time the source-faithful fixed launch sequence when the sequence is an
inseparable implementation of that operation. Split distinct production
operations into distinct semantic kinds. Never materialize maximum-capacity
placeholder slots merely to represent a runtime loop.

**Likely owners:** `top-split-model-into-kernels`,
`impl-wire-kernel-to-rust`, `impl-compose-worklet`.

**Evidence needed before promotion:** the clean Rust wiring preserves kernel
alignment attribution without slot-count growth proportional to maximum
request/chunk capacity.

### 6. Execute The Backend Environment, Do Not Emulate It With PYTHONPATH

**State:** validated locally; portability still needs cleanup.

**Observation:** adding another virtual environment's `site-packages` through
`PYTHONPATH` does not execute its `.pth` files. The vLLM editable extension
finder therefore never registered and public workers could not import
`vllm._C`, although a direct run with the vLLM interpreter worked.

**Proposed rule:** a specialized profiling environment must run its own Python
interpreter when package initialization or `.pth` files own native-extension
resolution. `additional_python_paths` may select the pinned source checkout but
must not impersonate site initialization.

**Likely owners:** `impl-register-kernel`, `profiling/exec/env.py` guidance,
`dev-create-worktree`.

**Evidence:** after selecting the checkout-local vLLM interpreter, public
H200 smokes passed for both the new `moe_align_block_size` runner and the
pre-existing `moe_fused_topk` runner. Python source resolved to the clean
checkout. Native extensions currently come from a reused precompiled
environment and must be made self-contained before calling the worktree fully
portable.

### 7. Do Not Preserve Workload Detail Without Measuring Its Value

**State:** validated for the `moe_align_block_size` schema decision; broader
kernel guidance still needs comparison against other MoE operations.

**Observation:** upstream expert popularity does not imply that every
downstream MoE helper needs the full `expert_counts` vector in its public cache
key. Adding that field to `moe_align_block_size` would invalidate the existing
four-field Qwen profile table and require a migration. Existing H200 rows show
that non-outlier popularity variants change this small kernel by roughly
1.7% at 128 tokens, 8% at 384 tokens, and 11% at 2,560 tokens, but the absolute
difference is only about 0.0001--0.0016 ms per call. Isolated near-2x rows must
not be treated as distribution sensitivity without controlled repeats because
they sit outside the otherwise tight timing clusters.

**Proposed rule:** preserve a workload dimension in a kernel schema only when
controlled same-shape measurements show that it materially changes the
kernel's absolute contribution or the required cache fidelity. Compare that
benefit against profile-table compatibility, grid size, migration cost, and
runtime query complexity. Keep distribution detail on kernels that directly
consume it, such as expert dispatch or expert GEMMs; do not propagate it
mechanically through every helper in the call path.

**Likely owners:** `dev-explore-kernel`,
`orchestrator-add-kernel-to-python-profile`, `impl-register-kernel`,
`impl-validate-kernel-cache`.

**Evidence:** grouped existing H200 `moe_align_block_size:vllm_cuda` rows by
all non-popularity fields. Normal timing clusters showed microsecond-scale
absolute variation, while a few isolated rows were about twice as slow. This
does not justify replacing the compatible
`(num_tokens, num_experts, top_k, block_size)` schema with an
`expert_counts`-dependent schema. Popularity remains important for operations
whose actual work scales with the routed expert distribution.

### 8. Require Evidence Before Creating A Kernel Kind

**State:** validated by the `tensor_zero_fill` and `moe_sum` boundary reviews.

**Observation:** model-specific names encouraged the old implementation to
create duplicate kinds. `tensor_zero_fill` is exactly the zero-input path of
the existing `elementwise:torch` contract and should reuse it. Conversely,
`moe_sum` cannot reuse either generic elementwise curve even when logical bytes
match: its schema and operation are a shaped BF16 sum rather than a byte-level
fan-in operation.

**Proposed rule:** before adding a kind, document the nearest existing kind and
prove why it cannot be reused. Compare semantics, Args fields, exact production
callable, dtype/layout, and launch boundary. Use a backend only for another
implementation of the same frozen Args contract. When source evidence does not
settle performance reuse, run matched-shape measurements with identical
logical I/O and record relative plus absolute differences.

**Likely owners:** `top-split-model-into-kernels`, `top-add-kernel`,
`dev-explore-kernel`, `orchestrator-add-kernel-to-python-profile`.

**Evidence:** matched H200 measurements at T=128/1024/8192 used identical input
and output byte counts. `elementwise:torch` was 29.0%/77.6%/86.0% slower than
production `moe_sum`; `elementwise:triton` was 44.2%/48.0%/49.1% faster. The
result supports a separate `moe_sum` kind. Source inspection independently
supports reusing `elementwise:torch` for DeepSeek's plain `output.zero_()`.

## Open Questions

- Which validation and tensor-construction helpers are truly shared across
  kernels without hiding kernel-specific semantics?
- What LOC ranges are useful for a simple library wrapper, a ragged attention
  kernel, and a communication primitive?
- When a public callable launches multiple native kernels, which launch-count
  checks are stable enough to keep without binding tests to incidental compiler
  behavior?
- Which existing generic registry tests already cover lazy loading, generated
  facades, and schema wiring, so per-kernel duplication can be deleted?

## Promotion Checklist

- [ ] Finish and validate the clean `moe_align_block_size` implementation.
- [x] Record its first-pass LOC by schema, oracle, runner, and tests.
- [ ] Replace the reused vLLM binary environment with a self-contained pinned
      build or artifact and repeat both public smokes.
- [ ] Repeat the exercise for at least one ragged attention kernel.
- [ ] Audit the proposed rules against a clean Qwen or GLM sibling.
- [ ] Edit only the owning skills; do not repeat the same rule at every level.
- [ ] Run the skill validator and search for stale or contradictory guidance.
## Public compound-operation and topology rules learned from DeepSeek V4

- Prefer the production public callable over compiling private kernel classes.
  Validate its internal launch inventory with `Timer.cupti(kernel_name=None)`;
  multiple ordered physical launches can still be one semantic L1 sample.
- Physical chunking inside one model operation is a runner detail, not a reason
  to manufacture fixed CostTree slots. Profile the whole public loop once and
  sum its real launch set; keep the runtime chunk-size constant in the cache
  identity when it controls the number of launches.
- Derive a ragged source vector in the existing L5 execution adapter from
  already-owned partition groups. Do not create a model-specific worker,
  optional model hook, or admission policy merely to transport model input.
- Repeating heterogeneous schedules should compile their unique bodies once and
  use CostTree `Scale`/composition. Rank ids, request chunks, and layer copies
  are runtime/cardinality facts, not reasons for permanent slot multiplication.
- Aggregate counts are insufficient when kernels branch or plan from per-row
  metadata. Preserve ordered row positions, request ownership, or ragged valid
  counts when those values affect boundaries, page lookup, or launch planning.
- Runtime capacity must not be replaced by checkpoint capability. For example,
  a C128 cache width derived from a 65K runtime context differs physically from
  the same checkpoint configured for 1M context.
- Correctness must include the awkward physical boundaries that motivated the
  schema: partial first windows, ratio boundaries, multiple requests, padded
  page strides, inactive rows, and untouched poisoned sentinels.

## Do not preserve old profile keys by changing runtime workload semantics

- A profile DB is evidence produced from the current kernel-input contract; it
  does not own that contract. When shared workload derivation changes for a
  source-backed reason, do not add a model-specific legacy derivation merely to
  keep old DB keys queryable.
- Before retaining a model-specific routing or shape helper, require evidence
  that the production callable receives different inputs. A comment saying the
  helper avoids relabeling imported rows is evidence to reprofile, not evidence
  for a permanent fork.
- In the DeepSeek squash, `marlin_per_expert_counts` and
  `balanced_full_local_counts` bypassed the shared active-set projection solely
  to preserve old Marlin/alignment keys. The clean rebuild will use shared
  routing semantics where expert distribution is physical and omit popularity
  from `moe_align_block_size`, whose production schema is only tokens, experts,
  top-k, and block size.
# New-kind evidence rule

- Before adding a model-specific kind, require all three: a production source
  callable with known launch granularity, an observed serving-trace launch, and
  a failed reuse comparison against existing physical kernel contracts.
- A fused production callable stays one semantic L1 kind even when it combines
  several mathematical operations. Do not replace it with a bytes-based
  elementwise proxy merely because its components resemble existing kinds.

## Mixed-phase worklets share the common spine

- Do not implement prefill and decode as two complete worklets when the serving
  iteration executes their common projections on one merged batch. Keep one
  common spine and separate only the phase-specific physical callables.
- A runtime chunk loop remains one fixed semantic slot when the L1 runner times
  the complete public callable. Never turn capacity bounds into `chunk_00...N`
  or `rank0...N` CostTree leaves.
- A serial-stream alignment variant and an overlapped production variant should
  share the same leaf inventory and inputs. Change the branch composition, not
  the modeled kernels.

## Preserve source barriers and complete semantic tails

- A `Max` is valid only among branches launched from the same source fanout.
  When production joins those branches before a normalization or metadata
  dependency, close that `Max`, emit the barrier operation, then open a new
  fanout. One broad `Max` across both stages creates impossible overlap.
- A semantic compound slot must include every production launch inside its
  boundary. If a public module always performs a state-save launch immediately
  before its compress/store callable, profile both launches in one closure;
  omitting the prologue is not justified by keeping the CostTree slot count low.
- When the same semantic operation has backend-specific physical layouts, reuse
  one kind only if its Args schema can express both identities exactly. Each
  backend must fail closed on head size, cache-row layout, scale format, and
  launch path; do not coerce one layout into the other's cache key.

## Tensor shape is not a storage-layout proof

- For packed or quantized caches, derive the byte layout from the production
  writer/reader or its framework correctness test.  A public tensor shaped
  `[blocks, block_size, heads, data_bytes + scale_bytes]` may still store the
  whole page's data plane before its scale plane rather than interleaving scale
  bytes per token.
- Treat allocator page alignment separately from within-page layout.  Padding
  can change only the stride between pages while the callable continues to
  interpret the real page bytes in a block-segregated format.  A new L1 runner
  must validate both the alias offsets and the public callable against an
  independent reference before profiling a full grid.

## Synthetic values must exercise the production algorithm path

- A shape-correct tensor is not automatically a representative profiler input.
  Histogram and radix kernels can branch or overflow bounded candidate buffers
  according to the value distribution even when shape, dtype, and strides are
  identical.
- Start from the framework's own correctness data distribution and the
  production producer. Keep the seed deterministic and the construction
  outside timing, then retain an independent numerical oracle at the algorithm
  boundary. Do not weaken the oracle merely to make a pathological synthetic
  distribution pass.

## Standalone public modules must recreate framework-owned setup context

- A production `CustomOp` module may require engine configuration only while
  its dispatch method is selected. Construct it under the framework's standard
  config context, then keep that context setup outside the timed closure.
- Treat a missing context as a runner setup bug, not evidence to replace the
  public module with a private op or a Torch approximation.

## Do not turn alignment traces into scheduler policy

- A serving trace may establish request lengths, arrival times, and measured
  kernel inputs. It does not authorize replaying an observed DP rank or forcing
  neighboring requests to have equal shapes.
- When the framework hard-caps chunked prefill, implement the generic L5
  lifecycle: reserve the full KV footprint once, keep partial requests outside
  the pending policy, and expose the actual `(prefix, append)` chunk each
  iteration. Do not rewrite the trace into pre-chunked synthetic requests.
- Validate chunking with one externally observable lifecycle test (chunk shapes
  plus first-token boundary), not tests that restate internal ledgers or build a
  profile database.

## Audit concurrency clocks before interpreting TTFT

- `max_concurrency` is an observable replay invariant: when all logical arrivals
  are already due, exactly the initial capacity may retain the trace timestamp.
  Every request admitted after a completion must receive the slot-open time.
- If one completion opens multiple slots and the scheduler drains several
  already-due deferred requests in one pass, stamp every drained request with
  that same capacity-opening timestamp. Clearing the backlog marker after the
  first request silently corrupts client/server TTFT while leaving GPU work
  apparently plausible.
- Prove a clock-accounting fix with an A/B: request completion and throughput
  should remain unchanged while arrival/TTFT distributions change. Do not use a
  kernel multiplier to compensate for a frontend timestamp defect.
- Report clean predictive alignment separately from observed conditioning.
  Replaying captured DP rank or engine ingress may diagnose a residual, but it
  must not be presented as a predictive simulator result or introduced merely
  to improve one trace's score.
