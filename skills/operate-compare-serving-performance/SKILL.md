---
name: operate-compare-serving-performance
description: >-
  Compare two real serving engines by end-to-end performance, then compare
  broad GPU-time regions without per-iteration alignment. Not for simulator
  alignment or an A/B within one implementation.
---

# Compare Serving Performance

Answer two questions in order:

1. Which complete serving system is faster for the same workload?
2. Where does each system spend its GPU time?

The first answer comes from normal benchmark runs. The second comes from
separate, bounded profiling runs and is diagnostic only.

## 1. Make the workloads comparable

Use the same materialized requests and load policy for both engines. Record:

- engine, model, checkpoint, tokenizer, and load-generator revisions;
- hardware, GPU count, topology, and memory limits;
- input requests, sampling settings, concurrency or request rate, and seed;
- compute, weight, and KV precision, quantization/scaling method, attention
  backend, graph mode, and cache policy.

Differences are allowed when they are the point of the experiment, such as a
more aggressive quantization policy. State them explicitly and limit the claim
accordingly. Do not call two runs equivalent when their model, workload, or
resource contracts differ.

## 2. Check results before a long benchmark

Run a small deterministic request set first. Check request success, input and
output token counts, and a few exact output-token sequences. Output length alone
does not establish sequence consistency.

Treat sequence consistency and downstream quality as separate checks. During
development, a few fixed sequences are enough to catch obvious corruption. If
the experiment intentionally changes numerical behavior, run the appropriate
quality evaluation before making a final performance recommendation.

## 3. Measure overall performance

Use normal, unprofiled runs for the performance result. Apply the same warmup,
measurement window, offered load, request population, and drain rule. Report:

- successful and failed requests, input tokens, and output tokens;
- input-token, output-token, and request throughput, plus fixed-workload
  completion time;
- TTFT, TPOT or ITL, and end-to-end latency distributions;
- GPU count and peak memory or KV-cache use.

Keep startup, steady state, and final drain separate. If the workload mixes
prefill and decode, report that mix rather than treating every iteration as the
same kind of work.

To claim saturation, sweep offered concurrency or request rate with enough
requests to sustain multiple admission waves. Saturation is observed only when
more offered work no longer improves the chosen throughput metric.

## 4. Explain GPU time with broad regions

Capture a short, representative profile for each engine with
`operate-profile-serving-run`. Aggregate the complete steady window; do not
align iterations or individual kernel launches across engines.

Classify device work into a small shared set:

- attention, including projection and KV-cache work;
- dense or MoE FFN, including routing and combine;
- communication;
- model input/output work, including embedding, final norm, LM head, and
  sampling;
- other or unmapped device work.

For each rank, report total time and share by region, classification coverage,
and the observed batch/shape distribution. Normalize by completed work when the
two captures complete different amounts. Keep the bottleneck rank visible; do
not sum replicated ranks into a fictional serial runtime.

Schedulers may change batching, overlap, graph buckets, shapes, and kernel
counts. Therefore these broad regions explain where work moved, but they do not
need to add up to the end-to-end performance gap. When overlap matters,
distinguish summed kernel duration from actual GPU-busy wall time.

## 5. Report the conclusion in two layers

Report overall serving metrics first. State the workload, resource contract,
intentional numeric differences, correctness/quality status, and whether the
result represents startup, steady state, or a fixed finite workload.

Then report the coarse GPU-region breakdown, shape differences, unmapped share,
and profiling limitations. Do not let an incomplete kernel classification
override the measured end-to-end result, and do not claim a causal explanation
without a controlled experiment.

## Boundaries

- Simulator-to-framework comparison: `operate-run-alignment`.
- Profiling or A/B attribution inside one real engine:
  `operate-profile-serving-run`.
