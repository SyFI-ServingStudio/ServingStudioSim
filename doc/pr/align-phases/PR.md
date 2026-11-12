# alignment: Phase 1/2 cases, DP forward ordinals, and capture fixes

## Purpose

Shared measurement fixes found while aligning DeepSeek-V4.1-Flash on vLLM at DP4. Nothing here is
model-specific; the DeepSeek-V4.1 PR builds on it.

**Alignment Phase 1 / Phase 2.**
- A pack can declare `phase: 1` cases that replay a tracked trace file.
  `alignment-campaign --alignment-phase {1,2}` selects them.
- Each case's calibration inputs are read from its own workload pass:
  - acceptance by output position;
  - the global acceptance chain;
  - the vLLM KV pool.
- `check` requires acceptance by output position for speculative cases. One case-wide chain gives
  every request the same acceptance. GLM-5.3 DFlash2 cases 07/08 read TPOT -26%/-24% with one chain
  and -11%/-6% with the buckets.
  - The old GLM-5.2 spec5 packs declare a `legacy_single_chain` exemption with its reason, so `check`
    warns for them instead of failing.
- One `campaign.yaml` writer. Existing packs are rewritten in compact form.
- The skills define the two phases and hold one command block for both.

**DP step identity: forward ordinals** (vLLM patches 0019, 0020).
- Before, an idle DP rank's dummy forward wrote no record, so the ranks' records could not be joined
  and the analyzer paired steps by wall clock.
- Every forward now writes a schema-5 record carrying the rank's `forward_ordinal`. Dummy forwards
  write one too.
- `alignment-workload` and Check 1 join the ranks by ordinal. Both refuse a DP capture that has no
  ordinals and tell the user to recapture it. Record extraction checks each rank's ordinal coverage.
- `timing-predict` accepts schema-5 records.

**Capture fixes.**
- **Patch 0021: compile MXFP8 tactics at warmup.** An autotune cache hit skipped the dummy runs that
  compile each M bucket's CuTe-DSL MXFP8 tactic. Each tactic was then compiled mid-serving, about
  0.5 s each, and every DP rank stalled in its next collective. There were 11 such stalls in a 20 s
  DSv4.1 DP4 window.
- **Patch 0022: open alignment NVTX scopes only while the CUDA profiler runs.**
  - The bug: after an nsys capture range ends, registering a new per-iteration NVTX scope name blocked
    with the GIL held. The other DP ranks then waited forever in the next DP all-reduce.
  - Evidence: py-spy stacks of DSv4.1 DP4 capture 20822. Captures 20617 and 20801 hung the same way.
  - Not reproduced: a standalone repro of "register after stop" did not hang in 3 runs. The last hop is
    inferred from the stacks and the nvtx source.
- **Docker in its own session.** The first docker call of a container chunk now runs in its own
  session. Inside a Slurm job, `docker` is a shim that starts a job-scoped Podman service on first
  use. That service joined the predictor's process group, outlived it, and the launcher killed it as a
  leaked descendant.
- **Stale `error.json`.** The launcher deletes the binary's `error.json` before each run. Before, a
  later failure printed an earlier run's error.
- **Truncated drafts.** timing-predict input excludes verify steps whose draft was cut short at
  `max_model_len`, identified by the request's own progress row. Before, one such step failed the whole
  capture.

## Test Plan

- `CUDA_VISIBLE_DEVICES= just test-cpu` on the branch head:
  - Rust `--lib`: 1583 passed, 8 ignored.
  - pytest: 5181 passed, 35 skipped. The 4 `tests/test_l1_single_gemm.py::test_energy_*` tests fail.
    They fail the same way on `origin/main` 8fe2ff59; this PR does not touch them.
- vLLM fork test `tests/v1/worker/test_alignment_trace.py`: 14 passed in the image.
  `tests/v1/engine/test_iteration_logging.py`: 10 passed in the image.
- Real DSv4.1-Flash DP4/EP4 captures on B200, built from `reference_engines/vllm` base + patches
  through 0022 (image `vibesim-profiler-vllm:fe6a3d9a6af96e67`).

## Test Result

**Forward ordinals, DSv4.1 DP4 workload capture.**
- Workload fold: 9439 steps; 0 ragged records.
- Check 1 join, with ordinals: 9439 steps, 0 unpaired.
- Check 1 join, previous kernel-overlap pairing: 6626 steps, 4277 peer windows unpaired.
  - 2813 forwards with real work had no step under that pairing, because rank 0 ran a dummy forward
    there.
- Shifting one rank's ordinals by +1 is refused. The dummy windows of forward 1 no longer overlap.

**Patch 0022, DSv4.1 DP4 nsys captures:**

| Job | Patches | Result |
|---|---|---|
| 20617, 20801, 20822 | 0019–0021 | hung after the profiler stop; DP all-reduce deadlock |
| 20826 | 0019–0022 | completed, 100/100 requests; the stop paused the app 161 s, then serving resumed |
| 20842 | 0019–0022 | completed; logged forwards went from 17164 to 27584 after the stop |

**Patch 0021, DSv4.1 DP4 (Check 2 / workload pass):**
- Multiplier screen: 7 excluded steps, against 17 in the previous capture without 0021.
- Measured diverse_100 workload pass, without 0021 → with it: mean server TTFT 701 → 545 ms, mean
  server TPOT 8.3 → 6.9 ms.

## Dependencies

- vLLM: patches 0019–0022 in `reference_engines/vllm/patches/`, applied to the pinned official base.
  No fork branch has to be published.
- Captures taken before patch 0019 at DP > 1 must be recaptured: `alignment-workload` and Check 1
  refuse them. SGLang DP passes emit no ordinals yet. Single-rank and TP captures are unchanged.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
