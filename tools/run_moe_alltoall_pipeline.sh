#!/usr/bin/env bash
# Wait for the box to go idle, then finish the MoE-communication rework's
# GPU-gated half: profile the two new MNNVL kernels, re-run the alignment
# pipeline end to end, and run the GPU test tier.
#
# The box is shared. Every stage here needs all 8 GPUs to itself — an 8-rank
# all-to-all measured against someone else's saturating training job would
# record contention, not the kernel — so the script blocks until the GPUs are
# genuinely free rather than racing for them.
#
# "Genuinely free" is deliberately the profiler's OWN definition (see
# `find_idle_gpus` in profiling/exec/local.py: < 1000 MiB, < 10% util). A looser
# gate here is worse than no gate: the first attempt passed a 4 GiB gate at
# 19:12:01 and the profiler refused four seconds later with "need 8 idle GPU(s),
# found 7", because a neighbouring job had taken one GPU inside that window.
#
# Losing the race is expected on a shared box, so it is not a failure: the run
# goes back to waiting and tries again. Only a real error — the MNNVL workspace
# refusing to come up, a stage crashing — stops the script.
set -uo pipefail

REPO="/m-coriander/coriander/kanzhu/MLSim_workspace/wt-glm52-dp-alignment"
EXPERIMENT="$REPO/logs/20260803_8_glm52_fp8_dp8_ep8_alignment_ctx8k_out1k_c64"
STATE="$EXPERIMENT/moe_alltoall_rerun"
# Exactly `find_idle_gpus`' thresholds, so this gate can never pass while the
# profiler's own reservation would fail.
IDLE_MEMORY_MIB=1000
IDLE_UTILIZATION_PERCENT=10
# Three samples, not six: the gate exists to catch a neighbour arriving inside
# the window between the check and the profiler's own reservation, and 90s is
# already longer than that window. Six cost three minutes of an idle box.
IDLE_CONSECUTIVE_SAMPLES=3
IDLE_SAMPLE_SECONDS=30
WAIT_TIMEOUT_SECONDS=$((24 * 3600))
MAX_ATTEMPTS=40

# FORCE_GPUS=1 skips the idle wait and pins profiling to all 8 GPUs via
# `VIBESIM_PROFILE_GPUS`, which bypasses `find_idle_gpus` entirely. That is a
# deliberate override, not a shortcut: an 8-rank all-to-all sharing a GPU with
# somebody else's kernels measures contention. Use it only when the neighbours
# are parked (holding memory at ~0% util), and read `neighbour_util.log`
# afterwards — the run samples every GPU's utilisation throughout, so a
# contaminated window is visible rather than silently averaged in.
FORCE_GPUS="${FORCE_GPUS:-0}"

mkdir -p "$STATE"
LOG="$STATE/pipeline.log"
UTILIZATION_LOG="$STATE/neighbour_util.log"

say() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

fail() {
    say "FAILED: $*"
    printf 'failed\n' > "$STATE/status"
    exit 1
}

# A stage that lost the GPU race leaves a recognisable line behind. That is a
# scheduling accident, not a result, so it restarts the wait instead of ending
# the run. `lost_the_race` is read by the attempt loop.
lost_the_race=0

# The reservation error does not always surface on the stage's own stdout: the
# timing-predict launcher swallows it and only the simulator's `stdout.log`
# carries the "need 8 idle GPU(s), found 0" line. Look in both.
gpu_reservation_failed() {
    grep -q 'idle GPU(s), found' "$1" && return 0
    grep -rql 'idle GPU(s), found' "$EXPERIMENT"/*/stdout.log 2>/dev/null && return 0
    return 1
}

# Every stage is wrapped in a timeout. The transfer runner now hands the kernel
# hand-built send/recv index tensors, and an inconsistent plan does not raise —
# the ranks that finish early spin inside the collective and the whole group
# hangs, holding 8 GPUs indefinitely. A wall clock is the only thing that ends
# that. `timeout` reports 124, which is not a lost GPU race, so `fail` stops the
# run and leaves the log to read.
run_stage() {
    local name="$1"
    local limit="$2"
    shift 2
    say "--- $name (timeout ${limit}s) ---"
    local stage_log="$STATE/stage.out"
    if timeout "$limit" "$@" > "$stage_log" 2>&1; then
        cat "$stage_log" >> "$LOG"
        say "--- $name ok ---"
        return 0
    fi
    cat "$stage_log" >> "$LOG"
    if gpu_reservation_failed "$stage_log"; then
        say "--- $name lost the GPU race; will wait and retry ---"
        lost_the_race=1
        return 1
    fi
    fail "$name"
}

gpus_are_idle() {
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', ' -v mem="$IDLE_MEMORY_MIB" -v util="$IDLE_UTILIZATION_PERCENT" \
              '$1 > mem || $2 > util { count++ } END { print count + 0 }')
    [ "$busy" -eq 0 ]
}

# How many GPUs are actually computing, ignoring memory residency.
#
# The strict gate above needs < 1000 MiB per GPU, which is what `find_idle_gpus`
# demands and therefore what an unforced run needs. But a neighbour that parks —
# holding 90 GB at 0% util for hours — never clears it, while costing this
# benchmark nothing. That is precisely the window where FORCE_GPUS is the right
# call, and it is invisible unless the two conditions are reported apart. Every
# wait sample logs both so the log answers "should I force?" rather than only
# "may I start?".
gpus_computing() {
    nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits \
        | awk -v util="$IDLE_UTILIZATION_PERCENT" '$1 > util { count++ } END { print count + 0 }'
}

wait_for_idle_gpus() {
    if [ "$FORCE_GPUS" = "1" ]; then
        say "FORCE_GPUS=1 — skipping the idle gate, profiling pinned to GPUs 0-7"
        say "$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
        printf 'running\n' > "$STATE/status"
        return 0
    fi
    printf 'waiting\n' > "$STATE/status"
    say "waiting for 8 idle GPUs (need $IDLE_CONSECUTIVE_SAMPLES consecutive idle samples, ${IDLE_SAMPLE_SECONDS}s apart)"
    local deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
    local idle_streak=0
    local quiet_streak=0
    while [ "$idle_streak" -lt "$IDLE_CONSECUTIVE_SAMPLES" ]; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            fail "gave up after ${WAIT_TIMEOUT_SECONDS}s waiting for idle GPUs"
        fi
        local computing
        computing=$(gpus_computing)
        if [ "$computing" -eq 0 ]; then
            quiet_streak=$((quiet_streak + 1))
        else
            quiet_streak=0
        fi
        if gpus_are_idle; then
            idle_streak=$((idle_streak + 1))
            say "idle sample $idle_streak/$IDLE_CONSECUTIVE_SAMPLES"
        else
            [ "$idle_streak" -gt 0 ] && say "GPUs busy again, restarting the idle streak"
            idle_streak=0
            # Neighbours resident but not computing: an unforced run still cannot
            # start, but a forced one would measure the kernel rather than the
            # contention. Recorded so the decision is a reading, not a guess.
            say "waiting: $computing/8 GPU(s) computing (quiet for $quiet_streak consecutive samples)"
        fi
        [ "$idle_streak" -lt "$IDLE_CONSECUTIVE_SAMPLES" ] && sleep "$IDLE_SAMPLE_SECONDS"
    done
    say "GPUs idle; starting"
    printf 'running\n' > "$STATE/status"
}

# One end-to-end attempt. Returns non-zero only when a stage lost the GPU race;
# anything else calls `fail` and never comes back.
run_all_stages() {
    cd "$REPO" || fail "cd $REPO"

    # -----------------------------------------------------------------------
    # Stage 1 — MNNVL probe. The plan's largest risk: vLLM brings the fabric
    # workspace up over its own CPU process group, and nothing yet proves the
    # profiler's own spawn can. One shape per kind answers it in seconds, and a
    # failure here is worth stopping on: every later stage would inherit it.
    #
    # Every probe shape is deliberately OFF the sweep grid (1000 tokens sits
    # between the axis's 896 and 1024; 3000 rows between 2048 and 4096). JIT only
    # fills rows that are missing, so a probe landing on a grid point makes the
    # grid fill skip that cell — leaving one point of the curve measured in a
    # different, single-shape spawn. That is not a small difference: the first
    # run left prepare@1024 at 55.7 us against ~30.4 us interpolated from its
    # neighbours, +83%, right at the decode-to-prefill transition. Off-grid rows
    # are inert (the cache reads grid points only) and keep the evidence.
    # -----------------------------------------------------------------------
    run_stage "probe moe_alltoall_prepare" 300 \
        uv run python -m profiling run moe_alltoall_prepare \
            --backend flashinfer_mnnvl \
            --spec '{"ep_size": 8, "tokens_per_rank": 1000, "top_k": 8, "slot_count": 256, "fabric": "nvlink"}' \
            --json --output-dir "$STATE/probe_prepare.$RUN_ID.$ATTEMPT" || return 1

    run_stage "probe moe_alltoall dispatch" 300 \
        uv run python -m profiling run moe_alltoall \
            --backend flashinfer_mnnvl \
            --spec '{"ep_size": 8, "max_send_rows": 3000, "max_recv_rows": 3000, "top_k": 8, "slot_count": 256, "hidden_bytes": 12288, "direction": "dispatch", "fabric": "nvlink"}' \
            --json --output-dir "$STATE/probe_dispatch.$RUN_ID.$ATTEMPT" || return 1

    run_stage "probe moe_alltoall combine" 300 \
        uv run python -m profiling run moe_alltoall \
            --backend flashinfer_mnnvl \
            --spec '{"ep_size": 8, "max_send_rows": 3000, "max_recv_rows": 3000, "top_k": 8, "slot_count": 256, "hidden_bytes": 12288, "direction": "combine", "fabric": "nvlink"}' \
            --json --output-dir "$STATE/probe_combine.$RUN_ID.$ATTEMPT" || return 1

    # -----------------------------------------------------------------------
    # Stage 2 — timing-predict. It enables JIT profiling itself, so this is also
    # what fills the two new kernels' full sweep grids.
    # -----------------------------------------------------------------------
    [ -d "$EXPERIMENT/timing_predict" ] && mv "$EXPERIMENT/timing_predict" "$STATE/timing_predict.previous"
    run_stage "timing-predict" 14400 \
        uv run python -m launcher alignment timing-predict "$EXPERIMENT/timing_predict.yaml" || return 1

    # -----------------------------------------------------------------------
    # Stage 3 — labeling. The rule file already carries the new decisions;
    # `apply` is what resolves their slot suffixes against the freshly emitted
    # manifest, and it fails loudly on a suffix the model no longer emits.
    # -----------------------------------------------------------------------
    local manifest
    manifest=$(ls "$EXPERIMENT"/timing_predict/raw/cost_manifest/*.json 2>/dev/null | head -1)
    [ -n "$manifest" ] || fail "no cost manifest emitted"
    [ -f "$STATE/kernel_sequences_labeled.pre_alltoall.json" ] || \
        cp "$EXPERIMENT/kernel_sequences_labeled.json" "$STATE/kernel_sequences_labeled.pre_alltoall.json"
    run_stage "label apply" 900 \
        uv run python -m alignment label apply \
            "$EXPERIMENT/kernel_sequences_labeled.json" "$manifest" "$EXPERIMENT/labeling_rules.json" || return 1
    run_stage "label check" 900 \
        uv run python -m alignment label check "$EXPERIMENT/kernel_sequences_labeled.json" || return 1

    # -----------------------------------------------------------------------
    # Stage 4 — kernel-align, then the sim with the multiplier that pass
    # derives, then e2e.
    # -----------------------------------------------------------------------
    [ -d "$EXPERIMENT/analysis_kernel" ] && mv "$EXPERIMENT/analysis_kernel" "$STATE/analysis_kernel.previous"
    run_stage "analyze kernel-align" 3600 \
        uv run python -m launcher alignment analyze "$EXPERIMENT/analyze_kernel.yaml" || return 1

    [ -d "$EXPERIMENT/simulation" ] && mv "$EXPERIMENT/simulation" "$STATE/simulation.previous"
    run_stage "sim" 7200 \
        uv run python -m launcher alignment sim "$EXPERIMENT/simulation.yaml" \
            --gpu-time-multiplier-from "$EXPERIMENT/analysis_kernel" --refresh || return 1

    [ -d "$EXPERIMENT/analysis_e2e" ] && mv "$EXPERIMENT/analysis_e2e" "$STATE/analysis_e2e.previous"
    run_stage "analyze e2e" 3600 \
        uv run python -m launcher alignment analyze "$EXPERIMENT/analyze_e2e.yaml" || return 1

    # -----------------------------------------------------------------------
    # Stage 5 — the GPU test tier, including the throughput goldens that are the
    # acceptance point for "new kernel, not changed op".
    # -----------------------------------------------------------------------
    run_stage "just test-gpu" 7200 just test-gpu || return 1
    return 0
}

# `--output-dir` is an immutable artifact root: the CLI refuses to write into a
# directory that already holds a request. Scope it per attempt so a retry writes
# somewhere fresh and the previous attempt's evidence survives. The run id is in
# the path too: `ATTEMPT` restarts at 1 every time the script is relaunched, so
# attempt-only scoping collided with the previous run's probe dir and burned an
# idle window on an error that had nothing to do with the GPUs.
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ATTEMPT=0

# Build BEFORE queuing for GPUs. The first successful attempt spent 2 of its
# ~3 idle minutes on `cargo build`, and the reservation was gone by the time the
# predictor actually needed it. Nothing here touches a GPU.
cd "$REPO" || fail "cd $REPO"
say "pre-building simulator + analyzer so no GPU window is spent compiling"
if ! uv run cargo build --release -p simulator >> "$LOG" 2>&1; then
    fail "cargo build -p simulator"
fi
if ! uv run cargo build --release --manifest-path analyzer/rust/Cargo.toml >> "$LOG" 2>&1; then
    fail "cargo build analyzer"
fi
say "pre-build done"

# The first pass recorded both tables with an eager CUDA-event loop, which
# measures a rendezvous floor vLLM never pays: decode replays a captured graph,
# and `prepare` came out flat at ~78 us from 1 token to 4096 against 15.6 us of
# real kernel time. The runner now captures and replays a graph; those rows
# describe a different quantity and would be silently reused, so they go. Backed
# up first: they are the evidence for why the basis changed.
#
# `moe_alltoall`'s columns changed with the basis (`tokens_per_rank` ->
# `max_send_rows`/`max_recv_rows`), so that table is dropped outright rather
# than emptied.
if [ ! -f "$STATE/profile.db.event_timed_rows.json" ]; then
    say "--- invalidating the event-timed alltoall rows ---"
    if ! uv run python tools/invalidate_moe_alltoall_rows.py \
        --backup "$STATE/profile.db.event_timed_rows.json" >> "$LOG" 2>&1; then
        fail "invalidate event-timed rows"
    fi
fi

if [ "$FORCE_GPUS" = "1" ]; then
    export VIBESIM_PROFILE_GPUS=0,1,2,3,4,5,6,7
    # Sample every GPU while the run proceeds. A neighbour that wakes up mid-run
    # is the one thing that would quietly invalidate the numbers, and this is the
    # only record of whether it happened.
    (
        while true; do
            printf '%s %s\n' "$(date '+%F %T')" \
                "$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                    --format=csv,noheader,nounits | tr '\n' '|')" >> "$UTILIZATION_LOG"
            sleep 5
        done
    ) &
    UTILIZATION_SAMPLER_PID=$!
    trap 'kill "$UTILIZATION_SAMPLER_PID" 2>/dev/null' EXIT
    say "utilisation sampler running (pid $UTILIZATION_SAMPLER_PID) -> $UTILIZATION_LOG"
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    ATTEMPT="$attempt"
    say "===== attempt $attempt/$MAX_ATTEMPTS ====="
    lost_the_race=0
    wait_for_idle_gpus
    if run_all_stages; then
        say "ALL STAGES OK"
        printf 'ok\n' > "$STATE/status"
        exit 0
    fi
    [ "$lost_the_race" -eq 1 ] || fail "a stage failed without losing the GPU race"
    say "attempt $attempt lost the GPU race; waiting for the box again"
done
fail "still could not hold 8 GPUs after $MAX_ATTEMPTS attempts"
