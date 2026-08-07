#!/usr/bin/env bash
# Wait for 8 genuinely idle GPUs, then run the CUDA-graph probe for the MNNVL
# MoE calls.
#
# Why this is queued rather than run on demand: the first attempt was overrun
# mid-flight by a Megatron/SGLang job that took all 8 cards, and it produced
# graph times ~100x the eager ones — physically impossible, so the run told us
# nothing except that it was contended. An 8-rank collective measured against a
# saturating neighbour records the neighbour.
#
# The question the probe answers: vLLM replays decode as a captured CUDA graph,
# so eager per-call rendezvous is a cost the target never pays. If the MNNVL
# alltoall captures, per-iteration time should collapse from the eager floor
# (prepare 76 us) toward the in-situ trace value (15.6 us). If it does not
# capture, the run says so explicitly via `graph_error` and the modelling has to
# take another route.
set -uo pipefail

REPO="/m-coriander/coriander/kanzhu/MLSim_workspace/wt-glm52-dp-alignment"
STATE="$REPO/logs/20260803_8_glm52_fp8_dp8_ep8_alignment_ctx8k_out1k_c64/moe_alltoall_rerun"
PROBE="$STATE/graph_probe.py"
# `find_idle_gpus`' own thresholds, so this gate cannot pass while the profiler
# would refuse. Three samples rather than six: enough to reject a transient dip,
# half the dead time once the box is actually free.
IDLE_MEMORY_MIB=1000
IDLE_UTILIZATION_PERCENT=10
IDLE_CONSECUTIVE_SAMPLES=3
IDLE_SAMPLE_SECONDS=30
WAIT_TIMEOUT_SECONDS=$((24 * 3600))

LOG="$STATE/graph_probe.log"
STATUS="$STATE/graph_probe.status"

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

gpus_are_idle() {
    local busy
    busy=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F', ' -v mem="$IDLE_MEMORY_MIB" -v util="$IDLE_UTILIZATION_PERCENT" \
              '$1 > mem || $2 > util { count++ } END { print count + 0 }')
    [ "$busy" -eq 0 ]
}

printf 'waiting\n' > "$STATUS"
say "waiting for 8 idle GPUs ($IDLE_CONSECUTIVE_SAMPLES consecutive samples, ${IDLE_SAMPLE_SECONDS}s apart)"
deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
streak=0
while [ "$streak" -lt "$IDLE_CONSECUTIVE_SAMPLES" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        say "gave up waiting"; printf 'failed\n' > "$STATUS"; exit 1
    fi
    if gpus_are_idle; then
        streak=$((streak + 1)); say "idle sample $streak/$IDLE_CONSECUTIVE_SAMPLES"
    else
        [ "$streak" -gt 0 ] && say "busy again, restarting the streak"
        streak=0
    fi
    [ "$streak" -lt "$IDLE_CONSECUTIVE_SAMPLES" ] && sleep "$IDLE_SAMPLE_SECONDS"
done

say "GPUs idle; running the graph probe"
printf 'running\n' > "$STATUS"
cd "$REPO" || { printf 'failed\n' > "$STATUS"; exit 1; }

# Record what the cards looked like at the start AND at the end: a probe that
# was overrun mid-flight is worthless, and the only way to know is to check both
# ends rather than trust the gate that let it in.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    > "$STATE/graph_probe.gpu_before.txt"

PYTHONPATH="$REPO" timeout 2400 uv run python "$PROBE" \
    > "$STATE/graph_probe.out" 2> "$STATE/graph_probe.err"
probe_exit=$?

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    > "$STATE/graph_probe.gpu_after.txt"

if [ "$probe_exit" -ne 0 ]; then
    say "probe exited $probe_exit; see graph_probe.err"
    printf 'failed\n' > "$STATUS"
    exit 1
fi

if awk -F', ' '$2 > 1000 || $3 > 10 { found = 1 } END { exit !found }' \
        "$STATE/graph_probe.gpu_after.txt"; then
    say "WARNING: a neighbour took the cards during the run — results are contended"
    printf 'contended\n' > "$STATUS"
    exit 1
fi

say "probe done on clean cards"
printf 'ok\n' > "$STATUS"
