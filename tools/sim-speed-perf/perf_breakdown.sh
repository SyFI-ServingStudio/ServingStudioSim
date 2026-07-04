#!/bin/bash
#
# Parallel per-thread + per-symbol breakdown of a sim-speed `perf.data`, over a
# single warmed-up steady window. Companion to the `profile-sim-speed` skill:
# instead of running five `perf report` invocations by hand (each slow on a
# 500MB capture), this fires the common views concurrently on the SAME window so
# the whole picture lands in one shot.
#
# The five views (see the skill's Step 3 / 3a / 3b):
#   1. per-thread CPU split (`--sort comm`)      — which thread is the gate
#   2. simulator thread flat self-times          — the sim hot path (sim=100%)
#   3. mlsim-net-logger thread flat self-times   — the single net-log writer
#   4. mlsim-cost-logger threads flat self-times — the (sharded) cost writers
#   5. source-line annotations of the top N sim symbols (`perf annotate -l`) —
#      instruction samples bucketed by file:line (the "why is THIS symbol hot")
#
# Usage:
#   perf_breakdown.sh <perf.data> [start] [stop]      # NSYM=4 annotated symbols
#   NSYM=6 perf_breakdown.sh <perf.data>              # annotate more symbols
#
# With no window, a ~10s steady slice is auto-derived from the simulator thread's
# sample bounds: skip the front (one-time model + MoE-curve build), take 10s from
# the middle-to-late range (warmed up, small ⇒ fast to fold). Pass explicit
# absolute-second bounds (from `perf script -F time`) to override.
#
# Thread-name filters use perf's 15-char-truncated comm (mlsim-net-logge,
# mlsim-cost-logg) — the kernel truncates TASK_COMM_LEN, so the full names miss.
set -u

DATA="${1:?usage: perf_breakdown.sh <perf.data> [start] [stop]}"
START="${2:-}"
STOP="${3:-}"

if [[ -z "$START" || -z "$STOP" ]]; then
  # Auto-window: simulator-thread sample bounds → skip ~55% (build) → 10s slice.
  read -r START STOP < <(
    perf script -i "$DATA" --comm=simulator -F time 2>/dev/null \
    | awk 'NR==1{f=$1} {l=$1} END{
        gsub(/:/,"",f); gsub(/:/,"",l); span=l-f;
        s=f+0.55*span; e=s+10; if(e>l){e=l; s=e-10}
        printf "%.6f %.6f", s, e
      }'
  )
  echo "[auto-window] steady slice: ${START},${STOP}"
fi

# Persist everything next to the perf.data (override with OUTDIR=...). Each view's
# FULL perf output (every symbol / every annotated source line — not just the
# head shown on stdout) is kept, plus a combined `breakdown.txt` of what's printed.
OUTDIR="${OUTDIR:-$(cd "$(dirname "$DATA")" && pwd)/perf_breakdown}"
mkdir -p "$OUTDIR"
W="--time ${START},${STOP}"

# Full flat views (all symbols) → *.full.txt.
perf report -i "$DATA" -g none -F overhead --sort comm $W 2>/dev/null \
  | grep -vE '^#|^$' > "$OUTDIR/1_threads.full.txt" &
perf report -i "$DATA" -g none -F overhead,symbol --comm=simulator --percentage relative $W 2>/dev/null \
  | grep -vE '^#|^$' > "$OUTDIR/2_sim.full.txt" &
perf report -i "$DATA" -g none -F overhead,symbol --comm=mlsim-net-logge --percentage relative $W 2>/dev/null \
  | grep -vE '^#|^$' > "$OUTDIR/3_netlog.full.txt" &
perf report -i "$DATA" -g none -F overhead,symbol --comm=mlsim-cost-logg --percentage relative $W 2>/dev/null \
  | grep -vE '^#|^$' > "$OUTDIR/4_costlog.full.txt" &
wait

# 5. Source-line annotations of the hottest OUR-CODE sim symbols. `perf annotate
# -l` emits a "Sorted summary" that buckets instruction samples by source line —
# the instruction-level attribution, mapped back to file:line. Auto-picks the top
# `NSYM` symbols from view 2 whose name is in the `simulator::` crate (skips libc /
# parquet / alloc leaves, which have no source here). Release caveat: a line's %
# absorbs INLINED callees (see the profile-sim-speed skill's Step 3b). Full sorted
# summary (all lines) → 5_anno_*.full.txt.
NSYM="${NSYM:-4}"
mapfile -t SYMS < <(sed 's/.*\[\.\] //' "$OUTDIR/2_sim.full.txt" | grep 'simulator::' | head -"$NSYM")
for i in "${!SYMS[@]}"; do
  sym="${SYMS[$i]}"
  ( printf -- '----- %s -----\n' "$sym"
    perf annotate -i "$DATA" --stdio -l --percent-limit 2 "$sym" 2>/dev/null \
      | sed -n '/Sorted summary/,/Percent |/p' | grep -vE 'Sorted|^---|Percent|^\s*$'
  ) > "$OUTDIR/5_anno_$i.full.txt" &
done
wait

# Combined, head-limited summary → stdout AND breakdown.txt.
{
  echo "# perf_breakdown ${DATA} window ${START},${STOP}"
  echo "########## 1. PER-THREAD CPU SPLIT ##########"; head -8 "$OUTDIR/1_threads.full.txt"
  echo; echo "########## 2. SIMULATOR THREAD (rescaled sim=100%) ##########"; head -16 "$OUTDIR/2_sim.full.txt"
  echo; echo "########## 3. NET-LOGGER THREAD (single writer) ##########"; head -16 "$OUTDIR/3_netlog.full.txt"
  echo; echo "########## 4. COST-LOGGER THREADS (sharded) ##########"; head -14 "$OUTDIR/4_costlog.full.txt"
  echo; echo "########## 5. SOURCE-LINE ANNOTATIONS — top $NSYM sim symbols ##########"
  for i in "${!SYMS[@]}"; do head -10 "$OUTDIR/5_anno_$i.full.txt"; echo; done
} | tee "$OUTDIR/breakdown.txt"

echo "[saved] full raw views + breakdown.txt under $OUTDIR"
