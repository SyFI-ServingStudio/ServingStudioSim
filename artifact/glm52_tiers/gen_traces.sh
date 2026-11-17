#!/bin/bash
# Closed-loop TraceLab monotonic traces for the GLM-5.2 tier presets, written to logs/glm52_tiers_traces/.
# Run from the repository root:  artifact/glm52_tiers/gen_traces.sh [concurrency ...]
# With no arguments it writes every concurrency the presets use. Each file is the first `c` sessions starting at once
# and the next one joining as a session ends, `--after 20c`, seed 0 (README.md lists the expected sha256 of each).
set -euo pipefail
OUT=logs/glm52_tiers_traces
SESSIONS=$OUT/mono_sessions_d80.csv
mkdir -p "$OUT"
if [ ! -f "$SESSIONS" ]; then
  # trace/tracelab_preserving.csv is untracked; its manifest gives the source and sha256.
  uv run python trace/session_decode_wait.py trace/tracelab_preserving.csv "$SESSIONS" --decode-tok-s 80
fi
concs=("$@")
[ ${#concs[@]} -eq 0 ] && concs=(100 200 250 375 500 750 1000 1125 1250 1375 1500 2000 2500 3000 4000)
for c in "${concs[@]}"; do
  uv run python trace/session_closed_loop.py "$SESSIONS" "$OUT/closed_c${c}_s0.csv" --concurrency "$c" --after $((20 * c)) --seed 0 &
done
wait
