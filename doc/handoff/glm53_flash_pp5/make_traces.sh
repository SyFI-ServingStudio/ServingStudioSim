#!/usr/bin/env bash
# Build the handoff's closed-loop traces under logs/glm53_flash_pp5/traces and check them against traces.sha256.
# Every step is deterministic (seed 0). About 3 GB and two minutes; the 8 h files are the large ones.
#
# Input: trace/tracelab_preserving.csv, built from the public TraceLab v0.0.1 corpus as trace/README.md
# ("Regenerating") describes.
#
# Usage, from the ServingStudioSim root:  doc/handoff/glm53_flash_pp5/make_traces.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

H=doc/handoff/glm53_flash_pp5
T=logs/glm53_flash_pp5/traces
SRC=trace/tracelab_preserving.csv
SHORT=(500 600 650 750 1000 1250 1500 2000 2500 2656 3000)  # 16-minute checks
LONG=(2000 2500 2656 3000 3500)                         # 8-hour runs

if [[ ! -f $SRC ]]; then
    echo "missing $SRC: build it as trace/README.md (Regenerating) describes" >&2
    exit 1
fi
grep " $SRC\$" "$H/traces.sha256" | sha256sum --check --quiet  # wrong source: fail before the 2-minute build
mkdir -p "$T"
py() { uv run --no-project python "$@" > /dev/null; }

# Decode runs elsewhere: fold (output - 1) / 80 s into each round's tool wait.
py trace/session_decode_wait.py "$SRC" "$T/mono_sessions_d80.csv" --decode-tok-s 80

# A closed loop of N sessions that starts in the renewal equilibrium, then 20 N whole sessions.
for c in $(printf '%s\n' "${SHORT[@]}" "${LONG[@]}" | sort -nu); do
    py trace/session_closed_loop.py "$T/mono_sessions_d80.csv" "$T/closed_c$c.csv" \
        --concurrency "$c" --after $((20 * c)) --seed 0
done

# Cut to what 16 minutes can reach, so the short checks load seconds of trace instead of 0.5 GB.
for c in "${SHORT[@]}"; do
    py "$H/cut_trace.py" "$T/closed_c$c.csv" "$T/closed16m_c$c.csv" --minutes 16 --sessions $((2 * c))
done

sha256sum --check --quiet "$H/traces.sha256"
echo "traces OK in $T"
