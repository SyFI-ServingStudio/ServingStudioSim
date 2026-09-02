#!/usr/bin/env bash
# Generate JSONL workloads + session_execution_v2 CSV traces for DES presets.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python3 gen_realtext_workload.py
python3 jsonl_to_session_v2.py \
 workloads/idle_24x8.jsonl \
 workloads/idle_64x8.jsonl \
 workloads/fork_s8_f4.jsonl

echo "traces in $(cd ../.. && pwd)/trace/dsv4_*.csv"
