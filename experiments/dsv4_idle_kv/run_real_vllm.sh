#!/usr/bin/env bash
# Real-GPU bring-up for DeepSeek-V4-Flash-0731 on ptc.
# Default is --check only. Full serve needs 8×H100 or 4×H200.
set -euo pipefail
MODE="${1:---check}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}"

ngpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
echo "gpus=$ngpu"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

need_full=8
name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)
if echo "$name" | grep -qi H200; then
  need_full=4
fi

echo "verified_full_launch_min_gpus=$need_full"
echo "model=$MODEL"
echo "vllm>=0.25 required for DSpark on 0731"

if [[ "$MODE" == "--check" ]]; then
  if [[ "$ngpu" -lt "$need_full" ]]; then
    echo "REFUSE full launch: have ${ngpu} GPU(s), need ${need_full}."
    echo "Smoke (unverified, short context, no DSpark) would use DP=${ngpu}."
    exit 2
  fi
  echo "OK to attempt --full"
  exit 0
fi

if [[ "$MODE" == "--smoke" ]]; then
  echo "UNVERIFIED smoke on ${ngpu} GPU(s), max-model-len=4096, no DSpark"
  exec vllm serve "$MODEL" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --data-parallel-size "$ngpu" \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
fi

if [[ "$MODE" == "--full" ]]; then
  if [[ "$ngpu" -lt "$need_full" ]]; then
    echo "REFUSE --full: ${ngpu} < ${need_full}" >&2
    exit 2
  fi
  exec vllm serve "$MODEL" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-expert-parallel \
    --tensor-parallel-size "$ngpu" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
fi

echo "usage: $0 --check|--smoke|--full" >&2
exit 1
