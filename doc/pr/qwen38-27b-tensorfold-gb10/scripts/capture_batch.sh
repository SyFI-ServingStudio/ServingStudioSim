#!/bin/bash
# nsys captures of TensorFold serving Qwen3.8-27B-NVFP4 (+DFlash2) on one GB10 at 1/2/4/8/16 concurrent RAG-agent
# conversations, in one server session configured as perf_study/serve/serve_tf_27b_nvfp4.sh (07:02 version:
# --parallel 16 --context 131072, reserve 30 GiB, --prompt-cache-gib 16) plus --checkpoint-slots 32 (2 x 16) and the
# perf-study-nvtx ranges. Per level: replay N lanes, warm W s outside the window, capture C s, stop the client, wait
# for the GPU to idle. Lifecycle and cleanup as in ../20261023_4_tf27b_nsys/capture.sh. Run inside tmux.
set -u
A=$(pwd); P=/home/yilegu/airbox/perf_study; TFW=$P/src/tensorfold-0.6.6-nvtx
NSYS_BIN=$(readlink -f /usr/local/cuda/bin/nsys); PORT=8310; URL=http://127.0.0.1:$PORT; S=tf27b-batch-$$
LEVELS=${LEVELS:-"1 2 4 8 16"}; CAPTURE_S=${CAPTURE_S:-60}
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
mono() { python3 -c 'import time; print(time.monotonic_ns())'; }
[ "$(avail)" -lt 90 ] && { echo "only $(avail) GB available; not starting"; exit 1; }
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 1; }
pgrep -u "$USER" -f 'tensorfold serve' >/dev/null && { echo "another tensorfold server is running"; exit 1; }
export MAX_JOBS=4 FLASHINFER_JIT_PREBUILD_MAX_JOBS=4 TENSORFOLD_MEMORY_RESERVE_GIB=30
export TENSORFOLD_NVTX=1 TENSORFOLD_ITER_RECORDS=$A/iter_records.jsonl PYTHONPATH=$TFW/src
export PATH=$P/envs/tf-dev/bin:$PATH
SERVER=($P/envs/tf-dev/bin/tensorfold serve $P/models/Qwen3.8-27B-NVFP4 --name Qwen3.8-27B-NVFP4
        --drafter $P/models/Qwen3.8-27B-DFlash2 --context 131072 --parallel 16 --checkpoint-slots 32
        --prompt-cache-gib 16 --host 127.0.0.1 --port $PORT --no-update-check)
FLAGS=(--session=$S --trace=cuda,nvtx,osrt --cuda-graph-trace=node --trace-fork-before-exec=true)
rm -f iter_records.jsonl phases.tsv
{ echo "date: $(date -Is)"; echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader)"
  echo "nsys: $NSYS_BIN $($NSYS_BIN --version)"; echo "nsys launch flags: ${FLAGS[*]}"
  echo "nsys start flags: --sample=none --cpuctxsw=none, one report per level"
  echo "tensorfold: $TFW $(git -C $TFW rev-parse HEAD) dirty=$(git -C $TFW status --porcelain | wc -l)"
  echo "torch: $($P/envs/tf-dev/bin/python -c 'import torch;print(torch.__version__)')"
  echo "server: ${SERVER[*]}"
  echo "env: MAX_JOBS=4 TENSORFOLD_MEMORY_RESERVE_GIB=30 TENSORFOLD_NVTX=1 TENSORFOLD_ITER_RECORDS=$A/iter_records.jsonl PYTHONPATH=$TFW/src"
  echo "workload: replay.py over replay.jsonl (build_replay.py from the RAG-agent trajectories), closed loop, seed = level"
  echo "levels: $LEVELS; warm W = ${WARM_S:-max(60, 15 N)} s outside the window; capture $CAPTURE_S s"
  echo "artifacts: $A"; } > provenance.txt
"$NSYS_BIN" launch "${FLAGS[@]}" "${SERVER[@]}" > server.log 2>&1 &
NPID=$!; RPID=0
cleanup() {
  [ "$RPID" != 0 ] && kill $RPID 2>/dev/null
  "$NSYS_BIN" shutdown --session=$S --kill sigterm >/dev/null 2>&1
  sleep 5; kill -0 $NPID 2>/dev/null && kill $NPID; kill -0 ${WPID:-0} 2>/dev/null && kill $WPID
}
trap cleanup EXIT
( while kill -0 $NPID 2>/dev/null; do
    if [ "$(avail)" -lt 10 ]; then echo "$(date -Is) WATCHDOG: MemAvailable < 10 GB, killing" | tee -a server.log
      "$NSYS_BIN" shutdown --session=$S --kill sigkill; kill -9 $NPID; pkill -9 -u "$USER" -f "cicc|nvcc|ptxas"; fi
    sleep 1; done ) &
WPID=$!
for i in $(seq 1 900); do
  kill -0 $NPID 2>/dev/null || { echo "server exited before ready"; tail -20 server.log; exit 1; }
  curl -sf $URL/health >/dev/null && curl -sf $URL/v1/models >/dev/null && break; sleep 2
done
curl -sf $URL/v1/models >/dev/null || { echo "not ready after 30 min"; exit 1; }
echo "$(date -Is) ready" | tee -a provenance.txt
idle_wait() {
  for i in $(seq 1 120); do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    [ "${u:-100}" -lt 5 ] && return 0; sleep 2; done
  echo "GPU not idle after 240 s"; return 1
}
for N in $LEVELS; do
  W=${WARM_S:-$(( N * 15 > 60 ? N * 15 : 60 ))}
  python3 replay.py $URL replay.jsonl --lanes $N --seconds 100000 --seed $N > replay_n$N.jsonl 2> replay_n$N.err &
  RPID=$!
  echo "$(date -Is) level $N: replay pid $RPID, warm ${W}s" | tee -a provenance.txt
  sleep $W
  kill -0 $RPID 2>/dev/null || { echo "replay for N=$N died"; tail -5 replay_n$N.err; exit 1; }
  T0=$(mono)
  "$NSYS_BIN" start --session=$S --sample=none --cpuctxsw=none --output=$A/tf27b_n$N --force-overwrite=true
  sleep $CAPTURE_S
  "$NSYS_BIN" stop --session=$S
  T1=$(mono)
  printf "%s\t%s\t%s\n" "$N" "$T0" "$T1" >> phases.tsv
  kill $RPID 2>/dev/null; wait $RPID 2>/dev/null; RPID=0
  echo "$(date -Is) level $N captured; $(wc -l < replay_n$N.jsonl) requests completed" | tee -a provenance.txt
  idle_wait || exit 1
done
echo "$(date -Is) all levels done" | tee -a provenance.txt
