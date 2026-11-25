#!/bin/bash
# Hold a Slurm allocation and run the launcher's jobs in it, one at a time.
#
#   sbatch --nodes=4 --exclusive [...] launcher/remote/allocation_runner.sh QUEUE_DIR
#
# then name the allocation in remote.yaml (`launch: {slurm: {allocation: JOBID,
# ...}}`). The launcher writes each job as QUEUE_DIR/<job id>.job, a command
# line; this script runs it as the batch script itself, so it sees the same
# environment a submitted job does (SLURM_JOB_NODELIST, the head node's GPUs,
# `srun` across the nodes). A job ends by writing its jobdir's exit_code (the
# job scripts do); a `cancel` file beside the .job file stops it. Between jobs
# nothing gives the nodes back to the scheduler: on a busy or preemptible pool
# that gap is where they are lost. `touch QUEUE_DIR/stop` releases them.
set -uo pipefail
queue=${1:?usage: allocation_runner.sh QUEUE_DIR}
mkdir -p "$queue"
echo "$SLURM_JOB_ID" > "$queue/allocation"
echo "[allocation] $SLURM_JOB_ID on $SLURM_JOB_NODELIST, queue $queue"
child=
stop() { [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; exit 143; }
trap stop TERM INT HUP
while [ ! -f "$queue/stop" ]; do
  job=$(ls "$queue"/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 5; continue; fi
  name=${job%.job}
  mv "$job" "$name.running"
  echo "[allocation] $(date -u +%FT%TZ) start $(basename "$name")"
  bash -c "$(cat "$name.running")" &
  child=$!
  while kill -0 "$child" 2>/dev/null; do
    if [ -f "$name.cancel" ]; then kill -TERM "$child" 2>/dev/null; fi
    sleep 5
  done
  wait "$child"; status=$?
  child=
  echo "$status" > "$name.done"
  rm -f "$name.running"
  echo "[allocation] $(date -u +%FT%TZ) end $(basename "$name") status $status"
done
echo "[allocation] stop file found; releasing $SLURM_JOB_ID"
