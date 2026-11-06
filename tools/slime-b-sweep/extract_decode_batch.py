"""SGLang's own per-iteration decode telemetry, for one arm.

Runs ON CORIANDER. Emits CSV on stdout:

    pid,abs_ts,running_req,token,queue_req

Each row is one `Decode batch` line. `token` is the batch's total KV, and
`running_req / gen_throughput` recovers a per-iteration time that can be matched
against the simulator's cost_log at the same (batch, KV) point -- the only
measurement here that isolates the cost model from every scheduling effect.

Careful: `gen throughput` is averaged over the logging interval while
`#running-req` and `#token` are instantaneous, so per-point matching is noisy
(r = 0.54 over 2k points). Bucket by `token` and compare medians; do not try to
separate a constant per-iteration overhead from a proportional one with this.

    scp tools/slime-b-sweep/extract_decode_batch.py coriander:/m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/
    ssh coriander 'python3 /m-coriander/coriander/kanzhu/MLSim_workspace/wt-sgl-fa3-profile/tools/slime-b-sweep/extract_decode_batch.py 128' > data/dbatch.txt
"""

import gzip
import re
import sys

ROOT = (
    "/m-coriander/coriander/mjacob2/slime/experiments/long_rl_training/"
    "deepseek_r1_8b/results_fixed_B_sweep_20roll"
)
LINE = re.compile(
    r"\(SGLangEngine pid=(\d+)\).*\[(2026-\d\d-\d\d \d\d:\d\d:\d\d)\] Decode batch, "
    r"#running-req: (\d+), #token: (\d+).*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)

b = sys.argv[1] if len(sys.argv) > 1 else "128"
print("pid,abs_ts,running_req,token,gen_tok_s,queue_req")
with gzip.open(f"{ROOT}/batch_thresh_agg_{b}_mc0/run.log.gz", "rt", errors="replace") as f:
    for line in f:
        hit = LINE.search(line)
        if hit:
            print(",".join(hit.groups()))
