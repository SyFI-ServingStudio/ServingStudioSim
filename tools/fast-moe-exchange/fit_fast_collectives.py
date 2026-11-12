"""Fit `a + bytes / L` per fast-collective backend from B200 8-GPU profile.db rows.

bytes = the busiest rank's one-direction bytes of a two-shot all-reduce,
2 (N-1)/N * message. Relative least squares (minimise sum ((model-meas)/meas)^2).
Prints the fit and per-row error so the constants in simulator/src/arch/ep_comm.rs
can be checked. Usage: python fit_fast_collectives.py [profile.db]
"""
import sqlite3, sys, json
import numpy as np

db = sys.argv[1] if len(sys.argv) > 1 else "profiling/profile.db"
con = sqlite3.connect(db)
N = 8

def fit(name, rows, link_gb_per_s=None):
    x = np.array([r[0] for r in rows], float)  # bytes per rank
    y = np.array([r[1] for r in rows], float)  # us
    if link_gb_per_s is None:
        A = np.stack([np.ones_like(x) / y, x / y], 1)
        (a, s), *_ = np.linalg.lstsq(A, np.ones_like(y), rcond=None)
    else:  # fixed link: relative LSQ for a alone
        s = 1e-3 / link_gb_per_s
        r = 1 - s * x / y
        a = float((r / y).sum() / (1 / y ** 2).sum())
    model = a + s * x
    err = model / y - 1
    out = {"backend": name, "a_us": round(a, 3), "link_gb_per_s": round(1e-3 / s, 1),
           "rows": len(rows), "rms_pct": round(100 * float(np.sqrt((err ** 2).mean())), 2),
           "worst_pct": round(100 * float(err[np.argmax(abs(err))]), 2),
           "table": [(int(r[2]), round(r[1], 2), round(m, 2), round(100 * e, 1)) for r, m, e in zip(rows, model, err)]}
    return out

res = []
# MNNVL two-shot (num_tokens*H*N*2 > 1 MiB): TP8 hidden 4096 rows above 16 tokens.
q = con.execute("""select num_tokens, time_ms*1000 from all_reduce_fusion where gpu_name like '%B200%'
  and backend='flashinfer_mnnvl' and num_gpus=8 and hidden_dim=4096 and dtype='bf16' and is_outlier=0 order by num_tokens""").fetchall()
two = [(2 * (N - 1) / N * t * 4096 * 2, us, t) for t, us in q if t * 4096 * N * 2 > 1 << 20]
res.append(fit("mnnvl (all_reduce_fusion TP8 h4096 two-shot)", two))
# The four rows span <= 1.8 MB per rank, too little to pin the link: hold it at
# the MNNVL transport's 638 GB/s (moe_alltoall fit to 800 MB, arch/ep_comm.rs).
res.append(fit("mnnvl (two-shot, link held at 638)", two, 638.0))
one = [(t * 4096 * 2, us, t) for t, us in q if t * 4096 * N * 2 <= 1 << 20]
res.append(fit("mnnvl one-shot (reference only)", one))
# NVSHMEM all_reduce 8 GPUs, up to 64 MiB.
q = con.execute("""select message_size_bytes, time_ms*1000 from all_reduce where gpu_name like '%B200%'
  and backend='nvshmem' and num_gpus=8 and is_outlier=0 and message_size_bytes<=67108864 order by 1""").fetchall()
res.append(fit("nvshmem (all_reduce 8 GPU, 4 KiB..64 MiB)", [(2 * (N - 1) / N * m, us, m) for m, us in q]))
# Lamport fused RS + short conv + AG + RMSNorm, TP8 (hidden 6144 until the 4096 rows land).
for h in (4096, 6144):
    q = con.execute(f"""select num_tokens, avg(time_ms)*1000 from rs_sconv_ag_residual_rms_norm where gpu_name like '%B200%'
      and backend='vllm_lamport_triton' and num_gpus=8 and hidden_dim={h} and is_outlier=0 group by num_tokens order by 1""").fetchall()
    if q:
        res.append(fit(f"lamport (rs_sconv_ag_residual_rms_norm TP8 h{h}, has_shared averaged)",
                       [(2 * (N - 1) / N * t * h * 2, us, t) for t, us in q]))
json.dump(res, sys.stdout, indent=1)
