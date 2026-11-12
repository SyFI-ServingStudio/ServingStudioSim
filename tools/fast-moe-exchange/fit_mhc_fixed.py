"""Fit fixed + per-row us of mhc_fused_post_pre_rms_norm (B200, hidden 4096, hc 4).

Relative least squares over the rows 1..max_tokens of one backend; prints a, b
and per-row error. Usage: python fit_mhc_fixed.py [db] [backend] [max_tokens]
"""
import sqlite3, sys
import numpy as np
db = sys.argv[1] if len(sys.argv) > 1 else "profiling/profile.db"
backend = sys.argv[2] if len(sys.argv) > 2 else "deepgemm_mega_nonshifted"
cap = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
q = sqlite3.connect(db).execute(
    """select num_tokens, avg(time_ms)*1000 from mhc_fused_post_pre_rms_norm where gpu_name like '%B200%'
       and backend=? and hidden_size=4096 and hc_mult=4 and is_outlier=0 and num_tokens<=? group by 1 order by 1""",
    (backend, cap)).fetchall()
x = np.array([r[0] for r in q], float); y = np.array([r[1] for r in q], float)
A = np.stack([1 / y, x / y], 1)
(a, b), *_ = np.linalg.lstsq(A, np.ones_like(y), rcond=None)
err = (a + b * x) / y - 1
print(f"backend {backend} rows {len(q)} (1..{int(x.max())}): fixed a = {a:.3f} us, per-row b = {b*1e3:.3f} ns/row; "
      f"rms {100*np.sqrt((err**2).mean()):.1f}%, worst {100*err[np.argmax(abs(err))]:+.1f}%")
for t in (1, 128, 256, 640, 768, 1024, 1280):
    if t <= x.max():
        print(f"  T={t}: measured~{np.interp(t, x, y):.2f}  fit {a + b*t:.2f}  per-row part {b*t:.2f}")
