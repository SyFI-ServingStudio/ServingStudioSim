"""Fit `simulator/src/arch/ep_comm.rs`'s analytic all-to-all on profiled MNNVL rows.

The constants there (DEFAULT_A2A_LATENCY_US, A2A_ROW_FLOOR_GB_PER_S,
A2A_PER_ROW_US, DEFAULT_A2A_LINK_GB_PER_S, A2A_RECV_ROW_WEIGHT,
DEFAULT_PREPARE_LATENCY_US, PREPARE_US_PER_TOKEN) are this script's rounded
output on the B200 EP8 flashinfer_mnnvl rows profiled 2026-10-11.

One moe_comm call on the busiest rank:
    t = a + max(B / c, rows * hypot(p, B / L)),   rows = max(send, w * recv)
a: per-call floor (us); c: the row-width-dependent small-message floor (GB/s);
p: per-row issue cost (ns); L: per-direction link (GB/s); w: weight of a received row.
Fit: relative least squares over rows with send, recv >= 8 and max/min <= 2
(the shapes the analytic exchange produces), all three row widths together.
Prepare: t = F + s * tokens + tokens * D8 * 8 top_k / L  (metadata over the link),
relative least squares over all 68 prepare rows.
Usage: uv run python -I tools/ep-comm-fit/fit_alltoall.py profiling/profile.db
"""
import sqlite3, sys
import numpy as np

con = sqlite3.connect(sys.argv[1])
rows = np.array(con.execute(
    "select hidden_bytes, max_send_rows, max_recv_rows, time_ms*1000 from moe_alltoall "
    "where gpu_name='NVIDIA B200' and ep_size=8 and top_k=8 and slot_count=256 and is_outlier=0").fetchall(), float)
B, S, R, T = rows.T
prep = np.array(con.execute(
    "select tokens_per_rank, time_ms*1000 from moe_alltoall_prepare "
    "where gpu_name='NVIDIA B200' and ep_size=8 and top_k=8 and slot_count=256 and is_outlier=0 order by 1").fetchall(), float)


def nelder_mead(f, x0, iters=40000):
    n = len(x0)
    simplex = [np.array(x0, float)] + [np.array(x0, float) * (1 + 0.2 * np.eye(n)[i]) for i in range(n)]
    vals = [f(x) for x in simplex]
    for _ in range(iters):
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(simplex[:-1], axis=0)
        xr = c + (c - simplex[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - simplex[-1]); fe = f(xe)
            simplex[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = xc, fc
            else:
                simplex = [simplex[0] + 0.5 * (x - simplex[0]) for x in simplex]
                vals = [f(x) for x in simplex]
    return simplex[int(np.argmin(vals))]


def call(x, b, s, r):
    a, c, p, L, w = x
    rows = np.maximum(s, w * r)
    return a + np.maximum(b / (c * 1e3), rows * np.hypot(p * 1e-3, b / (L * 1e3)))


band = (S >= 8) & (R >= 8) & (np.maximum(S, R) <= 2 * np.minimum(S, R))
obj = lambda x: np.sum(((call(x, B, S, R) - T) / T)[band] ** 2) if np.all(x > 0) else 1e9
x = nelder_mead(obj, [4.0, 4.0, 2.5, 640.0, 0.9])
a, c, p, L, w = x
print(f"call: a={a:.3f} us  c={c:.3f} GB/s  p={p:.4f} ns  L={L:.1f} GB/s  w={w:.4f}")
x = np.array([round(a, 2), round(c, 2), round(p, 2), round(L), round(w, 2)])
print("rounded:", x)
e = (call(x, B, S, R) - T) / T
for b in (3072, 384, 12288):
    m = B == b
    print(f"B={int(b):5d} band rms {np.sqrt(np.mean(e[m & band]**2)):.3f} max {np.abs(e[m & band]).max():.3f} | "
          f"all rows>=8 rms {np.sqrt(np.mean(e[m & (S>=8) & (R>=8)]**2)):.3f} max {np.abs(e[m & (S>=8) & (R>=8)]).max():.3f}")

# Uniform fan-out at EP8, top_k 8 of 256.
miss = np.prod([(224 - i) / (256 - i) for i in range(8)])
D8 = 8 * (1 - miss)
tok, tp = prep.T
meta = tok * D8 * 64 / (x[3] * 1e3)
A = np.stack([np.ones_like(tok), tok], 1) / tp[:, None]
F, s = np.linalg.lstsq(A, (tp - meta) / tp, rcond=None)[0]
print(f"prepare: F={F:.3f} us  s={s*1e3:.3f} ns/token (D8={D8:.4f})")
F, s = round(F, 1), round(s, 4)
pe = (F + s * tok + meta - tp) / tp
print(f"prepare rms {np.sqrt(np.mean(pe**2)):.3f} max {np.abs(pe).max():.3f}")

print("\n## Diagonal (send = recv), us: measured | model (err)")
for b, name in ((3072, "dispatch x"), (384, "dispatch scales"), (12288, "combine")):
    print(f"\n{name} ({b} B/row)")
    for n in (8, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536):
        i = np.where((B == b) & (S == n) & (R == n))[0][0]
        new = call(x, b, n, n)
        print(f"  {n:6d}  {T[i]:8.2f} | {new:8.2f} ({(new/T[i]-1)*100:+5.1f}%)")
print("\n## Off-diagonal sample (send, recv), us: measured | model (err)")
for b in (3072, 384, 12288):
    print(f"\n{int(b)} B/row")
    for s_, r_ in ((64, 128), (128, 64), (256, 1024), (1024, 256), (1024, 2048), (2048, 1024), (4096, 8192), (8192, 4096), (1024, 8192), (8192, 1024), (16384, 32768), (32768, 16384)):
        i = np.where((B == b) & (S == s_) & (R == r_))[0]
        if len(i) == 0:
            continue
        i = i[0]; new = call(x, b, s_, r_)
        print(f"  ({s_:5d},{r_:5d})  {T[i]:8.2f} | {new:8.2f} ({(new/T[i]-1)*100:+5.1f}%)")
print("\n## Prepare, us: tokens measured | model (err)")
for n in (1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 24576, 65536):
    i = np.where(tok == n)[0][0]
    new = F + s * n + n * D8 * 64 / (x[3] * 1e3)
    print(f"  {n:6d}  {tp[i]:8.2f} | {new:8.2f} ({pe[i]*100:+5.1f}%)")
