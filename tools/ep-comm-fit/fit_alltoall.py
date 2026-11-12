"""Fit `simulator/src/arch/ep_comm.rs`'s analytic all-to-all on profiled MNNVL rows.

The constants there (DEFAULT_A2A_LATENCY_US, A2A_ROW_FLOOR_GB_PER_S,
A2A_PER_ROW_US, DEFAULT_A2A_LINK_GB_PER_S, A2A_RECV_ROW_WEIGHT,
DEFAULT_PREPARE_LATENCY_US, PREPARE_US_PER_TOKEN) are this script's rounded
output on the B200 EP8 rows of both backends of the same `moe_comm` kernel:
`flashinfer_mnnvl` (FlashInfer 0.6 `trtllm_alltoall`, 2026-10-11; 384 /
3,072 / 12,288-byte rows, 256 slots) and `flashinfer_nvlink_two_sided`
(FlashInfer 0.7 `moe_ep`, 2026-11-12; 256 / 2,048 / 2,368 / 8,192-byte rows
at 288 slots, and 54 repeats of the old shapes at 256 slots).

One moe_comm call on the busiest rank:
    t = a + max(B / c, rows * hypot(p, B / L)),   rows = max(send, w * recv)
a: per-call floor (us); c: the row-width-dependent small-message floor (GB/s);
p: per-row issue cost (ns); L: per-direction link (GB/s); w: weight of a received row.
Fit: relative least squares over EP8 rows with send, recv >= 8 and max/min <= 2
(the shapes the analytic exchange produces), every row width together.
Prepare: t = F + s * tokens + tokens * D8 * 8 top_k / L  (metadata over the link),
relative least squares over every EP8 prepare row (D8 per slot count).
Usage: uv run python -I tools/ep-comm-fit/fit_alltoall.py profiling/profile.db
"""
import sqlite3, sys
import numpy as np

con = sqlite3.connect(sys.argv[1])
BACKENDS = ("flashinfer_mnnvl", "flashinfer_nvlink_two_sided")
every = np.array(con.execute(
    "select hidden_bytes, max_send_rows, max_recv_rows, time_ms*1000, ep_size, slot_count, "
    "direction = 'combine' from moe_alltoall where gpu_name='NVIDIA B200' and top_k=8 and is_outlier=0 "
    "and backend in (?, ?)", BACKENDS).fetchall(), float)
ep8 = every[:, 4] == 8
B, S, R, T, _, SLOT, COMBINE = every[ep8].T
prep = np.array(con.execute(
    "select tokens_per_rank, time_ms*1000, slot_count from moe_alltoall_prepare "
    "where gpu_name='NVIDIA B200' and ep_size=8 and top_k=8 and is_outlier=0 and backend in (?, ?) order by 1",
    BACKENDS).fetchall(), float)


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


def residual_table(x, title):
    """Median / rms / worst relative error by (EP, row width): fit band and every row with both sides >= 8."""
    b_, s_, r_, t_, ep_ = every[:, :5].T
    e = (call(x, b_, s_, r_) - t_) / t_
    inband = (s_ >= 8) & (r_ >= 8) & (np.maximum(s_, r_) <= 2 * np.minimum(s_, r_))
    print(f"\n## {title}: (model - measured) / measured\n")
    print("| EP | row B | band rows | median | rms | worst | rows >= 8 | median | worst |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ep in sorted(set(ep_), reverse=True):
        for b in sorted(set(b_[ep_ == ep])):
            m = (ep_ == ep) & (b_ == b)
            mb, ma = m & inband, m & (s_ >= 8) & (r_ >= 8)
            w = lambda v: v[np.argmax(np.abs(v))] * 100 if len(v) else float("nan")
            med = lambda v: np.median(v) * 100 if len(v) else float("nan")
            rms = np.sqrt(np.mean(e[mb] ** 2)) * 100 if mb.any() else float("nan")
            print(f"| {int(ep)} | {int(b)} | {mb.sum()} | {med(e[mb]):+.1f}% | {rms:.1f}% | {w(e[mb]):+.1f}% | "
                  f"{ma.sum()} | {med(e[ma]):+.1f}% | {w(e[ma]):+.1f}% |")


def fixed_width_table(x, title):
    """The fixed-width exchange's 16 rows: 2,368 B dispatch / 8,192 B combine, EP8, send = recv."""
    print(f"\n## {title}: fixed-width rows (EP8, 288 slots, send = recv), us\n")
    print("| leg | row B | rows | measured | model | err |\n|---|---:|---:|---:|---:|---:|")
    errs = []
    for leg, b, cmb in (("dispatch", 2368, 0), ("combine", 8192, 1)):
        for n in (112, 224, 448, 896, 1792, 2688, 4480, 8960):
            i = np.where((B == b) & (S == n) & (R == n) & (SLOT == 288) & (COMBINE == cmb))[0]
            if len(i) == 0:
                continue
            m = T[i[0]]; p_ = call(x, b, n, n); errs.append(p_ / m - 1)
            print(f"| {leg} | {b} | {n} | {m:.2f} | {p_:.2f} | {(p_/m-1)*100:+.1f}% |")
    if errs:
        errs = np.array(errs)
        print(f"\nfixed-width: median {np.median(errs)*100:+.1f}%, worst {errs[np.argmax(np.abs(errs))]*100:+.1f}%")


PREVIOUS = np.array([4.11, 3.81, 2.55, 638.0, 0.9])
residual_table(PREVIOUS, "Previous constants (2026-10-11 fit)")
residual_table(x, "Refit")
fixed_width_table(PREVIOUS, "Previous constants")
fixed_width_table(x, "Refit")

# Uniform fan-out at EP8, top_k 8 of 256.
def fanout(slots):
    return 8 * (1 - np.prod([(slots * 7 / 8 - i) / (slots - i) for i in range(8)]))


tok, tp, slots = prep.T
D8 = np.array([fanout(sl) for sl in slots])
meta = tok * D8 * 64 / (x[3] * 1e3)
A = np.stack([np.ones_like(tok), tok], 1) / tp[:, None]
F, s = np.linalg.lstsq(A, (tp - meta) / tp, rcond=None)[0]
print(f"prepare: F={F:.3f} us  s={s*1e3:.3f} ns/token (D8 {D8.min():.4f}..{D8.max():.4f})")
F, s = round(F, 1), round(s, 4)
pe = (F + s * tok + meta - tp) / tp
print(f"prepare rms {np.sqrt(np.mean(pe**2)):.3f} max {np.abs(pe).max():.3f}")

print("\n## Diagonal (send = recv), us: measured | model (err)")
for b, name in ((3072, "dispatch x"), (384, "dispatch scales"), (12288, "combine"),
                (2048, "dispatch x, Flash"), (256, "dispatch scales, Flash"), (8192, "combine, Flash")):
    print(f"\n{name} ({b} B/row)")
    for n in (8, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536):
        i = np.where((B == b) & (S == n) & (R == n))[0]
        if len(i) == 0:
            continue
        i = i[0]
        new = call(x, b, n, n)
        print(f"  {n:6d}  {T[i]:8.2f} | {new:8.2f} ({(new/T[i]-1)*100:+5.1f}%)")
print("\n## Off-diagonal sample (send, recv), us: measured | model (err)")
for b in (3072, 384, 12288, 2048, 256, 8192):
    print(f"\n{int(b)} B/row")
    for s_, r_ in ((64, 128), (128, 64), (256, 1024), (1024, 256), (1024, 2048), (2048, 1024), (4096, 8192), (8192, 4096), (1024, 8192), (8192, 1024), (16384, 32768), (32768, 16384)):
        i = np.where((B == b) & (S == s_) & (R == r_))[0]
        if len(i) == 0:
            continue
        i = i[0]; new = call(x, b, s_, r_)
        print(f"  ({s_:5d},{r_:5d})  {T[i]:8.2f} | {new:8.2f} ({(new/T[i]-1)*100:+5.1f}%)")
print("\n## Prepare, us: tokens measured | model (err)")
for n in (1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 24576, 65536):
    for i in np.where(tok == n)[0]:
        new = F + s * n + n * D8[i] * 64 / (x[3] * 1e3)
        print(f"  {n:6d} slots {int(slots[i])}  {tp[i]:8.2f} | {new:8.2f} ({pe[i]*100:+5.1f}%)")
