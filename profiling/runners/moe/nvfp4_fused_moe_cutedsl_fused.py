"""nvfp4_fused_moe / cutedsl_routed_fused_fc1_fc2_sm100.

A routing kernel, then one persistent CuTe DSL launch that runs FC1 (SwiGLU + NVFP4
requant), FC2 and the weighted finalize, overlapped with the router by PDL.

The kernels are a temporary vendored copy of the MoE session's lane P (pfused_vendor/,
see the file headers). Chain construction, inputs and correctness checks follow that
lane's bench_p.py. The chain per token band is fixed (no per-shape search):

    T <= 64           router (tile table)        + fused kernel, 2-tile L2 weight prefetch
    64 < T <= 128     same                       + early first-stage A loads (P_EARLY=6),
                                                   dynamic tile claim after 4 static rounds
    128 < T <= 512    same                       + early loads, dynamic claim after 8 rounds
    512 < T <= 2048   router MODE 7              + fused kernel as for T <= 64,
                                                   dynamic claim after 8 rounds
    T > 2048          router MODE 5 (cl2 table)  + fused kernel on 2-CTA clusters

Lane P round 19 adopted the dynamic claim (p_fused_dyn.py) at T = 128, 256, 512, 1024
and 2048; the band edges between those points are this runner's choice.

Every call checks, outside timing: routing outputs exact; FC1 code and scale factors
bit-exact against the lane's standalone FC1 kernel on host-gathered input; FC2 gates
reset; output cosine >= COS_MIN against an fp32 torch reference built from the
kernel's FC1 output. The timed boundary is the whole chain (router + fused kernel)
under Timer.cupti (interval union, L2 cleared), like flashinfer_trtllm_routed_sm100.
A CUDA-graph replay time goes to a side JSON only, under
$HOME/.cache/vibesim-pfused/side (a profiler container's HOME is its /cache bind
mount, so the files land in that image's host cache directory).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics
from profiling.runners.moe.exact_topk import exact_topk_ids

_VENDOR = Path(__file__).resolve().parent / "pfused_vendor"
E_LOCAL, HIDDEN, INTER, TOPK = 36, 4096, 2048, 8
COS_MIN = 0.99998  # lane used 0.99999; [0.99998, 0.99999) rows are listed apart

# lane P run.sh defaults + bench_p base_env for the fused chains
# (H_ZBYTES=0: the fused kernel zeroes out itself)
_RUN_ENV = dict(G_KTILE=4, G_CSTAGE=2, G_CREFINE=1, N_RVER=1)
_FUSED_BASE = dict(
    H_FUSED=1, H_SCHED=0, G_ABSTAGE=0, H_ZPOS=1, H_PDLTRIG=1, J_VN=1, J_X=1, H_PDLWAIT=1, H_ZBYTES=0
)
_VARIANTS = {  # bench_p P_VARIANTS entries of the adopted chains
    "f_rwepf2_tab": dict(P_GL=32, P_RLX=1, P_ORD=2, P_EVF=1, H_PREPF=1, P_PFN=2),
    "f_rwepf2_tabea6": dict(P_GL=32, P_RLX=1, P_ORD=2, P_EVF=1, H_PREPF=1, P_PFN=2, P_EARLY=6),
    "f_rwepf_cl2tab": dict(P_GL=32, P_RLX=1, P_EVF=1, H_PREPF=1, P_CL=2, P_ORD=2),
    "f_rwepf2_tabea6d4": dict(
        P_GL=32, P_RLX=1, P_ORD=2, P_EVF=1, H_PREPF=1, P_PFN=2, P_EARLY=6, P_DYN=4
    ),
    "f_rwepf2_tabea6d8": dict(
        P_GL=32, P_RLX=1, P_ORD=2, P_EVF=1, H_PREPF=1, P_PFN=2, P_EARLY=6, P_DYN=8
    ),
    "f_rwepf2_tabd8": dict(P_GL=32, P_RLX=1, P_ORD=2, P_EVF=1, H_PREPF=1, P_PFN=2, P_DYN=8),
}


def chain_for(num_tokens: int) -> dict[str, Any]:
    """The fixed chain of a token band.

    Fused variant, router (TB, NT, MODE) and table bits (256 cl1 / 512 cl2).
    """
    t = num_tokens
    if t <= 64:
        return dict(name="tfusedrepf2", variant="f_rwepf2_tab", ocfg=(1, 128, 0), bits=256)
    if t <= 128:
        return dict(
            name="tfusedrepf2ea6d4", variant="f_rwepf2_tabea6d4", ocfg=(1, 128, 0), bits=256
        )
    if t <= 512:
        tb = 2 if t <= 256 and t % 2 == 0 else 1
        return dict(
            name="tfusedrepf2ea6d8", variant="f_rwepf2_tabea6d8", ocfg=(tb, 128, 0), bits=256
        )
    if t <= 2048:
        tb = next(
            (b for b in ((8, 16) if t <= 1024 else (16, 8)) if t % b == 0 and t // b <= 128), None
        )
        if tb is None:
            raise ProfilerNotImplemented(
                f"no MODE 7 router block size for T={t} (needs T % TB == 0, T / TB <= 128)"
            )
        return dict(name="tfusedrepf2d8", variant="f_rwepf2_tabd8", ocfg=(tb, 256, 7), bits=256)
    if t % 2:
        raise ProfilerNotImplemented(f"router TB=2 needs even T, got {t}")
    return dict(name="ufusedrepfcl2", variant="f_rwepf_cl2tab", ocfg=(2, 128, 5), bits=512)


_MODULES: dict[tuple, Any] = {}


def _load(file: str, cfg: dict[str, Any]) -> Any:
    key = (file, tuple(sorted(cfg.items())))
    if key not in _MODULES:
        name = "pfused_" + hashlib.sha1(repr(key).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, _VENDOR / file)
        m = importlib.util.module_from_spec(spec)
        m._PFUSED_CFG = {k: str(v) for k, v in cfg.items()}
        sys.modules[name] = m
        spec.loader.exec_module(m)
        _MODULES[key] = m
    return _MODULES[key]


def _cache_dir(sub: str) -> Path:
    # the checkout is mounted read-only in a profiler container; HOME is writable on both sides
    return Path.home() / ".cache" / "vibesim-pfused" / sub


def profile_nvfp4_fused_moe_cutedsl_fused(**args: Any) -> ComputeMetrics:
    try:
        import torch
        import torch.nn.functional as F

        _load("p_route.py", {})
    except ImportError as exc:
        raise ProfilerNotImplemented(
            f"CuTe DSL / torch extension build unavailable: {exc}"
        ) from exc
    shape = (
        args["hidden_size"],
        args["intermediate_size"],
        args["num_local_experts"],
        args["top_k"],
    )
    if (
        shape != (HIDDEN, INTER, E_LOCAL, TOPK)
        or args["weight_format"] != "nvfp4_e2m1"
        or args["group_size"] != 16
    ):
        raise ProfilerNotImplemented(
            f"lane P kernels are built for H=4096 I=2048 E_local=36 top_k=8 NVFP4/16, got {shape}"
        )
    try:
        return _run(torch, F, args)
    except (ProfilerNotImplemented, AssertionError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise KernelLaunchFailed(f"cutedsl fused MoE failed: {exc!r}") from exc


def _run(torch: Any, F: Any, args: dict[str, Any]) -> ComputeMetrics:
    E, H, II = E_LOCAL, HIDDEN, INTER
    T = int(args["num_tokens"])
    peb = tuple(int(v) for v in args["per_expert_batches"])
    ch = chain_for(T)
    build_dir = str(_cache_dir("route8_build"))
    route_mod = _load("p_route.py", {})
    var = _VARIANTS[ch["variant"]]
    pm = _load(
        "p_fused_dyn.py" if var.get("P_DYN") else "p_fused.py", dict(_RUN_ENV, **_FUSED_BASE, **var)
    )
    # lane bench_p: host FC1 reference module = n_fc1 "vnb" family with the T-dependent zero bytes
    fc1m = _load(
        "n_fc1.py",
        dict(
            _RUN_ENV,
            H_FUSED=1,
            H_SCHED=0,
            G_ABSTAGE=0,
            H_ZBYTES=T * H * 2,
            H_ZPOS=1,
            H_PDLTRIG=1,
            J_VN=1,
            J_X=1,
        ),
    )

    counts = list(peb[:E])
    slots = max(128, -(-max(counts) // 128) * 128)
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    w1 = torch.randint(0, 256, (E, 2 * II, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    sfw1 = torch.randint(
        0x30,
        0x40,
        (E, 2 * II // 128, H // 64, 32, 4, 4),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    alpha1 = (0.5 + torch.rand(E, device="cuda", generator=g)) * 4e-3
    x_src = torch.randint(0, 256, (T, H // 2), dtype=torch.uint8, device="cuda", generator=g)
    xsf_src = torch.randint(0x30, 0x40, (T, H // 16), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (E, H, II // 2), dtype=torch.uint8, device="cuda", generator=g)
    sfw2 = torch.randint(
        0x30, 0x40, (E, H // 128, II // 64, 32, 4, 4), dtype=torch.uint8, device="cuda", generator=g
    )
    w2scale = (0.5 + torch.rand(E, device="cuda", generator=g)) * 4e-3
    logits = torch.randn(T, TOPK, device="cuda", generator=g)
    topk_w = torch.softmax(logits, -1).contiguous()
    dummy1 = torch.empty(E, slots, 2 * II, dtype=torch.bfloat16, device="cuda").permute(2, 1, 0)
    sfw1_flat, sfw2_flat = sfw1.view(E, -1), sfw2.view(E, -1)
    pfb = torch.zeros(32, dtype=torch.int64, device="cuda")
    pfb[16] = w1.data_ptr()
    pfb[17] = sfw1.data_ptr()
    jdummy = torch.zeros(16, dtype=torch.int32, device="cuda")
    orc = route_mod.PRouter(
        T, TOPK, E, H, slots, ch["ocfg"], copy=True, pdl=True, build_dir=build_dir
    )
    rtr = route_mod.PRouter(
        T, TOPK, E, H, slots, ch["ocfg"], copy=True, pdl=True, shared=orc, build_dir=build_dir
    )
    rtr.bits |= ch["bits"]
    gates = torch.zeros(
        2 * 36 * 64 + 8 + 8 * 160 + 8192 + 8 * 160 + 16 * 160 + 64 + 64,
        dtype=torch.int32,
        device="cuda",
    )
    gates[2 * 36 * 64 + 1] = T * H * 2 // 16

    ids_cpu = torch.tensor(
        exact_topk_ids(num_tokens=T, top_k=TOPK, per_expert_batches=peb), dtype=torch.int64
    )
    ids = ids_cpu.int().cuda().contiguous()
    assert torch.bincount(ids_cpu.flatten(), minlength=args["num_experts"])[:E].tolist() == counts
    perm = torch.full((E, slots), -1, dtype=torch.int64)
    route_w = torch.zeros(E, slots, dtype=torch.float32)
    tw_cpu = topk_w.cpu()
    for e in range(E):
        hit = ids_cpu == e
        tok = hit.any(1).nonzero().flatten()
        perm[e, : len(tok)] = tok
        route_w[e, : len(tok)] = tw_cpu[tok, hit[tok].int().argmax(1)]
    route_w = route_w.cuda()
    perm = perm.cuda()
    valid = perm >= 0
    x = torch.zeros(E, slots, H // 2, dtype=torch.uint8, device="cuda")
    x[valid] = x_src[perm[valid]]
    sfx = torch.zeros(E, slots // 128, H // 64, 32, 4, 4, dtype=torch.uint8, device="cuda")
    sfx_flat = sfx.view(E, -1)
    sf_written = torch.zeros_like(sfx_flat, dtype=torch.bool)
    kb = torch.arange(H // 16, device="cuda")
    for e in range(E):
        if counts[e]:
            ix = _sw_index(torch.arange(counts[e], device="cuda"), kb, H // 64).flatten()
            sfx_flat[e][ix] = xsf_src[perm[e, : counts[e]]].flatten()
            sf_written[e][ix] = True
    masked = torch.tensor(counts, dtype=torch.int32, device="cuda")
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    amax = 0.0
    for e in range(E):
        c = counts[e]
        if c:
            Xd = _deq_sw(lut, x[e, :c], sfx_flat[e], c, H)
            Wd = _deq_sw(lut, w1[e], sfw1_flat[e], 2 * II, H)
            G = ((Xd @ Wd.T) * alpha1[e]).view(c, 2 * II // 128, 2, 64)
            amax = max(
                amax,
                float((G[:, :, 0].reshape(c, II) * F.silu(G[:, :, 1].reshape(c, II))).abs().max()),
            )
            del Wd, G
    # a rank with no local rows: nothing to scale; the chain must still run and write zeros
    amax = amax or 1.0
    gs = torch.tensor([448.0 * 6.0 / amax], dtype=torch.float32, device="cuda")
    alpha2 = (w2scale / gs).contiguous()
    h_out = torch.full((E, slots, II // 2), 0xAB, dtype=torch.uint8, device="cuda")
    SFB = E * (slots // 128) * (II // 64) * 512
    sfz = torch.full((SFB + T * H * 2,), 0xCD, dtype=torch.uint8, device="cuda")
    h_sf = sfz[:SFB]
    h_out64 = h_out.view(torch.int64)
    out = sfz[SFB:].view(torch.bfloat16).view(T, H)

    def fc1_host() -> None:
        fc1m._grouped_gemm_nt_masked_sm100(
            (x.permute(1, 2, 0), sfx),
            (w1.permute(1, 2, 0), sfw1),
            dummy1,
            masked,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            is_swap_ab=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            alpha=alpha1,
            alpha_dtype="float32",
            h_fused=(h_out64, h_sf, gs, jdummy, jdummy, pfb),
        )

    def fused() -> None:
        pm._grouped_gemm_nt_masked_sm100(
            (orc.xg.permute(1, 2, 0), orc.sfb),
            (w1.permute(1, 2, 0), sfw1),
            dummy1,
            orc.counts,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            is_swap_ab=True,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(getattr(pm, "_PCL", 1), 1),
            alpha=alpha1,
            alpha_dtype="float32",
            h_fused=(
                h_out64,
                h_sf,
                gs,
                jdummy,
                jdummy,
                pfb,
                w2,
                sfw2,
                h_out,
                h_sf,
                alpha2,
                out,
                orc.perm,
                orc.route_w,
                gates,
            ),
        )

    def route() -> None:
        rtr(ids, topk_w, x_src, xsf_src)

    def chain() -> None:
        route()
        fused()

    # reference FC1 (lane's standalone FC1 kernel on host-gathered rows),
    # then an fp32 FC2 + finalize reference
    out.zero_()
    h_out.fill_(0xAB)
    h_sf.fill_(0xCD)
    fc1_host()
    torch.cuda.synchronize()
    ref_h, ref_sf = h_out.clone(), h_sf.clone()
    oref = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    vmask_h = torch.zeros(E, slots, dtype=torch.bool, device="cuda")
    sfo_written = torch.zeros(E, SFB // E, dtype=torch.bool, device="cuda")
    kb2 = torch.arange(II // 16, device="cuda")
    for e in range(E):
        c = counts[e]
        if not c:
            continue
        vmask_h[e, :c] = True
        sfo_written[e][_sw_index(torch.arange(c, device="cuda"), kb2, II // 64).flatten()] = True
        Hd = _deq_sw(lut, ref_h[e, :c], ref_sf.view(E, -1)[e], c, II)
        W2d = _deq_sw(lut, w2[e], sfw2_flat[e], H, II)
        oref.index_add_(0, perm[e, :c], ((Hd @ W2d.T) * alpha2[e]) * route_w[e, :c, None])
        del W2d

    def reset_gates() -> None:
        gates[: 2 * 36 * 64 + 1].zero_()
        gates[2 * 36 * 64 + 2 : 2 * 36 * 64 + 8].zero_()
        gates[-64:].zero_()
        torch.cuda.synchronize()

    reset_gates()
    orc.perm[~valid] = -1
    route()
    torch.cuda.synchronize()
    chk: dict[str, Any] = dict(
        route_counts_ok=bool(torch.equal(orc.counts, masked)),
        route_perm_ok=bool(torch.equal(orc.perm.long()[valid], perm[valid])),
        route_w_ok=bool(torch.equal(orc.route_w[valid], route_w[valid])),
        route_sfb_ok=bool(torch.equal(orc.sfb.view(E, -1)[sf_written], sfx_flat[sf_written])),
        route_xg_ok=bool(torch.equal(orc.xg[valid], x[valid])),
    )
    h_out.fill_(0xAB)
    h_sf.fill_(0xCD)
    out.fill_(1.0)
    torch.cuda.synchronize()
    fused()
    torch.cuda.synchronize()
    chk.update(
        fc1_code_bitexact=bool(torch.equal(h_out[vmask_h], ref_h[vmask_h])),
        fc1_sf_bitexact=bool(
            torch.equal(h_sf.view(E, -1)[sfo_written], ref_sf.view(E, -1)[sfo_written])
        ),
        out_cos=_cos(F, out, oref),
    )
    time_ms = Timer.cupti(chain, warmup=3, interval_union=True)
    graph_us = _graph_us(torch, chain)
    out.fill_(1.0)
    torch.cuda.synchronize()
    chain()
    torch.cuda.synchronize()
    gb = gates[: 2 * 36 * 64].view(2, -1)
    chk.update(
        chain_out_cos_after=_cos(F, out, oref),
        gates_ok_after=bool(((gb == 0) | (gb == 32)).all()) and int(gb.max(1).values.min()) == 0,
        fc1_code_bitexact_after=bool(torch.equal(h_out[vmask_h], ref_h[vmask_h])),
    )
    ok = (
        all(v for k, v in chk.items() if k.endswith("_ok") or "bitexact" in k)
        and min(chk["out_cos"], chk["chain_out_cos_after"]) >= COS_MIN
    )
    side = dict(
        num_tokens=T,
        per_expert_batches=list(peb),
        chain=ch["name"],
        variant=ch["variant"],
        ocfg=list(ch["ocfg"]),
        slots=slots,
        cupti_us=round(time_ms * 1e3, 3),
        graph_us=graph_us,
        checks=chk,
        correct=ok,
        load=open("/proc/loadavg").read().split()[0],
    )
    sd = _cache_dir("side")
    sd.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(json.dumps([T, peb]).encode()).hexdigest()[:16]
    (sd / f"{T}_{tag}.json").write_text(json.dumps(side))
    if not ok:
        raise KernelLaunchFailed(f"cutedsl fused MoE correctness failed at T={T}: {chk}")
    energy_j = Energy.perf(chain, per_iter_time_ms=time_ms)
    local_rows = sum(counts)
    flops = 2 * local_rows * (H * 2 * II + II * H)
    active = sum(c > 0 for c in counts)
    nbytes = active * (2 * II * H // 2 + H * II // 2) + local_rows * H // 2 + T * H * 2
    s = time_ms / 1e3
    return ComputeMetrics(
        time_ms=time_ms,
        tflops=flops / s / 1e12,
        memory_bandwidth_gbps=nbytes / s / 1e9,
        energy_j=energy_j,
    )


def _sw_index(rows: Any, kblocks: Any, rk_total: int) -> Any:
    r = rows[:, None]
    kb = kblocks[None, :]
    return (((r // 128) * rk_total + kb // 4) * 32 + r % 32) * 16 + ((r % 128) // 32) * 4 + kb % 4


def _deq_sw(lut: Any, q: Any, sf_flat: Any, rows_: int, nk: int) -> Any:
    import torch

    v = torch.stack([lut[(q & 15).long()], lut[(q >> 4).long()]], -1).reshape(*q.shape[:-1], -1)
    rr = torch.arange(rows_, device="cuda")
    s = (
        sf_flat[_sw_index(rr, torch.arange(nk // 16, device="cuda"), nk // 64)]
        .view(torch.float8_e4m3fn)
        .float()
    )
    return (v.view(rows_, -1, 16) * s[..., None]).reshape(rows_, -1)


def _cos(F: Any, out: Any, ref: Any) -> float:
    if not bool(ref.any()):  # no local rows: the output must be all zeros
        return 1.0 if not bool(out.any()) else 0.0
    return round(F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item(), 7)


def _graph_us(torch: Any, fn: Any, n: int = 50, reps: int = 3) -> float:
    gr = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(gr):
        fn()
    vals = []
    for _ in range(reps):
        for _ in range(3):
            gr.replay()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            gr.replay()
        e1.record()
        torch.cuda.synchronize()
        vals.append(e0.elapsed_time(e1) * 1e3 / n)
    return round(sorted(vals)[len(vals) // 2], 2)
