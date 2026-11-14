"""Vendored unchanged from lane N (wt-glm53-decode-investigate tmp/moe-combine/cb.py, md5 of the source file in
the commit message; jobs 22341/22349/22350): `mhc_post2_tilelang`, a copy of vLLM 5629b2614's `mhc_post_tilelang`
(vllm/model_executor/kernels/mhc/tilelang_kernels.py) with one more bf16 input d2. The kernel computes
x = bf16(float(d) + float(d2)) in-register (fp32 opmath, one RNE rounding, as a torch bf16 add) and then the
fork's post body unchanged, so its output is bit-exact against the fork's add + mhc_post_tilelang."""
from __future__ import annotations

import math

from vllm.model_executor.kernels.mhc.tilelang_kernels import ENABLE_PDL
from vllm.tilelang_utils import T, tilelang, tilelang_jit


@tilelang_jit
def mhc_post2_tilelang(
    a,
    b,
    c,
    d,
    d2,
    x,
    hc: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    b: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    c: T.Tensor((n, hc), T.float32)  # type: ignore[no-redef, valid-type]
    d: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    d2: T.Tensor((n, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    x: T.Tensor((n, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    with T.Kernel(n, threads=n_thr) as i_n:
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)
        d2_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)
        d2_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        if ENABLE_PDL:
            T.pdl_sync()
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Serial(T.ceildiv(h, h_blk)):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)
            T.copy(d2[i_n, i0_h * h_blk], d2_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            T.copy(d2_shared, d2_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                # combine_glue folded: bf16(shared + routed), fp32 opmath, one RNE rounding (= torch bf16 add)
                x_local[i_hco, i1_h] = c_local[i_hco] * T.cast(
                    T.cast(d_local[i1_h] + d2_local[i1_h], T.bfloat16), T.float32
                )
                for i_hci in T.vectorized(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

            T.copy(x_local, x[i_n, 0, i0_h * h_blk])
        if ENABLE_PDL:
            T.pdl_trigger()
