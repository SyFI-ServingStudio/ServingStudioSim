"""Vendored Triton Hadamard quant + head-weight scale epilogue (``post_b``), kept byte-identical to its origin.

Origin: wt-glm53-decode-investigate/tmp/dsa-fusion/front/triton_post.py
(md5 68eea2d6be50f56af867b9e27f2d0215), lines 4-6 (imports) and 32-78: ``_fwht_stage``,
``_post_b`` and ``post_b`` (MoE-session lane F cut B lite, jobs 22122 / 22147). The rest of
that file (``post_a``, cut A) is vendored separately in profiling/runners/norm/row_split_vendor.
Do not reformat: ``tests/test_hadamard_fp8_quant_weight_scale.py`` pins these lines to the
origin's bytes.

The head geometry is compiled in: 32 heads of 128 columns per token (``rows // 32``) and the
weight scale 128^-0.5 * 32^-0.5 = 0.015625.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------- cut B lite: FWHT + ue8m0 quant + weight scale
@triton.jit
def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _post_b(y, ldy, col0, hw, ldhw, qout, sout, wout, n_rows, BLOCK_R: tl.constexpr):
    """Fork _fwht_quant_kernel on the strided wq_b columns of the concatenated GEMM output (row r = token*32 + head),
    plus the fork weight scale: wts = head_w * q_scale * 128^-0.5 * 32^-0.5."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    tok = rows // 32
    hd = rows % 32
    offs = tl.arange(0, 128)
    x = tl.load(y + tok[:, None] * ldy + col0 + hd[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    N: tl.constexpr = BLOCK_R * 128
    x = tl.reshape(x, (N,))
    x = _fwht_stage(x, N, BLOCK_R * 64, 1)
    x = _fwht_stage(x, N, BLOCK_R * 32, 2)
    x = _fwht_stage(x, N, BLOCK_R * 16, 4)
    x = _fwht_stage(x, N, BLOCK_R * 8, 8)
    x = _fwht_stage(x, N, BLOCK_R * 4, 16)
    x = _fwht_stage(x, N, BLOCK_R * 2, 32)
    x = _fwht_stage(x, N, BLOCK_R, 64)
    x = x * 0.08838834764831845
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.reshape(x, (BLOCK_R, 128))
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
    yq = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
    tl.store(qout + rows[:, None] * 128 + offs[None, :], yq, mask=rmask[:, None])
    tl.store(sout + rows, scale, mask=rmask)
    w = tl.load(hw + tok * ldhw + hd, mask=rmask, other=0.0)
    tl.store(wout + rows, w * scale * 0.015625, mask=rmask)


def post_b(y, col0, hw, qout, sout, wout):
    n = y.shape[0] * 32
    BLOCK_R = 32
    _post_b[(triton.cdiv(n, BLOCK_R),)](y, y.stride(0), col0, hw, hw.stride(0), qout, sout, wout, n, BLOCK_R=BLOCK_R, num_warps=2)
