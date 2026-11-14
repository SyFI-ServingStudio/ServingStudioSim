"""Vendored Triton row-split epilogue (``post_a``), kept byte-identical to its origin.

Origin: wt-glm53-decode-investigate/tmp/dsa-fusion/front/triton_post.py
(md5 68eea2d6be50f56af867b9e27f2d0215), lines 4-30: the imports, ``_post_a``
and ``post_a``. The rest of that file (``post_b``, a different cut) is not
vendored. Do not reformat: ``tests/test_fp32_row_split_layer_norm.py`` pins
these lines to the origin's bytes.

The kernel's row geometry is compiled in: an fp32 row of 2336 values is
[2048 bf16-rounded | 128 LayerNorm | 128 bf16-rounded | 32 left fp32].
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _post_a(y, qkv, k, gs, w, b, eps, BQ: tl.constexpr):
    r = tl.program_id(0)
    base = y + r * 2336
    oq = tl.arange(0, BQ)
    v = tl.load(base + oq)
    tl.store(qkv + r * 2048 + oq, v.to(tl.bfloat16))
    o = tl.arange(0, 128)
    kv = tl.load(base + 2048 + o).to(tl.bfloat16).to(tl.float32)
    mean = tl.sum(kv, 0) / 128.0
    d = kv - mean
    var = tl.sum(d * d, 0) / 128.0
    rs = 1.0 / tl.sqrt(var + eps)
    out = d * rs * tl.load(w + o) + tl.load(b + o)
    tl.store(k + r * 128 + o, out.to(tl.bfloat16))
    g = tl.load(base + 2176 + o)
    tl.store(gs + r * 128 + o, g.to(tl.bfloat16))


def post_a(y, qkv, k, gs, w, b, eps):
    _post_a[(y.shape[0],)](y, qkv, k, gs, w, b, eps, BQ=2048, num_warps=4)

