"""Analytic per-rank estimate of each GLM-5.3-Flash Neuron region (one LNC2 core).

Used only to shape DERIVED rows (``grid.py``): a derived cell is this estimate times the
measured/analytic ratio of nearby measured cells, so the estimate's absolute level matters only
where a kind has no measured row at all. Roofline with fixed efficiencies:

    time = max(flops / (EFF_PE * 158 TF/s), bytes / (EFF_HBM * 725 GB/s)) + REGION_FLOOR_MS

The Trainium2 LNC2 peaks are the catalog's (``gpu/spec.json`` ``Trainium2-LNC2``; 725 GB/s is an
estimate). FP8 weights are upcast in-graph, so their bytes count once as FP8 read plus a BF16
write and read of the upcast copy. fp32 work (router, KDA recurrence, mHC Sinkhorn) runs at a
quarter of the BF16 rate.
"""

from __future__ import annotations

import geometry as G

PEAK_BF16 = 158e12
HBM = 725e9
EFF_PE = 0.6
EFF_HBM = 0.6
REGION_FLOOR_MS = 0.05
BF16, FP32 = 2, 4


def _t(flops: float, nbytes: float, fp32_flops: float = 0.0) -> float:
    pe = (flops + 4 * fp32_flops) / (EFF_PE * PEAK_BF16)
    mem = nbytes / (EFF_HBM * HBM)
    return max(pe, mem) * 1e3 + REGION_FLOOR_MS


def estimate(kind: str, a: dict) -> tuple[float, float, float]:
    """(time_ms, flops, bytes) of one call."""
    H = G.H
    fp32 = 0.0
    if kind in ("kda_prefill", "kda_decode"):
        n = a.get("num_tokens") or a.get("batch_size")
        hl = a["num_heads"]
        kh = hl * G.KDA_DIM
        w = 4 * H * kh + 2 * G.KDA_DIM * kh + H * hl + 3 * kh * G.CONV + 2 * H * G.KDA_DIM
        flops = 2 * n * w
        state = hl * G.KDA_DIM * G.KDA_DIM
        if kind == "kda_prefill":
            fp32 = 8 * n * state  # chunked delta rule
            nbytes = w * BF16 + n * H * BF16 * 2 + 2 * state * FP32
        else:
            fp32 = 6 * n * state
            nbytes = w * BF16 + n * H * BF16 * 2 + 2 * n * state * FP32
    elif kind in ("dsa_prefill", "dsa_decode"):
        n = a.get("num_tokens") or a.get("batch_size")
        ctx = a["context_len"]
        hl = a["num_heads"]
        ih = G.IDX_HEADS * G.IDX_DIM
        w_rep = H * G.Q_LORA + H * G.KV_LORA + G.Q_LORA * ih + H * G.IDX_DIM + H * G.IDX_HEADS
        w_sh = G.Q_LORA * hl * G.MLA_DIM + G.KV_LORA * hl * 2 * G.MLA_DIM + hl * G.MLA_DIM * H
        sel = min(G.IDX_TOPK, ctx)
        attn = n * hl * sel * 2 * G.KV_LORA * 2
        absorb = n * hl * 2 * G.MLA_DIM * G.KV_LORA * 2
        scores = 2 * n * ctx * ih
        flops = 2 * n * (w_rep + w_sh) + attn + absorb + scores
        cache_row = (G.KV_LORA + 2 * G.IDX_DIM) * BF16
        reads = ctx * cache_row if kind == "dsa_prefill" else n * ctx * cache_row
        nbytes = (w_rep + w_sh) * BF16 + reads + n * H * BF16 * 2
    elif kind == "moe_partial":
        n = a["num_tokens"]
        le, inter, sh = a["local_experts"], a["intermediate_size"], a["shared_intermediate_size"]
        routed_rows = n * G.TOP_K / (G.EXPERTS / le) * 1.3  # load imbalance on the critical rank
        flops = 2 * (routed_rows * 3 * H * inter + n * 3 * H * sh)
        fp32 = 2 * n * H * G.EXPERTS
        w = le * 3 * H * inter + 3 * H * sh
        nbytes = w * (1 + 2 * BF16) + n * H * BF16 * 3 + G.EXPERTS * H * FP32
    elif kind == "dense_ffn_partial":
        n = a["num_tokens"]
        w = 3 * H * a["intermediate_size"]
        flops = 2 * n * w
        nbytes = w * (1 + 2 * BF16) + n * H * BF16 * 2
    elif kind == "moe_router":
        n = a["num_tokens"]
        flops = 0.0
        fp32 = 2 * n * H * G.EXPERTS
        nbytes = G.EXPERTS * H * FP32 + n * H * BF16
    elif kind == "mhc_pre":
        n = a["num_tokens"]
        d = G.HC * H
        flops = 2 * n * d * G.HC * (2 + G.HC)
        fp32 = n * (G.SINKHORN * 2 * G.HC * G.HC + 4 * d)
        nbytes = n * d * (BF16 + FP32 * 2) + n * H * BF16
    elif kind == "mhc_post":
        n = a["num_tokens"]
        d = G.HC * H
        flops = 0.0
        fp32 = n * d * (2 * G.HC + 2)
        nbytes = n * d * (BF16 + FP32 * 2 + BF16) + n * H * BF16
    elif kind == "lm_head":
        n, v = a["num_tokens"], a["vocab_size"]
        flops = 2 * n * H * v
        nbytes = v * H * BF16 + n * v * FP32 + n * G.HC * H * BF16
    elif kind == "embed":
        n, v = a["num_tokens"], a["vocab_size"]
        flops = 0.0
        nbytes = n * H * BF16 * (1 + G.HC) + n * 8
    else:
        raise KeyError(kind)
    return _t(flops, nbytes, fp32), flops + fp32, nbytes
