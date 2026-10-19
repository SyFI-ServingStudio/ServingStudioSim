"""Launch units of the vLLM Neuron Llama 3.1 8B BF16 TP4 forward, built standalone.

Worker only: this module imports torch and vllm_neuron, so it loads inside the
vendor image. Each unit is one ``torch.nn.Module``. Its forward is exactly one
launch of the production forward at the per-rank shapes of TP4 (hidden 4096,
8 query heads and 2 KV heads of 128 per rank, MLP intermediate 3584 per rank).
Weights and a resident paged KV cache live on the device. Call signatures are
copied by hand from vllm_neuron 0.24.0.1.1.0 (nki_bench records the versions in
each row's provenance). ``build`` returns the module with example inputs for
one shape.

Units:
  prefill.qkv_proj        NKI ``nkilib.core.qkv``: hidden[1,T,H] @ W[H,1536] + RoPE
  prefill.flash_attention NKI ``attention_cte``: q/k[8,128,T], v[8,T,128], causal
  prefill.o_proj          NKI ``output_projection``: [1,8,128,T] @ W[1024,H]
  prefill.mlp / decode.mlp NKI ``mlp``: [rows,H] x gate/up[H,3584], down[3584,H]
  decode.attention_nki    NKI ``attention_block_tkg`` through ``attention_decode``
                          with a per-head block table (the patched Llama3 path)
  prefill.glue / decode.glue  the torch glue of one layer (norms, residuals,
                          layouts, prefill KV write), compiled by neuronx-cc
  noop                    a near-empty NEFF: the fixed per-execution floor
"""

from __future__ import annotations

import os

# torch_xla, which vllm_neuron imports, wants a PJRT device at import time. The
# units run on neuron:0 through libtorch, not PJRT, so CPU only stops the probe.
os.environ.setdefault("PJRT_DEVICE", "CPU")

import torch  # noqa: E402
import vllm_neuron  # noqa: E402,F401  registers the neuron_libtorch backend
import vllm_neuron.functional as NF  # noqa: E402

from profiling.runners.neuron.stock_contract import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE_PER_RANK,
    KV_BLOCKS,
    KV_HEADS_PER_RANK,
    Q_HEADS_PER_RANK,
    QKV_OUT_PER_RANK,
)

DEV = "neuron:0"
DT = torch.bfloat16
H, I_RANK, NQ, NKV, DH = (
    HIDDEN,
    INTERMEDIATE_PER_RANK,
    Q_HEADS_PER_RANK,
    KV_HEADS_PER_RANK,
    HEAD_DIM,
)
QKV_OUT = QKV_OUT_PER_RANK
BLOCK = BLOCK_SIZE
# Production compile options (neuron_model_runner.load_model); vllm-neuron forces
# -O1 on Neuron (vLLM's own default is O2).
COMPILE_OPTS = {
    "alias_meta_to_neuron": True,
    "compiler_args": [
        "--auto-cast=none",
        "--verbose=35",
        "-O1",
        "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
        "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
    ],
}
_GEN = torch.Generator().manual_seed(0)


def rnd(*shape, scale=0.02):
    return (torch.randn(*shape, generator=_GEN) * scale).to(DT)


def rope_tables(positions):
    """Llama 3.1 cos/sin [T, Dh] as LlamaRotaryEmbedding builds them."""
    inv = 1.0 / (500000.0 ** (torch.arange(0, DH, 2).float() / DH))
    wavelen = 2 * torch.pi / inv
    low, high = 8192 / 1.0, 8192 / 4.0  # llama3 scaling: factor 8, low 1, high 4
    smooth = (8192 / wavelen - 1.0) / (4.0 - 1.0)
    inv = torch.where(
        wavelen > low,
        inv / 8.0,
        torch.where(wavelen < high, inv, (1 - smooth) * inv / 8.0 + smooth * inv),
    )
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(DT), emb.sin().to(DT)


class Unit(torch.nn.Module):
    """One launch unit; weights and caches are attributes on the device."""

    def __init__(self, name):
        super().__init__()
        self.name = name


class QkvPrefill(Unit):
    def __init__(self):
        super().__init__("prefill.qkv_proj")
        self.w = rnd(H, QKV_OUT).to(DEV)

    def forward(self, hidden, cos, sin):
        return NF.qkv_proj(
            hidden=hidden.unsqueeze(0),
            qkv_weights=self.w,
            bias=None,
            d_head=DH,
            cos_cache=cos.unsqueeze(0),
            sin_cache=sin.unsqueeze(0),
            num_q_heads=NQ,
            num_kv_heads=NKV,
        ).squeeze(0)


class FlashAttn(Unit):
    def __init__(self):
        super().__init__("prefill.flash_attention")

    def forward(self, q, k, v):
        return NF.flash_attention(q, k, v, scale=DH**-0.5, tp_q=False, tp_out=True)


class OProj(Unit):
    def __init__(self):
        super().__init__("prefill.o_proj")
        self.w = rnd(NQ * DH, H).to(DEV)

    def forward(self, attn):  # [1, Nh, Dh, T] -> [T, H]
        return NF.o_proj(attn, self.w, None).squeeze(0)


class Mlp(Unit):
    def __init__(self, tag):
        super().__init__(f"{tag}.mlp")
        self.gate = rnd(H, I_RANK).to(DEV)
        self.up = rnd(H, I_RANK).to(DEV)
        self.down = rnd(I_RANK, H).to(DEV)

    def forward(self, hidden):  # [rows, H]
        return NF.mlp(hidden, self.gate, self.up, self.down)


def _rms(x, w, eps=1e-5):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


class PrefillGlue(Unit):
    """Every non-kernel torch op of one prefill layer: input RMSNorm, qkv
    split/view/transpose, paged KV write, GQA repeat, flash layouts, residual
    add, post-attention RMSNorm."""

    def __init__(self):
        super().__init__("prefill.glue")
        self.w1 = torch.ones(H, dtype=DT, device=DEV)
        self.w2 = torch.ones(H, dtype=DT, device=DEV)
        self.k_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)
        self.v_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)

    def forward(self, hidden, qkv, attn_out, slot_mapping):
        t = hidden.shape[0]
        h = _rms(hidden, self.w1)
        q, k, v = torch.tensor_split(qkv, [NQ * DH, NQ * DH + NKV * DH], dim=-1)
        q = q.reshape(t, NQ, DH).transpose(0, 1)
        k = k.reshape(t, NKV, DH).transpose(0, 1)
        v = v.reshape(t, NKV, DH).transpose(0, 1)
        blk = (slot_mapping // BLOCK).repeat(NKV)
        head = torch.arange(NKV, device=k.device).repeat_interleave(t)
        pos = (slot_mapping % BLOCK).repeat(NKV)
        self.k_cache.index_put_((blk, head, pos), k.reshape(NKV * t, DH))
        self.v_cache.index_put_((blk, head, pos), v.reshape(NKV * t, DH))
        k = k.repeat_interleave(NQ // NKV, dim=0)
        v = v.repeat_interleave(NQ // NKV, dim=0)
        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.contiguous()
        res = hidden + attn_out
        h2 = _rms(res, self.w2)
        # Return a slice of each layout so neuronx-cc keeps every op live; the
        # compiler may still drop part of a copy, so this unit approximates the
        # model's glue rather than reproducing it.
        return h + h2, q_flash[0, 0, :8], k_flash[0, 0, :8], v_flash[0, 0, :8]


class DecodeGlue(Unit):
    """RMSNorm x2 and residual add x2 of one decode layer at [B, H]."""

    def __init__(self):
        super().__init__("decode.glue")
        self.w1 = torch.ones(H, dtype=DT, device=DEV)
        self.w2 = torch.ones(H, dtype=DT, device=DEV)

    def forward(self, hidden, attn_out, mlp_out):
        h = _rms(hidden, self.w1)
        res = hidden + attn_out
        h2 = _rms(res, self.w2)
        return h + h2 + (res + mlp_out)


class AttnDecodeNki(Unit):
    """Decode attention block through the NKI attention_block_tkg kernel: the
    per-head 3-D block table, in-kernel mask from pos_ids and the caller's KV
    scatter, i.e. the per-head-table Llama3 forward_decode path."""

    def __init__(self):
        super().__init__("decode.attention_nki")
        self.w_qkv = rnd(H, QKV_OUT).to(DEV)
        self.w_out = rnd(NQ * DH, H).to(DEV)
        self.k_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)
        self.v_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)

    def forward(self, x, cos_k, sin_k, block_table, pos_ids, slot_mapping):
        # The per-head-table helpers live under qwen3_vl: the only production
        # GQA decoder that reaches the kernel in vllm-neuron 0.24.0.1.1.0.
        from vllm_neuron.model.qwen3_vl.utils.decode_kv import (
            build_per_head_block_table,
            scatter_new_kv,
        )

        out, k_new, v_new = NF.attention_decode(
            X=x,
            X_hidden_dim_actual=H,
            rmsnorm_X_enabled=False,
            W_qkv=self.w_qkv,
            bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=False,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_k,
            sin=sin_k,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=build_per_head_block_table(block_table, NKV),
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            pos_ids=pos_ids,
            swa_start_pos_ids=None,
            softmax_scale=DH**-0.5,
            sink=None,
            update_cache=False,
            W_out=self.w_out,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )
        scatter_new_kv(self.k_cache, self.v_cache, k_new, v_new, slot_mapping, BLOCK, NKV, DH)
        return out


class Noop(Unit):
    """A near-empty NEFF (one add on 128 elements): the empty-NEFF floor, the
    fixed cost one standalone execution pays. Recorded in provenance, not
    subtracted (decode glue at B1 runs below it)."""

    def __init__(self):
        super().__init__("noop")

    def forward(self, x):
        return x + 1


def _to_dev(*ts):
    return [t.to(DEV) for t in ts]


def scattered_block_table(b, ctx):
    """A block table shaped like live serving. Each request has ctx//32 blocks.
    The width is rounded up to a multiple of 4 columns. Blocks are interleaved
    across requests, int32, -1 for unused columns. Block 0 is vLLM's null block
    and stays free. Large buckets wrap inside the pool: 6781 is prime and above
    any per-request block count, so a request never repeats a block and only
    requests share one."""
    nblk = ctx // BLOCK
    width = -(-nblk // (128 // BLOCK)) * (128 // BLOCK)
    tbl = torch.full((b, width), -1, dtype=torch.int32)
    for r in range(b):
        tbl[r, :nblk] = (torch.arange(nblk) * b + r) % (KV_BLOCKS - 1) + 1
    return tbl


def _decode_rope(b, ctx):
    positions = torch.full((b,), ctx - 1, dtype=torch.int64)
    cos, sin = rope_tables(positions)
    cos_k = cos[:, : DH // 2].view(b, 1, DH // 2).permute(2, 0, 1).contiguous()
    sin_k = sin[:, : DH // 2].view(b, 1, DH // 2).permute(2, 0, 1).contiguous()
    return positions, cos_k, sin_k


def build(unit_name, tokens, ctx):
    """Return ``(module, example_args)`` for one unit at one shape.

    ``tokens`` is the prefill token count T or the decode batch B; ``ctx`` is
    the context the decode block table spans (ignored by prefill units).
    """
    T = B = tokens
    if unit_name == "prefill.qkv_proj":
        cos, sin = rope_tables(torch.arange(T))
        return QkvPrefill(), _to_dev(rnd(T, H, scale=1.0), cos, sin)
    if unit_name == "prefill.flash_attention":
        q, k, v = rnd(NQ, DH, T, scale=1.0), rnd(NQ, DH, T, scale=1.0), rnd(NQ, T, DH, scale=1.0)
        return FlashAttn(), _to_dev(q, k, v)
    if unit_name == "prefill.o_proj":
        return OProj(), _to_dev(rnd(1, NQ, DH, T, scale=1.0))
    if unit_name == "prefill.mlp":
        return Mlp("prefill"), _to_dev(rnd(T, H, scale=1.0))
    if unit_name == "decode.mlp":
        return Mlp("decode"), _to_dev(rnd(B, H, scale=1.0))
    if unit_name == "decode.attention_nki":
        tbl = scattered_block_table(B, ctx)
        positions, cos_k, sin_k = _decode_rope(B, ctx)
        slots = tbl[:, (ctx - 1) // BLOCK].to(torch.int64) * BLOCK + (ctx - 1) % BLOCK
        x = rnd(B, 1, H, scale=1.0)
        pos_ids = positions.view(B, 1).to(torch.float32)
        return AttnDecodeNki(), _to_dev(x, cos_k, sin_k, tbl, pos_ids, slots)
    if unit_name == "prefill.glue":
        slots = torch.arange(T, dtype=torch.int64) + BLOCK  # skip the null block 0
        hidden, qkv, attn = rnd(T, H, scale=1.0), rnd(T, QKV_OUT, scale=1.0), rnd(T, H, scale=1.0)
        return PrefillGlue(), _to_dev(hidden, qkv, attn, slots)
    if unit_name == "decode.glue":
        xs = [rnd(B, H, scale=1.0) for _ in range(3)]
        return DecodeGlue(), _to_dev(*xs)
    if unit_name == "noop":
        return Noop(), _to_dev(rnd(128, scale=1.0))
    raise ValueError(unit_name)


def compile_unit(module):
    """The production compile path: Dynamo full graph through neuron_libtorch."""
    # Each shape is its own graph. The units share module classes, so a batch of
    # many shapes would hit Dynamo's recompile limit. Reset Dynamo before each compile.
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend="neuron_libtorch", options=COMPILE_OPTS)


def sync(out):
    """Force completion: one element copied to the host."""
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o.reshape(-1)[:1].cpu()
