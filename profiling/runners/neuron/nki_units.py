"""Launch units of the vLLM Neuron Llama 3.1 8B BF16 TP4 forward, built standalone.

Worker only: this module imports torch and vllm_neuron, so it loads inside the
vLLM Neuron 0.29 image (Neuron SDK 2.33). Each unit is one ``torch.nn.Module``.
Except ``DecodeForward`` (a reduced-depth TP4 forward, see ``forward_bench``)
and ``noop``, its forward is one launch of the production forward at the per-rank
shapes of TP4 (hidden 4096, 8 query heads and 2 KV heads of 128 per rank, MLP
intermediate 3584 per rank). Weights and a resident paged KV cache live on the
``neuron`` device. ``build`` returns the module with example inputs for one
shape; ``compile_unit`` compiles it the way the serving engine compiles its
forward: one Dynamo full graph through vLLM Neuron's compile backend
(``vllm_neuron.envs.get_compile_backend_name()``, the native
``libtorch_neuronx_lite`` route) with the serving compiler flags.

The functional calls are those of ``vllm_neuron/model/llama3/model.py`` at
release-0.29.0.1.2.0 (``NF.qkv_proj``, ``NF.flash_attention``, ``NF.o_proj``,
``NF.mlp``, ``NF.attention_decode``); their keyword sets were checked against
the installed release; the worker records the image's package versions in
each row's provenance (``profiling/exec/local_worker.py``).

Units:
  prefill.qkv_proj        nkilib ``qkv``: hidden[1,T,H] @ W[H,1536] + RoPE
  prefill.flash_attention nkilib ``attention_cte``: q/k[8,128,T], v[8,T,128], causal
  prefill.o_proj          nkilib ``output_projection_cte``: [1,8,128,T] @ W[1024,H]
  prefill.mlp / decode.mlp nkilib ``mlp``: [rows,H] x gate/up[H,3584], down[3584,H]
  decode.attention_nki    ``attention_decode`` with the per-head (3-D) block table,
                          which routes GQA to the nkilib ``attention_block_tkg``
                          kernel; ``update_cache=False`` and the caller's paged
                          scatter of the new K/V, as the production per-head
                          decoders (qwen3_vl) issue it
  decode.attention_torch  ``attention_decode`` as the stock Llama3 forward_decode
                          calls it: 2-D block table, explicit decode mask,
                          ``update_cache=True``; at TP4 the torch fallback
  decode.forward_nki / decode.forward_torch
                          ``DecodeForward``: the TP4 decode forward with n layers on
                          four ranks (forward_bench), the two decode leaves
  prefill.glue / decode.glue  the torch glue of one layer (norms, residuals,
                          layouts, prefill KV write), compiled by neuronx-cc
  prefill.head / decode.head  embedding gather, final norm, lm_head shard, argmax
  noop                    a near-empty graph: the fixed per-execution floor
"""

from __future__ import annotations

import os

# torch_xla, which the image still ships, requires a PJRT device at import time.
# The units run on neuron:0 through the lite runtime, so CPU only stops the probe.
os.environ.setdefault("PJRT_DEVICE", "CPU")

import torch  # noqa: E402
import vllm_neuron  # noqa: E402,F401  registers the compile backends and the device
import vllm_neuron.envs as vllm_neuron_envs  # noqa: E402
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
    VOCAB_PER_RANK,
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
# Production compile options (NeuronModelRunnerV2.load_model): the native route
# defaults to -O2; `--logical-nc-config` and `--target` come from the
# reservation environment through the lite backend.
COMPILE_OPTS = {
    "alias_meta_to_neuron": True,
    "compiler_args": [
        "--auto-cast=none",
        "--verbose=35",
        "-O2",
        "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
        "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
    ],
}
_GEN = torch.Generator().manual_seed(0)


def compile_backend() -> str:
    """The torch.compile backend the serving engine uses under the configured
    route (`VLLM_NEURON_BACKEND`); the native route registers it on demand."""
    return vllm_neuron_envs.get_compile_backend_name()


def rnd(*shape, scale=0.02):
    return (torch.randn(*shape, generator=_GEN) * scale).to(DT)


def rope_tables(positions):
    """Llama 3.1 cos/sin [T, Dh] as LlamaRotaryEmbedding builds them."""
    freqs = positions.float()[:, None] * llama3_inv_freq()[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(DT), emb.sin().to(DT)


def llama3_inv_freq():
    """The Llama 3.1 scaled RoPE inverse frequencies [Dh / 2], float32."""
    inv = 1.0 / (500000.0 ** (torch.arange(0, DH, 2).float() / DH))
    wavelen = 2 * torch.pi / inv
    # Llama 3.1 rope_scaling (checked by validate_checkpoint_math): factor 8,
    # low_freq_factor 1, high_freq_factor 4, original context 8192; low/high
    # below are the wavelength bounds.
    low, high = 8192 / 1.0, 8192 / 4.0
    smooth = (8192 / wavelen - 1.0) / (4.0 - 1.0)
    inv = torch.where(
        wavelen > low,
        inv / 8.0,
        torch.where(wavelen < high, inv, (1 - smooth) * inv / 8.0 + smooth * inv),
    )
    return inv


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
        return NF.flash_attention(
            q, k, v, scale=DH**-0.5, tp_q=False, tp_out=True, allow_vector_exp=False
        )


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
        # compiler can still drop part of a copy, so this unit approximates the
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


class Head(Unit):
    """Embedding gather of the step's rows, final RMSNorm of the sampled rows,
    the vocab-sharded lm_head and a greedy argmax over the local logits."""

    def __init__(self, tag):
        super().__init__(f"{tag}.head")
        self.embed = rnd(VOCAB_PER_RANK, H).to(DEV)
        self.norm_w = torch.ones(H, dtype=DT, device=DEV)
        self.lm_head = rnd(H, VOCAB_PER_RANK).to(DEV)

    def forward(self, token_ids, hidden, sampled_rows):
        gathered = self.embed[token_ids]  # [rows, H]
        h = _rms(hidden[sampled_rows], self.norm_w)  # [sampled, H]
        logits = h @ self.lm_head  # [sampled, VOCAB/tp]
        # Every gathered row feeds an output, so the compiler keeps the whole gather.
        return logits.argmax(dim=-1), gathered.float().sum(0)[:8]


class AttnDecodeNki(Unit):
    """The decode attention block on the NKI kernel path: the per-head 3-D
    block table that routes GQA to ``attention_block_tkg`` (the kernel reads
    the paged cache itself), the in-kernel mask from pos_ids, and the caller's
    paged scatter of the returned new K/V, as the production GQA decoders that
    take this path issue it (``qwen3_vl`` ``forward_decode``). A Llama3
    ``forward_decode`` patched to build the per-head table launches this."""

    def __init__(self):
        super().__init__("decode.attention_nki")
        self.w_qkv = rnd(H, QKV_OUT).to(DEV)
        self.w_out = rnd(NQ * DH, H).to(DEV)
        self.k_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)
        self.v_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)

    def forward(self, x, cos_k, sin_k, block_table, pos_ids, slot_mapping):
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


class AttnDecodeTorch(Unit):
    """The decode attention block as the stock Llama3 ``forward_decode`` calls
    ``NF.attention_decode``: the 2-D block table (at TP4, two KV heads per
    rank, the functional routes it to its torch fallback), the explicit decode
    mask, ``update_cache=True`` with the new token's slot. The mask is built on
    every call here; the engine builds it once per forward."""

    def __init__(self):
        super().__init__("decode.attention_torch")
        self.w_qkv = rnd(H, QKV_OUT).to(DEV)
        self.w_out = rnd(NQ * DH, H).to(DEV)
        self.k_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)
        self.v_cache = torch.zeros(KV_BLOCKS, NKV, BLOCK, DH, dtype=DT, device=DEV)

    def forward(self, x, cos_k, sin_k, block_table, pos_ids, slot_idx):
        b = block_table.shape[0]
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=pos_ids,
            bs=b,
            q_head=NQ,
            s_active=1,
            s_prior=block_table.shape[1] * BLOCK,
            start_pos=None,
            block_len=BLOCK,
            local_filled_slots=None,
            dcp_active_mask=None,
        )
        return NF.attention_decode(
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
            active_blocks_table=block_table,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            attention_mask=attention_mask,
            softmax_scale=DH**-0.5,
            sink=None,
            update_cache=True,
            kv_cache_update_idx=slot_idx,
            fp8_packed=False,
            W_out=self.w_out,
            bias_out=None,
            transposed_out=False,
            out_in_sb=False,
        )


class Noop(Unit):
    """A near-empty graph (one add on 128 elements): the fixed cost one
    standalone execution pays. Recorded in provenance, not subtracted."""

    def __init__(self):
        super().__init__("noop")

    def forward(self, x):
        return x + 1


class DecoderLayers(Unit):
    """``n`` production decoder layers in one graph: per layer the input
    RMSNorm, the decode attention block on one path (``attention_cls``), the
    residual, the post-attention RMSNorm, the MLP and the residual. Each layer
    owns its weights and its paged KV cache, as the engine's layers do. This
    is the body of ``DecodeForward``: each block's output is all-reduced over
    the TP group in the graph, as in the engine's decode graph."""

    def __init__(self, n, attention_cls, tag):
        super().__init__(f"decode.layers_{tag}")
        self.attn = torch.nn.ModuleList([attention_cls() for _ in range(n)])
        self.mlp = torch.nn.ModuleList([Mlp("decode") for _ in range(n)])
        self.norm_w = torch.ones(H, dtype=DT, device=DEV)

    @staticmethod
    def _reduce(x):
        import torch.distributed as dist
        import torch.distributed._functional_collectives as fc

        return fc.all_reduce(x, "sum", dist.group.WORLD)

    def forward(self, x, cos, sin, block_table, pos_ids, slot):
        """``cos``/``sin`` are the step's [B, Dh] RoPE tables; each layer
        derives its own kernel layout from them in the graph, as the engine's
        ``forward_decode`` does."""
        h = x
        b = x.shape[0]
        half = DH // 2
        for attn, mlp in zip(self.attn, self.mlp, strict=True):
            cos_k = cos[:, :half].view(b, 1, half).permute(2, 0, 1).contiguous().to(DT)
            sin_k = sin[:, :half].view(b, 1, half).permute(2, 0, 1).contiguous().to(DT)
            hn = _rms(h, self.norm_w)
            a = attn(hn.view(h.shape), cos_k, sin_k, block_table, pos_ids, slot).view(h.shape)
            h = h + self._reduce(a)
            hn = _rms(h, self.norm_w)
            h = h + self._reduce(mlp(hn.view(-1, H)).view(h.shape))
        return h


class DecodeForward(Unit):
    """A reduced-depth TP4 decode forward with the engine's structure, one
    process per rank. The embedding is vocab-sharded: a masked local lookup,
    then an all-reduce of the [B, H] rows (``VocabDimShardedEmbedding``
    without sequence parallelism). Then ``n`` decoder layers reduce each
    block in the graph. The final RMSNorm, the rank's lm_head shard and the
    engine's sampler follow. The sampler is ``NF.sample`` with the engine's
    default on-device config (``all_greedy=False``, ``max_top_k=256``), so the
    graph runs the distributed top-k path even for temperature-0 rows. Two
    depths of ``t(n) = fixed + n * layer`` give the per-layer and the per-step
    terms, both with every TP collective in the graph."""

    def __init__(self, n, attention_cls, tag):
        super().__init__(f"decode.forward_{tag}")
        self.layers = DecoderLayers(n, attention_cls, tag)
        self.embed = rnd(VOCAB_PER_RANK, H).to(DEV)
        self.norm_w = torch.ones(H, dtype=DT, device=DEV)
        self.lm_head = rnd(H, VOCAB_PER_RANK).to(DEV)
        self.inv_freq = llama3_inv_freq().to(DEV)

    def forward(self, token_ids, positions, block_table, pos_ids, slot):
        import torch.distributed as dist
        import torch.distributed._functional_collectives as fc

        group = dist.group.WORLD
        start = dist.get_rank() * VOCAB_PER_RANK
        local = token_ids - start
        mask = (local >= 0) & (local < VOCAB_PER_RANK)
        rows = self.embed[torch.where(mask, local, 0)]
        rows = torch.where(mask.unsqueeze(-1), rows, torch.zeros_like(rows))
        h = fc.all_reduce(rows, "sum", group).view(-1, 1, H)
        # the step's RoPE tables, computed in the graph from the positions
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        h = self.layers(h, emb.cos(), emb.sin(), block_table, pos_ids, slot)
        logits = _rms(h.view(-1, H), self.norm_w) @ self.lm_head
        batch = logits.shape[0]
        return NF.sample(
            logits,
            temperature=torch.zeros(batch, dtype=torch.float32, device=logits.device),
            top_k=-1,
            top_p=1.0,
            all_greedy=False,
            tp_group=group,
            max_top_k=256,
            tp_rank=torch.full((1,), dist.get_rank(), dtype=torch.int32, device=logits.device),
        )


def _to_dev(*ts):
    return [t.to(DEV) for t in ts]


def scattered_block_table(b, ctx):
    """A block table shaped like the engine's: ``[B, max_blocks_per_seq]`` with
    ``max_blocks_per_seq * block_size == ctx``, int32. Blocks are interleaved
    across requests as in live serving. Block 0 is vLLM's null block and stays
    free. Large buckets wrap inside the pool: 6781 is prime and above any
    per-request block count, so a request never repeats a block and only
    requests share one."""
    nblk = ctx // BLOCK
    tbl = torch.empty((b, nblk), dtype=torch.int32)
    for r in range(b):
        tbl[r] = (torch.arange(nblk) * b + r) % (KV_BLOCKS - 1) + 1
    return tbl


def _decode_rope(b, ctx):
    positions = torch.full((b,), ctx - 1, dtype=torch.int64)
    cos, sin = rope_tables(positions)
    cos_k = cos[:, : DH // 2].view(b, 1, DH // 2).permute(2, 0, 1).contiguous()
    sin_k = sin[:, : DH // 2].view(b, 1, DH // 2).permute(2, 0, 1).contiguous()
    return positions, cos_k, sin_k


def _decode_attention_args(attention_cls, b, ctx):
    """The example inputs of one decode attention block (and of a layer stack
    built on it): every request at full context, block tables scattered as in
    live serving, the new token's slot. The two paths differ in the layout of
    ``pos_ids`` and of the slot index, as their production callers pass them."""
    tbl = scattered_block_table(b, ctx)
    positions, cos_k, sin_k = _decode_rope(b, ctx)
    slots = tbl[:, (ctx - 1) // BLOCK].to(torch.int64) * BLOCK + (ctx - 1) % BLOCK
    x = rnd(b, 1, H, scale=1.0)
    if attention_cls is AttnDecodeNki:
        pos_ids = positions.view(b, 1).to(torch.float32)
        return _to_dev(x, cos_k, sin_k, tbl, pos_ids, slots)
    # the new token's slot, as the engine's slot_mapping.view(B, 1).to(uint32)
    pos_ids = positions.view(1, b).to(torch.float32)
    return _to_dev(x, cos_k, sin_k, tbl, pos_ids, slots.view(b, 1).to(torch.uint32))


def build(unit_name, tokens, ctx):
    """Return ``(module, example_args)`` for one unit at one shape.

    ``tokens`` is the prefill token count T or the decode batch B; ``ctx`` is
    the context the decode block table spans (ignored by prefill units). The
    TP4 decode forward (``DecodeForward``) is built by ``forward_bench``.
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
        return AttnDecodeNki(), _decode_attention_args(AttnDecodeNki, B, ctx)
    if unit_name == "decode.attention_torch":
        return AttnDecodeTorch(), _decode_attention_args(AttnDecodeTorch, B, ctx)
    if unit_name == "prefill.glue":
        slots = torch.arange(T, dtype=torch.int64) + BLOCK  # skip the null block 0
        hidden, qkv, attn = rnd(T, H, scale=1.0), rnd(T, QKV_OUT, scale=1.0), rnd(T, H, scale=1.0)
        return PrefillGlue(), _to_dev(hidden, qkv, attn, slots)
    if unit_name == "decode.glue":
        xs = [rnd(B, H, scale=1.0) for _ in range(3)]
        return DecodeGlue(), _to_dev(*xs)
    if unit_name == "prefill.head":
        ids = torch.randint(0, VOCAB_PER_RANK, (T,), generator=_GEN)
        sampled = torch.tensor([T - 1], dtype=torch.int64)
        return Head("prefill"), _to_dev(ids, rnd(T, H, scale=1.0), sampled)
    if unit_name == "decode.head":
        ids = torch.randint(0, VOCAB_PER_RANK, (B,), generator=_GEN)
        sampled = torch.arange(B, dtype=torch.int64)
        return Head("decode"), _to_dev(ids, rnd(B, H, scale=1.0), sampled)
    if unit_name == "noop":
        return Noop(), _to_dev(rnd(128, scale=1.0))
    raise ValueError(unit_name)


def compile_unit(module):
    """The production compile path: one Dynamo full graph through the engine's
    compile backend with the serving compiler flags."""
    # Each shape is its own graph. The units share module classes, so a batch of
    # many shapes would hit Dynamo's recompile limit. Reset Dynamo before each compile.
    torch._dynamo.reset()
    return torch.compile(module, fullgraph=True, backend=compile_backend(), options=COMPILE_OPTS)


def sync(out):
    """Force completion: one element copied to the host."""
    o = out[0] if isinstance(out, (tuple, list)) else out
    return o.reshape(-1)[:1].cpu()
