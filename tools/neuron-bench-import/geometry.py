"""GLM-5.3-Flash geometry and the per-rank args of every Neuron kind (L1_CONTRACT.md).

One place for the shapes the bench rows map to and the grid fill enumerates, mirroring the Rust
arch (``simulator/src/arch/glm53_flash_neuron_fp8.rs``, ``Kernels::build``) and the Rust grids
(``simulator/src/timing/kernels/neuron_grid.rs``).
"""

from __future__ import annotations

H = 4096
KDA_HEADS, KDA_DIM, CONV = 64, 128, 4
MLA_HEADS, KV_LORA, MLA_DIM, Q_LORA = 64, 512, 256, 1536
IDX_HEADS, IDX_DIM, IDX_TOPK = 32, 128, 2048
EXPERTS, TOP_K, MOE_INTER, SHARED = 288, 8, 2048, 1
DENSE_INTER = 12288
HC, SINKHORN = 4, 20
VOCAB = 154880

# Rust neuron_grid.rs
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CHUNK = [128, 256, 512, 1024, 2048, 4096, 8192]
DSA_PREFILL_ROWS = [128, 256, 512, 1024, 2048, 4096, 8192]
DSA_CONTEXT = [2048, 4096, 8192, 16384, 32768]
FFN_ROWS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
HEAD_ROWS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
COMM_BYTES = [16384, 262144, 1048576, 4194304, 16777216]


def kda(tp: int) -> dict:
    return dict(hidden_size=H, num_heads=KDA_HEADS // tp, head_dim=KDA_DIM, conv_kernel_size=CONV,
                dtype="bf16", weight_dtype="bf16")


def dsa(tp: int) -> dict:
    return dict(hidden_size=H, num_heads=MLA_HEADS // tp, kv_lora_rank=KV_LORA, head_dim=MLA_DIM,
                index_heads=IDX_HEADS, index_head_dim=IDX_DIM, index_topk=IDX_TOPK, dtype="bf16",
                weight_dtype="bf16")


def moe(ep: int) -> dict:
    return dict(hidden_size=H, num_experts=EXPERTS, top_k=TOP_K, local_experts=EXPERTS // ep,
                intermediate_size=MOE_INTER, shared_intermediate_size=-(-MOE_INTER * SHARED // ep),
                dtype="bf16", weight_dtype="fp8_e4m3")


def dense(ep: int) -> dict:
    return dict(hidden_size=H, intermediate_size=-(-DENSE_INTER // ep), dtype="bf16",
                weight_dtype="fp8_e4m3")


def router() -> dict:
    return dict(hidden_size=H, num_experts=EXPERTS, top_k=TOP_K, dtype="bf16")


def mhc_pre() -> dict:
    return dict(hidden_size=H, hc_mult=HC, sinkhorn_iters=SINKHORN, dtype="bf16")


def mhc_post() -> dict:
    return dict(hidden_size=H, hc_mult=HC, dtype="bf16")


def head(tp: int) -> dict:
    return dict(hidden_size=H, vocab_size=-(-VOCAB // tp), hc_mult=HC, dtype="bf16")


# kind -> (shape arg names, grid of each, config builder taking (tp, ep))
def series(tps: list[int], world: int) -> list[tuple[str, tuple[str, ...], dict, list[tuple]]]:
    """Every (kind, shape args, config, grid cells) series the arch asks for at these layouts."""
    out = []
    one_d = lambda grid: [(x,) for x in grid]  # noqa: E731
    dsa_prefill_cells = [(r, c) for r in DSA_PREFILL_ROWS for c in DSA_CONTEXT if r <= c]
    dsa_decode_cells = [(b, c) for b in DECODE for c in DSA_CONTEXT]
    for tp in sorted(set(tps)):
        out.append(("kda_prefill", ("num_tokens",), kda(tp), one_d(CHUNK)))
        out.append(("kda_decode", ("batch_size",), kda(tp), one_d(DECODE)))
        out.append(("dsa_prefill", ("num_tokens", "context_len"), dsa(tp), dsa_prefill_cells))
        out.append(("dsa_decode", ("batch_size", "context_len"), dsa(tp), dsa_decode_cells))
        out.append(("lm_head", ("num_tokens",), head(tp), one_d(HEAD_ROWS)))
        out.append(("embed", ("num_tokens",), head(tp), one_d(TOKENS)))
    out.append(("moe_partial", ("num_tokens",), moe(world), one_d(FFN_ROWS)))
    out.append(("dense_ffn_partial", ("num_tokens",), dense(world), one_d(FFN_ROWS)))
    out.append(("moe_router", ("num_tokens",), router(), one_d(FFN_ROWS)))
    out.append(("mhc_pre", ("num_tokens",), mhc_pre(), one_d(TOKENS)))
    out.append(("mhc_post", ("num_tokens",), mhc_post(), one_d(TOKENS)))
    return out


def comm_groups(tps: list[int], world: int) -> list[tuple[int, str]]:
    """(ranks, fabric) of every collective group: each TP group and the world (contiguous)."""
    out = set()
    for n in [t for t in tps if t > 1] + [world]:
        out.add((n, "neuron_c2" if n < 4 else "neuron_c4"))
    return sorted(out)


# Default implementation of each kind (the Rust arch's Glm53NeuronImpls::default()).
DEFAULT_IMPL = {
    "kda_prefill": "torch", "kda_decode": "torch", "dsa_prefill": "radix", "dsa_decode": "radix",
    "moe_partial": ("dense_local", "cte_hybrid"), "dense_ffn_partial": "dense", "moe_router": "fp32",
    "mhc_pre": "torch_bf16", "mhc_post": "torch", "lm_head": "torch", "embed": "torch",
}  # fmt: skip
