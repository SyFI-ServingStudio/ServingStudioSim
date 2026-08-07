# Surrogate model configs

These configs approximate non-GQA frontier models with the `qwen3_moe`
GQA+MoE arch so their kernel shapes can be profiled (L1) and simulated today,
ahead of native MLA/DSA/KDA op support. MoE dims are exact (or best-published);
attention is a **KV-byte-equivalent GQA stand-in** at FlashInfer-supported head
dims. Known deltas are listed so results are read with the right sign.

## glm52_surrogate.json — GLM-5.2 (753B/40B-active)

Real (HF `zai-org/GLM-5.2`): 78 layers, hidden 6144, 64 q-heads x 192,
MLA kv_lora_rank 512 + qk_rope 64, DSA `index_topk` 2048, 256 routed experts
top-8 (+1 shared), moe_intermediate 2048, vocab 154880, first 3 layers dense.

Surrogate mapping and deltas:
- Exact: layers, hidden, MoE (256 experts, top-8, moe_int 2048), vocab.
- q width preserved: 96 heads x 128 = 12288 = real 64 x 192 (same qkv/o GEMMs).
- KV: 2 kv-heads x 128 -> 1024 B/layer/tok vs true MLA 1152 B (-11%).
- Dense attention: no DSA top-2048 cap -> decode attention cost is an UPPER
  bound at long context (real GLM-5.2 decodes cheaper than simulated).
- Omitted: the shared expert (+~6% expert FLOPs), first-3-dense layers.

## kimi_k3_surrogate.json — Kimi K3 (2.8T, ESTIMATED)

Published facts only: 2.8T total, 896 experts top-16 (~50B active), KDA with
every-4th-layer MLA, 1M context, MXFP4 weights. Layer dims are NOT published
(weights due 2026-07-27); dims here are scaled from Kimi K2 (hidden 7168,
moe_int 2048, vocab 163840): 70 layers x 896 x 3 x 7168 x 2048 = 2.76T ~ 2.8T,
active 16/896 -> ~49B.

Surrogate mapping and deltas:
- KV: 1 kv-head x 128 -> 512 B/layer/tok, 35.8 KB/tok total, vs ~24 KB/tok
  estimated for true KDA + 1/4-MLA (+~50% conservative on KV footprint).
- KDA's constant-state linear attention is modeled as dense MQA over the (small)
  KV -> decode attention cost again an upper bound at long context.
- Re-derive everything when the real config.json lands.

Both run bf16 in the sim (like the qwen3 baseline) so all three models share
one profiled backend set; real deployments would serve fp8/MXFP4, which scales
absolute throughput but not config rankings.
