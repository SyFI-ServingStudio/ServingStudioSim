---
name: dev-calculate-kv-cache-capacity
description: "Use when the user asks to calculate, estimate, or sanity-check KV cache size, attention-state size, available KV memory budget, bytes per token, max cached tokens, max concurrent requests, or capacity for a model. Uses a concise three-step workflow: determine available memory after model/reserves, calculate and show per-token or per-request cache/state size from config and public architecture sources, then draw the capacity conclusion. Covers dense MHA/MQA/GQA KV, MLA/DSA/GDN/custom cached states, sliding-window or paged allocation, and MLSim attention memory budgets."
---

# Dev Calculate KV Cache Capacity

Use this skill to answer capacity questions with a short, auditable calculation.
Do not teach every possible config key. Identify the mechanism from the model
config plus README/model card/blog/paper, inspect code only when those sources do
not establish the persistent cache/state tensors, then calculate.

## Workflow

1. Find available memory capacity.

- Use operate-gpu-spec skill to determine the per-GPU HBM size
- Determine the model weight memory occupation. Be careful about the dtype of the model weights

2. Calculate and show cache/state size.

First classify what persists between decode steps:

- dense `MHA`/`MQA`/`GQA`: per-token K and V;
- `MLA`: latent/compressed tensors and any RoPE-side persistent tensors;
- `DSA` or sparse/window attention: dense KV plus sparse metadata, or retained
  sparse tokens only, depending on source evidence;
- `GDN`/linear-attention/state-space-like blocks: recurrent and convolution
  state, often per request rather than per token;
- custom mechanisms: use README/model card/blog/paper as primary evidence, then
  code lookup if persistent tensors or shapes remain unclear.

MUST show the calculation process clearly, step by step, to the user. Search for online resources to validate the calculation. Again, pay attention to dtype of the cache.

3. Draw the capacity conclusion.

Use the available budget and shown cache/state size to answer the user's actual
question.

For per-request recurrent state, divide available bytes by per-request state
bytes. For mixed dense KV plus recurrent state, include both terms in the live
request memory equation.

## Evidence Rules

- Use `top-explore-models` first when the model repo/config is not resolved.
- Treat `config.json`, README/model card, official blog, technical report,
  paper, and release notes as primary evidence.
- Use `dev-lookup-transformers-model` when public/config sources do not prove
  persistent cached tensors or when code contradicts them.
- Do not guess for MLA, DSA, GDN, quantized KV, compressed KV, or custom
  mechanisms. If tensor shapes are unresolved, report the unresolved fields and
  the source needed.

## Report Back

Return:

- available memory calculation and reserve assumptions;
- DETAILED CALCULATION STEPS FOR PER-TOKEN OR PER-REQUEST CACHE/STATE SIZE, with substituted values;
- source evidence used for the attention/cache mechanism;
- per-token or per-request cache/state size, with formula and substituted values;
- raw and page-rounded capacity when relevant;
- final max tokens or max concurrent requests;
- caveats for sharding, mixed layer types, sliding window, quantization metadata,
  speculative decode, or unresolved tensor shapes.
