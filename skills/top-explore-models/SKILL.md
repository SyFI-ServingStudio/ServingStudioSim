---
name: top-explore-models
description: "Use as the top-level entry point when the user asks to understand a new Transformer model or Hugging Face checkpoint before VibeSim modeling work. Answers high-level model questions such as parameter scale, overall architecture, attention type, layer/head/hidden dimensions, context length, MoE layout, multimodal branches, and public architecture notes by collecting Hugging Face metadata/config/model-card evidence first. Route KV cache size/capacity calculations and attention/cache classification for MHA/GQA/MLA/DSA/GDN/custom mechanisms to dev-calculate-kv-cache-capacity. Route deeper Torch implementation, operation-list, forward-path, or kernel-semantics questions to dev-lookup-transformers-model."
---

# Top Explore Models

Use this skill to answer model-level questions before any implementation work:
how large the model is, what the component stack looks like, what attention or
state-space mechanism it uses, which dimensions control shapes, and what public
materials say about the architecture. Work from current evidence, not memory.

This skill does not implement VibeSim support and does not list every Torch
operation. If the request requires local PyTorch code semantics, exact forward
paths, kernel-relevant operation lists, cache behavior, or a runnable Torch
reference, FIRST EXECUTE THIS SKILL, then consider 
read and use `dev-lookup-transformers-model` after the high-level
evidence is collected.

If the request asks for KV cache size, KV capacity, bytes per token, attention
GPU memory pressure, or max cached tokens/requests, FIRST EXECUTE THIS SKILL,
then read and use `dev-calculate-kv-cache-capacity`; that skill owns both
attention/cache classification and capacity math.

## Source to explore

1. Use the exact Hugging Face repo metadata and `config.json` as checkpoint
   ground truth.
2. Use the Hugging Face model card, official release blog, technical report, or
   paper for architecture notes and claimed parameter scale.
3. Explore local Transformers code through `dev-lookup-transformers-model`
   when high-level sources are insufficient.

Download metadata only. Never download model weights unless the user explicitly
asks for that separate action.

## Workflow

1. Resolve the checkpoint.

If the user gives an exact repo id, use it. If they give a family name or an
ambiguous checkpoint name, search Hugging Face and public sources, then choose
only when the match is clear. Ask the user before choosing between plausible
repo ids that would change the answer.

2. Download Hugging Face metadata first.

Run from `VibeSim/` and prefer `uv run python` so the same environment is used as
the rest of VibeSim. Download `config.json` and `README.md`; optionally download
small metadata files such as `generation_config.json` or `tokenizer_config.json`
when they answer the user question. Do not download weights.

```bash
uv run python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download

model_id = "replace_with_exact_repo_id"
api = HfApi()
info = api.model_info(model_id)
print("repo", info.modelId)
print("sha", info.sha)
print("pipeline_tag", info.pipeline_tag)
print("tags", info.tags)
print("safetensors", getattr(info, "safetensors", None))

for filename in ["config.json", "README.md", "generation_config.json", "tokenizer_config.json"]:
    try:
        print(filename, hf_hub_download(model_id, filename=filename))
    except Exception as exc:
        print(filename, "missing_or_unavailable", type(exc).__name__, str(exc).splitlines()[0])
PY
```

If network or Hugging Face access is unavailable, say that the current repo page
was not checked and continue only from local or user-provided evidence.

3. Read the config before summarizing.

Inspect `model_type`, `architectures`, dtype fields, `auto_map`,
`trust_remote_code`, quantization metadata, and nested configs such as
`text_config`, `vision_config`, or `audio_config`. For the text path, identify
the fields that define:

- parameter scale: official parameter count, HF safetensors metadata, or a
  clearly labeled config-derived estimate when no official count is available;
- depth and width: `num_hidden_layers`, `hidden_size`, `intermediate_size`,
  embedding/vocab size, tied or untied output head;
- attention: attention implementation, `num_attention_heads`,
  `num_key_value_heads`, `head_dim`, sliding window, MHA/MQA/GQA/MLA/DSA/GDN,
  masks, layer-type mixes, and cache-relevant config fields;
- position encoding: RoPE theta/scaling, ALiBi, absolute positions, mrope
  sections, max context fields;
- MoE: expert count, experts per token/top-k, router fields, shared experts,
  expert intermediate sizes;
- multimodal branches: vision/audio encoders, projectors, merge layers, and how
  they feed the text path.

Treat values derived from config as inferences. Keep them separate from values
directly stated by the model card, paper, or HF metadata.

4. Search public architecture material.

Search for the exact model id and the model family. Prefer official sources:
the Hugging Face model card, organization blog, technical report, arXiv paper,
release notes, or code repository documentation. Record URLs and publication or
commit dates when available. Use secondary sources only to fill gaps, and label
them as secondary.

5. Answer at the right level.

For normal model-exploration questions, report:

- resolved repo id, HF commit sha, and metadata files read;
- model size, clearly marked as official, HF metadata-backed, or estimated;
- overall architecture: modality, embedding, repeated block, norm placement,
  attention or non-attention block, MLP/MoE, output head;
- key dimensions: layers, hidden size, heads, KV heads, head dim, intermediate
  size, vocab size, context length, expert counts;
- attention and position-encoding type;
- important config flags such as dtype, quantization, `trust_remote_code`, or
  required Transformers version;
- public sources used, plus gaps or contradictions.

For attention/cache classification or KV capacity questions, stop after
gathering the required model/config facts and route to
`dev-calculate-kv-cache-capacity`. Carry forward the source paths and values for
layers, query heads, KV heads, head dim, dtype, context length, layer types,
sliding-window/MLA/DSA/GDN flags, quantization flags, public architecture
sources, and any VibeSim memory budget.

## Routing To Code Exploration

Use `dev-lookup-transformers-model` after this top-level pass when the user asks
for details that cannot be answered reliably from config, model cards, blogs, or
papers. Route examples include:

- "list all operations in the model";
- "which exact PyTorch classes/functions implement attention";
- "what is the QKV projection layout and cache update behavior";
- "extract the smallest Torch reference for this kernel";
- "does the local Transformers version implement this model";
- "which inherited parent model supplies this method".

When entering the lower-level skill, carry forward the resolved repo id,
downloaded config/model-card paths, `model_type`, public sources read, and the
specific unresolved implementation question.


## Routing To KV Cache Capacity

Use `dev-calculate-kv-cache-capacity` only after this skill has collected the
model/config facts needed to choose the right cache model. For dense
MHA/MQA/GQA, carry forward `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `head_dim` or inferred head dim, KV dtype, context length,
page/block size if known, and any VibeSim `attn_gpu_memory_gb` budget.

If the attention type is MLA, DSA, GDN, sliding-window, quantized KV, compressed
KV, or a custom/still-unresolved documented mechanism, route directly to
`dev-calculate-kv-cache-capacity`; it must classify the cached-state model before
calculating capacity. The handoff should include enough evidence to decide
between dense per-token KV, compressed latent KV, sparse retained KV, recurrent
state, or a per-layer mixture.

Do not let KV capacity work skip this top-level evidence pass. The capacity
answer should cite the resolved repo id, config/model-card paths, attention
classification evidence, and every inferred dimension or assumption.
