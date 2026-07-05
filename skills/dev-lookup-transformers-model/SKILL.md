---
name: dev-lookup-transformers-model
description: "Use when the user mentions a Transformer model or checkpoint that has not yet been implemented or modeled in MLSim, and you need to locate its architecture semantics from Hugging Face metadata and the local Transformers package installed under main. Helps map a model family or checkpoint name to model_type, architecture components, config parameters, modeling files, class names, kernel-relevant operations, and optional Torch reference code. This is a dev lookup skill and does not edit code."
---

# Dev Lookup Transformers Model

Use this skill when the user mentions a Transformer model or checkpoint that has
not yet been implemented or modeled in MLSim. The goal is to understand the
model architecture from the Hugging Face model metadata and the local
`transformers` install: components, config parameters, operation shapes, and
kernel-relevant paths. A later task may use this to build a small Torch
reference for an operation or an MLSim kernel contract.

## Reality Check

Transformers often has a usable PyTorch reference for mainstream architectures
with official Hugging Face support. Do not assume it always does:

- a very new or private checkpoint may not exist in the installed version;
- some repos rely on `trust_remote_code` rather than built-in Transformers code;
- modeling files may be generated from `modular_*.py`, or may inherit behavior
  from another model family;
- Transformers code gives model semantics, not necessarily the optimized kernel
  or serving backend implementation.

Use this for any Transformer family. Always check the local environment rather
than relying on memory, because installed Transformers versions differ and some
families have several nearby variants.

## Workflow

Run commands from `main/` and use `uv run python`, so you inspect the same
environment MLSim uses.

1. Locate Transformers and list likely model modules.

```bash
uv run python - <<'PY'
import pathlib
import transformers

root = pathlib.Path(transformers.__file__).resolve().parent
needle = "replace_with_family_hint".lower()
print("transformers", transformers.__version__)
print("root", root)
for path in sorted((root / "models").glob(f"*{needle}*")):
    print(path.name)
PY
```

2. Map the requested name to a `model_type`.

Use the auto mappings first. This avoids guessing which nearby architecture
variant a checkpoint belongs to.

```bash
uv run python - <<'PY'
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

needle = "replace_with_family_hint".lower()
for key in sorted(CONFIG_MAPPING_NAMES):
    if needle in key:
        print(key, CONFIG_MAPPING_NAMES[key], MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(key))
PY
```

If the user provided a checkpoint and it is already cached locally, you can also
try `AutoConfig.from_pretrained(model_id, local_files_only=True)`. Do not trigger
a download unless the user asked for network access or model weights.

3. Check the Hugging Face model page for the exact checkpoint.

When the user gives a repo id or a close checkpoint name, use Hugging Face as the
checkpoint-level source of truth. If the repo id is ambiguous, search first and
ask the user before choosing between plausible matches. Download only metadata:
`config.json` and `README.md`, never model weights.

```bash
uv run python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download

query = "replace_with_model_or_family_name"
api = HfApi()
print("matches")
for model in api.list_models(search=query, limit=10):
    print(model.modelId)

model_id = "replace_with_exact_repo_id"
info = api.model_info(model_id)
print("repo", info.modelId)
print("sha", info.sha)
print("pipeline_tag", info.pipeline_tag)
print("tags", info.tags)

for filename in ["config.json", "README.md"]:
    try:
        print(filename, hf_hub_download(model_id, filename=filename))
    except Exception as exc:
        print(filename, "missing_or_unavailable", type(exc).__name__, str(exc).splitlines()[0])
PY
```

Read the downloaded `config.json` before choosing the Transformers module. The
`model_type`, nested `text_config`/`vision_config`, `architectures`, dtype,
expert counts, attention layout, RoPE settings, and any `auto_map` or
`trust_remote_code` fields can override what a family name suggests. Read the
model card for architecture notes, required Transformers version, dtype/GPU
claims, special kernels, and unsupported modes. If network is unavailable or not
authorized, report that the Hugging Face page was not checked and continue with
local cache/package evidence only.

4. Inspect the model directory.

For a chosen `model_type`, open:

- `configuration_<model_type>.py` for `model_type`, hidden size, heads, KV heads,
  head dim, MoE experts, router fields, sliding-window fields, and RoPE settings;
- `modeling_<model_type>.py` for the expanded PyTorch implementation;
- `modular_<model_type>.py` when the modeling file says it was generated from a
  modular source, or when inheritance/diffs are easier to read there.

Useful searches:

```bash
rg -n "model_type|class .*Config|class .*Attention|class .*MLP|class .*RMSNorm|def forward" .venv/lib/python*/site-packages/transformers/models/<model_type>
rg -n "repeat_kv|apply_rotary|eager_attention|SparseMoe|expert|router|topk|gate" .venv/lib/python*/site-packages/transformers/models/<model_type>
```

5. Follow inheritance.

Many model families reuse parents. If the model module imports pieces from
another family, or if the class body is `pass` or only overrides a small method,
follow the import to the parent file and record both paths.

6. Summarize the architecture and kernel-relevant operations.

First build a model-level picture. Record the component stack, such as token
embedding, decoder layer pattern, attention or linear-attention blocks, norm
placement, MLP or MoE blocks, router/shared-expert behavior, vision/audio
branches, and output heads. Tie each component to the config fields that control
its shape or behavior.

Then identify the kernel-relevant operations and their parameters:

- normalization: axis, epsilon, weight convention, dtype cast behavior;
- RoPE or positional encoding: rotary dimension, theta, layout, mrope sections,
  interleaving, position-id convention;
- attention: Q/K/V projection layout, head count, KV heads, head dim, KV repeat,
  mask, softmax dtype, output projection, decode/prefill cache behavior;
- linear attention or state-space blocks: projection layout, conv/recurrent
  state, head dimensions, chunk/recurrent paths, required optional packages;
- MLP: gate/up/down projections, activation, intermediate size;
- MoE: router logits, top-k choice, expert weighting, expert count, shared
  expert, expert tensor layout, grouped GEMM opportunities;
- multimodal branches: patch embedding, merge/projection layers, cross-modal
  token layout, and whether they affect the text path being modeled.

If the later task needs a Torch reference, extract the smallest operation that
establishes the required semantics instead of copying the whole model. Make that
reference runnable with representative tensors and state shape, dtype,
tolerance, edge cases, and unsupported cases.

## Report Back

Return:

- Transformers version and package root;
- requested checkpoint/name and resolved `model_type`;
- Hugging Face repo id, commit sha if checked, and downloaded `config.json` /
  `README.md` paths, or the reason they were not checked;
- config/modeling/modular files read;
- model architecture summary and major components;
- relevant classes/functions and any parent files followed;
- config fields that define shape and dtype semantics;
- kernel-relevant operations and their shape/dtype/capability parameters;
- the smallest candidate Torch reference to use, if one is needed;
- anything not found locally, including whether the model likely needs a newer
  Transformers version or a remote `trust_remote_code` repo.
