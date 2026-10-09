"""Production NxDI layer boundary with independent, untimed Llama math checks.

This deliberately compiles one layer. The simulator composes those measured
invocations; it does not claim the fusion or latency of NxDI's full-model graph.
Imports stay inside the isolated Neuron worker.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import math

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_VERSIONS = {
    "torch": "2.6.0+cpu",
    "torch-neuronx": "2.6.0.2.10.16998+e9bf8a50",
    "torch-xla": "2.6.1",
    "neuronx-cc": "2.27.5334.0+f702b353",
    "neuronx-distributed": "0.19.28492+435aae2b",
    "neuronx-distributed-inference": "0.9.0+4bcdc54b.dev",
    "transformers": "4.57.6",
    "islpy": "2024.2",
}


def validate_shape(
    phase, batch, q_tokens, kv_capacity, hidden, intermediate, q_heads, kv_heads, head_dim, dtype
):
    if DType.from_value(dtype) != DType.BF16:
        raise ProfilerNotImplemented("Neuron Llama decoder requires BF16 compute and KV")
    if (batch, hidden, intermediate, q_heads, kv_heads, head_dim) != (1, 4096, 14336, 32, 8, 128):
        raise ProfilerNotImplemented(
            "Neuron Llama decoder initially supports the TP1 Llama 3.1 8B dimensions"
        )
    if kv_capacity != 512:
        raise ProfilerNotImplemented("Neuron Llama decoder initially requires KV capacity 512")
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be prefill or decode")
    if not 1 <= q_tokens <= 128 or (phase == "decode" and q_tokens != 1):
        raise ValueError("initial prefill queries require 1..128 rows; decode requires one query")


def _make_model(torch, phase, q_tokens, kv_capacity):
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        LlamaInferenceConfig,
        NeuronLlamaDecoderLayer,
    )
    from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager

    values = {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        "tie_word_embeddings": False,
        "pad_token_id": 128001,
    }
    neuron = NeuronConfig(
        batch_size=1,
        seq_len=kv_capacity,
        max_length=kv_capacity,
        max_context_length=kv_capacity,
        n_active_tokens=q_tokens,
        tp_degree=1,
        logical_nc_config=2,
        torch_dtype=torch.bfloat16,
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_kernel_enabled=False,
        fused_qkv=False,
        on_cpu=False,
        disable_kv_cache_tiling=True,
        k_cache_transposed=False,
        is_continuous_batching=False,
        layer_boundary_markers=True,
        target="trn2",
    )
    neuron.is_prefill_stage = phase == "prefill"
    config = LlamaInferenceConfig(neuron, **values)

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = NeuronLlamaDecoderLayer(config).to(torch.bfloat16).eval()
            # Generic nonuniform gamma prevents the synthetic compiler probe
            # from specializing both normalization multiplies to identity.
            with torch.no_grad():
                for norm in (self.layer.input_layernorm, self.layer.post_attention_layernorm):
                    norm.weight.copy_(1 + 0.1 * torch.randn_like(norm.weight))
            self.cache = KVCacheManager(config, 8)

        def forward(self, hidden, attention_mask, positions):
            k, v = self.cache.past_key_values[:2]
            outputs = self.layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=positions,
                past_key_value=None if phase == "prefill" else (k, v),
                kv_mgr=self.cache,
                update_kv_per_layer=True,
                idx=0,
                is_for_context_encoding=phase == "prefill",
                seq_ids=torch.arange(hidden.shape[0], dtype=torch.int32),
                seq_len=kv_capacity,
                kvcache_buffer=[k, v],
            )
            return outputs[0], outputs[1][0], outputs[1][1]

    return Decoder().eval(), values


def _make_inputs(torch, phase, q_tokens, kv_capacity):
    hidden = torch.randn(1, q_tokens, 4096, dtype=torch.bfloat16)
    k = torch.randn(1, 8, kv_capacity, 128, dtype=torch.bfloat16) * 0.2
    v = torch.randn_like(k) * 0.2
    if phase == "prefill":
        positions = torch.arange(q_tokens).reshape(1, q_tokens).to(torch.int64)
        mask = torch.ones(q_tokens, q_tokens, dtype=torch.bool).tril()
        mask = mask.reshape(1, 1, q_tokens, q_tokens).expand(1, 32, -1, -1)
    else:
        positions = torch.full((1, 1), 127, dtype=torch.int64)
        mask = (torch.arange(kv_capacity) < 127).reshape(1, 1, 1, kv_capacity)
        mask = mask.expand(1, 32, 1, -1)
    return hidden, mask.contiguous(), positions.contiguous(), k, v


def independent_oracle(torch, model, inputs, phase):
    """Functional math, independent of timed forward/cache implementation."""
    h, mask, positions, old_k, old_v = inputs
    layer = model.layer

    def norm(x, weight):
        z = x.float()
        return (z * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-5) * weight.float()).to(
            torch.bfloat16
        )

    z = norm(h, layer.input_layernorm.weight)
    qkv = layer.self_attn.qkv_proj
    q = (
        torch.nn.functional.linear(z, qkv.q_proj.weight)
        .reshape(1, h.shape[1], 32, 128)
        .transpose(1, 2)
    )
    k = (
        torch.nn.functional.linear(z, qkv.k_proj.weight)
        .reshape(1, h.shape[1], 8, 128)
        .transpose(1, 2)
    )
    v = (
        torch.nn.functional.linear(z, qkv.v_proj.weight)
        .reshape(1, h.shape[1], 8, 128)
        .transpose(1, 2)
    )
    inv = 1.0 / (500000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128.0))
    wavelength = 2 * math.pi / inv
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    inv = torch.where(
        wavelength < 2048.0,
        inv,
        torch.where(wavelength > 8192.0, inv / 8.0, (1 - smooth) * inv / 8.0 + smooth * inv),
    )
    angles = positions.float().unsqueeze(-1) * inv
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos().to(torch.bfloat16).unsqueeze(1)
    sin = angles.sin().to(torch.bfloat16).unsqueeze(1)

    def rotate(t):
        first, second = t.chunk(2, dim=-1)
        return t * cos + torch.cat([-second, first], dim=-1) * sin

    q, k = rotate(q), rotate(k)
    kr, vr = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    active_scores = (q @ kr.transpose(-1, -2)) / math.sqrt(128)
    if phase == "prefill":
        scores = active_scores.masked_fill(~mask, torch.finfo(torch.bfloat16).min)
        attention = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16) @ vr
    else:
        cached_k = old_k.repeat_interleave(4, dim=1)
        cached_v = old_v.repeat_interleave(4, dim=1)
        past_scores = ((q @ cached_k.transpose(-1, -2)) / math.sqrt(128)).masked_fill(
            ~mask,
            torch.finfo(torch.bfloat16).min,
        )
        probs = torch.softmax(
            torch.cat([past_scores.float(), active_scores.float()], dim=-1), dim=-1
        ).to(torch.bfloat16)
        attention = probs[..., : old_k.shape[2]] @ cached_v + probs[..., old_k.shape[2] :] @ vr
    attention = attention.transpose(1, 2).reshape_as(h)
    projected = torch.nn.functional.linear(attention, layer.self_attn.o_proj.o_proj.weight)
    residual = h + projected
    z = norm(residual, layer.post_attention_layernorm.weight)
    gate = torch.nn.functional.linear(z, layer.mlp.gate_proj.weight)
    up = torch.nn.functional.linear(z, layer.mlp.up_proj.weight)
    mlp = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, layer.mlp.down_proj.weight
    )
    output = residual + mlp
    new_k, new_v = old_k.clone(), old_v.clone()
    if phase == "prefill":
        new_k[:, :, : h.shape[1], :] = k
        new_v[:, :, : h.shape[1], :] = v
    else:
        new_k[:, :, int(positions[0, 0]), :] = k[:, :, 0, :]
        new_v[:, :, int(positions[0, 0]), :] = v[:, :, 0, :]
    return output, new_k, new_v


def profile_llama_decoder(
    phase, batch, q_tokens, kv_capacity, hidden, intermediate, q_heads, kv_heads, head_dim, dtype
):
    validate_shape(
        phase,
        batch,
        q_tokens,
        kv_capacity,
        hidden,
        intermediate,
        q_heads,
        kv_heads,
        head_dim,
        dtype,
    )
    versions = {name: importlib.metadata.version(name) for name in _VERSIONS}
    if versions != _VERSIONS:
        raise ProfilerNotImplemented(
            f"Neuron Llama compiler path requires validated versions {_VERSIONS}; found {versions}"
        )
    import ml_dtypes
    import numpy as np
    import torch
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaModel
    from neuronx_distributed_inference.models.model_wrapper import (
        CONTEXT_ENCODING_MODEL_TAG,
        TOKEN_GENERATION_MODEL_TAG,
        ModelWrapper,
    )
    from torch_neuronx.xla_impl.trace import generate_hlo, generate_neff

    from profiling.profilers.neuron_timer import _artifact_directory, measure_neff

    torch.manual_seed(42)
    model, values = _make_model(torch, phase, q_tokens, kv_capacity)
    inputs = _make_inputs(torch, phase, q_tokens, kv_capacity)
    with torch.no_grad():
        model.cache.past_key_values[0].copy_(inputs[3])
        model.cache.past_key_values[1].copy_(inputs[4])
        reference = independent_oracle(torch, model, inputs, phase)
    flags = ModelWrapper(
        model.layer.config,
        NeuronLlamaModel,
        tag=CONTEXT_ENCODING_MODEL_TAG if phase == "prefill" else TOKEN_GENERATION_MODEL_TAG,
    ).compiler_args
    artifacts = _artifact_directory(
        {
            "source": inspect.getsource(_make_model),
            "versions": versions,
            "values": values,
            "phase": phase,
            "batch": batch,
            "q_tokens": q_tokens,
            "kv_capacity": kv_capacity,
            "compiler_args": flags,
            "seed": 42,
            "lnc": 2,
            "target": "trn2",
            "input_output_aliases": {"K": 1, "V": 2},
        }
    )
    neff = artifacts / "graph.neff"
    if not neff.exists():
        # These are the trace API's verified compilation stages. Avoid creating
        # a Torch runtime model just to extract a NEFF for the native timer.
        aliases = {model.cache.past_key_values[0]: 1, model.cache.past_key_values[1]: 2}
        hlo = generate_hlo(
            model,
            inputs[:3],
            input_output_aliases=aliases,
            cpu_backend=True,
            inline_weights_to_neff=True,
        )
        generate_neff(
            hlo, compiler_workdir=str(artifacts), compiler_args=flags, inline_weights_to_neff=True
        )
        (artifacts / "metaneff.pb").write_bytes(hlo.metaneff.SerializeToString())
    arrays = {
        f"input{i}": x.float().numpy().astype(ml_dtypes.bfloat16)
        if x.dtype == torch.bfloat16
        else x.numpy()
        for i, x in enumerate(inputs)
    }

    def check(outputs):
        for output, target in zip(outputs, reference, strict=True):
            np.testing.assert_allclose(
                output.reshape(target.shape).astype(np.float32),
                target.float().numpy(),
                rtol=0.05,
                atol=0.05,
            )

    time_ms = measure_neff(
        neff,
        arrays,
        check,
        output_dtype=ml_dtypes.bfloat16,
        output_names=("output0", "output1", "output2"),
        expected_aliases={"output1": "input3", "output2": "input4"},
    )
    # Count dense projections plus the two attention matmuls. Transcendentals
    # and logical traffic inside the compiler graph are not estimated here.
    weights = (
        hidden * (q_heads + 2 * kv_heads) * head_dim
        + hidden * q_heads * head_dim
        + 3 * hidden * intermediate
    )
    attention_length = q_tokens if phase == "prefill" else kv_capacity + 1
    flops = 2 * q_tokens * weights + 4 * q_tokens * q_heads * head_dim * attention_length
    return ComputeMetrics(time_ms, flops / (time_ms / 1000) / 1e12, 0.0)
