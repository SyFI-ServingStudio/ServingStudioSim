"""Compile and verify full Llama 3.1 8B checkpoints through public NxDI APIs.

Run with the isolated Neuron tracing interpreter, not the controller's Torch.
This functional proof does not establish timing alignment with llama3_neuron's
separately compiled, synthetic-weight decoder boundary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import time
from pathlib import Path


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_manifest(directory: Path) -> dict:
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    weights = []
    for name in files:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("checkpoint shard escapes the model directory")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        weights.append({"name": name, "size": path.stat().st_size, "sha256": digest})
    return {
        "directory": str(directory),
        "config": json.loads((directory / "config.json").read_text()),
        "weights": weights,
    }


def package_versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torch-neuronx",
            "torch-xla",
            "libneuronxla",
            "neuronx-cc",
            "neuronx-distributed",
            "neuronx-distributed-inference",
            "transformers",
            "islpy",
        )
    }


def check_model_config(config) -> None:
    expected = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"expected Llama 3.1 8B {field}={value}")
    rope = config.rope_scaling
    if rope.get("rope_type", rope.get("type")) != "llama3" or any(
        rope.get(field) != value
        for field, value in {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }.items()
    ):
        raise ValueError("expected Llama 3.1 scaled RoPE")


def build_config(model_path: Path):
    import torch
    from neuronx_distributed_inference.models.config import NeuronConfig
    from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

    neuron = NeuronConfig(
        batch_size=1,
        seq_len=512,
        max_length=512,
        max_context_length=128,
        context_encoding_buckets=[128],
        token_generation_buckets=[512],
        tp_degree=1,
        logical_nc_config=2,
        torch_dtype=torch.bfloat16,
        padding_side="right",
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        mlp_kernel_enabled=False,
        attn_kernel_enabled=False,
        fused_qkv=False,
        on_cpu=False,
        disable_kv_cache_tiling=True,
        k_cache_transposed=False,
        is_continuous_batching=False,
        target="trn2",
    )
    config = LlamaInferenceConfig(neuron, load_config=load_pretrained_config(str(model_path)))
    check_model_config(config)
    return config


def cases(tokenizer) -> list[tuple[str, dict, int]]:
    prompts = [
        ("factual", "The capital of France is", 8),
        ("numbers", "A list of prime numbers:", 8),
        ("story", "In a hole in the ground there lived", 32),
        ("code", "def fibonacci(n):\n    ", 16),
        ("arithmetic", "1 + 1 =", 8),
        ("json", '{"name": "Alice", "age":', 8),
        ("multiline", "Question: What is the speed of light?\nAnswer:", 8),
        ("minimal", "Hello", 8),
    ]
    result = [
        (name, tokenizer(prompt, return_tensors="pt"), count) for name, prompt, count in prompts
    ]
    # Exercise the largest compiled prefill bucket with identical raw token IDs
    # in both paths, without truncation or tokenizer-dependent prompt lengths.
    long_input = tokenizer(
        "The quick brown fox jumps over the lazy dog. " * 30, return_tensors="pt"
    )
    result.append(("prefill128", {key: value[:, :128] for key, value in long_input.items()}, 8))
    result.append(("reset", result[0][1], result[0][2]))
    return result


def generation_args(tokenizer, new_tokens: int) -> dict:
    return dict(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
    )


def validate(args, config, manifest: dict, versions: dict, device) -> None:
    import torch
    from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    saved_config = json.loads((args.compiled_dir / "neuron_config.json").read_text())
    if saved_config != json.loads(config.to_json_string()):
        raise ValueError("compiled config differs from the requested checkpoint/configuration")
    model = NeuronLlamaForCausalLM(str(args.model_dir), config)
    model.load(str(args.compiled_dir))
    adapter = HuggingFaceGenerationAdapter(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    suite = cases(tokenizer)
    native = []
    for name, inputs, count in suite:
        with torch.inference_mode():
            output = adapter.generate(**inputs, **generation_args(tokenizer, count))
        native.append((output.sequences.tolist(), [x.float().cpu() for x in output.logits]))
        print(f"NEURON {name}: {tokenizer.decode(output.sequences[0])}", flush=True)
    reset_consistent = native[0][0] == native[-1][0]
    del output, adapter, model
    gc.collect()
    reference = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    rows = []
    for (name, inputs, count), (sequence, logits) in zip(suite, native, strict=True):
        with torch.inference_mode():
            output = reference.generate(**inputs, **generation_args(tokenizer, count))
        expected = output.sequences.tolist()
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(sequence[0], expected[0])) if a != b), None
        )
        exact = sequence == expected
        divergence = None
        if mismatch is not None:
            step = mismatch - inputs["input_ids"].shape[1]
            native_scores = logits[step][0]
            reference_scores = output.logits[step][0].float()
            native_token = sequence[0][mismatch]
            reference_token = expected[0][mismatch]
            divergence = dict(
                generated_step=step,
                native_token=native_token,
                reference_token=reference_token,
                native_logits={
                    str(token): float(native_scores[token])
                    for token in (native_token, reference_token)
                },
                reference_logits={
                    str(token): float(reference_scores[token])
                    for token in (native_token, reference_token)
                },
                native_token_tied_for_reference_max=bool(
                    reference_scores[native_token] == reference_scores.max()
                ),
            )
        rows.append(
            dict(
                name=name,
                input_ids=inputs["input_ids"].tolist(),
                new_tokens=count,
                native_sequence=sequence,
                reference_sequence=expected,
                exact_tokens=exact,
                first_divergence=mismatch,
                divergence=divergence,
                # After a token divergence, subsequent contexts differ; these
                # deltas are diagnostics rather than a teacher-forced error bound.
                max_logit_abs_errors=[
                    float((a - b.float()).abs().max())
                    for a, b in zip(logits, output.logits, strict=True)
                ],
                native_text=tokenizer.decode(sequence[0]),
                reference_text=tokenizer.decode(expected[0]),
            )
        )
        print(f"CPU {name}: exact_tokens={exact}, first_divergence={mismatch}", flush=True)
    report = dict(
        checkpoint=manifest,
        packages=versions,
        config=saved_config,
        hardware=dict(
            name=device.gpu_name,
            instance_type=device.instance_type,
            neuron_device=device.device_id,
            logical_core=device.core_ids[0],
            logical_nc_config=device.lnc,
            logical_core_memory_bytes=device.logical_core_memory_bytes,
        ),
        implementation="public NxDI compile/load + HuggingFaceGenerationAdapter",
        reference="Transformers full checkpoint CPU BF16 eager attention",
        reset_sequence_consistent=reset_consistent,
        rows=rows,
        all_exact_tokens=all(row["exact_tokens"] for row in rows),
    )
    write_report(args.report_dir / "validation.json", report)
    if not reset_consistent or not report["all_exact_tokens"]:
        raise AssertionError("checkpoint validation failed; inspect validation.json")
    print("FULL CHECKPOINT VALIDATION PASSED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("compile", "validate"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--compiled-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    for field in ("model_dir", "compiled_dir", "report_dir"):
        setattr(args, field, getattr(args, field).resolve())
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    reservation = None
    try:
        if args.phase == "validate":
            from profiling.exec.neuron import LocalNeuronPool

            reservation = next(LocalNeuronPool().acquire_chunks(k=1))
            os.environ["NEURON_RT_VISIBLE_CORES"] = str(reservation.device.core_ids[0])
        import torch
        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM

        torch.set_num_threads(4)
        versions = package_versions()
        manifest = checkpoint_manifest(args.model_dir)
        config = build_config(args.model_dir)
        if args.phase == "compile":
            if (args.compiled_dir / "model.pt").exists():
                raise FileExistsError("compiled model already exists; use a new output directory")
            start = time.monotonic()
            model = NeuronLlamaForCausalLM(str(args.model_dir), config)
            # Public metadata mode: the AL2023 Torch 2.6 profiler lacks the
            # optional constructor arguments used by newer NxD metadata.
            model.compile(str(args.compiled_dir), debug="none")
            write_report(
                args.report_dir / "compile.json",
                dict(
                    checkpoint=manifest,
                    packages=versions,
                    config=json.loads(config.to_json_string()),
                    wall_seconds=time.monotonic() - start,
                ),
            )
            print("FULL CHECKPOINT COMPILE PASSED", flush=True)
        else:
            validate(args, config, manifest, versions, reservation.device)
    finally:
        if reservation is not None:
            reservation.lock.close()


if __name__ == "__main__":
    main()
