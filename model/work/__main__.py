"""CLI: minimum necessary work for a model + workload.

    uv run python -m model.work model/config/llama3_8b.json --decode 256x4096
    uv run python -m model.work model/config/llama3_8b.json --prefill 8192 --gpu B200
    uv run python -m model.work <config> --batch-file batch.json --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .core import Workload
from .registry import build_model


def _parse_decode(spec: str) -> Workload:
    # "BxS": B decode tokens, each attending to kv_len S.
    batch_str, _, kv_str = spec.partition("x")
    if not kv_str:
        raise SystemExit(f"--decode expects BxS (e.g. 256x4096), got {spec!r}")
    batch, kv_len = int(batch_str), int(kv_str)
    return Workload.causal_lm(decode=[kv_len] * batch, sampled=batch)


def _parse_prefill(spec: str) -> Workload:
    # "P" (from empty) or "P@C" (append P onto C already-cached tokens).
    append_str, _, prefix_str = spec.partition("@")
    append_len = int(append_str)
    prefix_len = int(prefix_str) if prefix_str else 0
    return Workload.causal_lm(prefill=[(append_len, prefix_len)], sampled=1)


def _parse_batch_file(path: str) -> Workload:
    with open(path) as handle:
        raw = json.load(handle)
    return Workload.causal_lm(
        prefill=[tuple(pair) for pair in raw.get("prefill", [])],
        decode=raw.get("decode", []),
        sampled=raw.get("sampled"),
    )


def _workload(args: argparse.Namespace) -> Workload:
    given = [args.decode, args.prefill, args.batch_file]
    if sum(value is not None for value in given) != 1:
        raise SystemExit("give exactly one of --decode / --prefill / --batch-file")
    if args.decode is not None:
        return _parse_decode(args.decode)
    if args.prefill is not None:
        return _parse_prefill(args.prefill)
    return _parse_batch_file(args.batch_file)


def _fmt_count(value: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3f}{unit}"
    return f"{value:.0f}"


def _fmt_bytes(value: float) -> str:
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.3f} {unit}"
    return f"{value:.0f} B"


def _describe_attn(attn) -> str:
    if hasattr(attn, "num_qo_heads"):  # GQA (optionally gated)
        gate = "+gate" if getattr(attn, "output_gate", False) else ""
        return f"GQA{gate} q{attn.num_qo_heads}/kv{attn.num_kv_heads} d{attn.head_dim}"
    if hasattr(attn, "num_v_heads"):  # GatedDeltaNet linear attention
        return f"GDN v{attn.num_v_heads}/k{attn.num_k_heads} d{attn.head_k_dim}"
    return type(attn).__name__


def _describe_layers(model) -> str:
    parts = []
    for stack in model.layers:
        tag = stack.tag or type(stack.attn).__name__
        parts.append(f"{stack.count}x[{tag}: {_describe_attn(stack.attn)}]")
    return "  ".join(parts)


def _print_human(raw_config: dict, model, workload: Workload, label, gpu: str, dtype: str,
                 num_gpus: int) -> None:
    print(f"model:  {model.name}   L={model.num_layers}  h={model.hidden}  V={model.vocab}")
    print(f"layers: {_describe_layers(model)}")
    breakdown = label.params["breakdown"]
    activated = label.params["activated"]
    print(f"params: total={_fmt_count(label.params['total'])}   "
          f"activated: layers-only={_fmt_count(activated['layers'])}  "
          f"+embed+head={_fmt_count(activated['with_embed_head'])}")
    print(f"  breakdown: attn={_fmt_count(breakdown['attn'])} "
          f"ffn={_fmt_count(breakdown['ffn'])} experts={_fmt_count(breakdown['experts'])} "
          f"router={_fmt_count(breakdown['router'])} shared={_fmt_count(breakdown['shared'])} "
          f"embed={_fmt_count(breakdown['embedding'])} head={_fmt_count(breakdown['lm_head'])}")
    print(f"batch:  matmul_tokens={workload.matmul_tokens}  "
          f"head_positions={workload.head_positions}  interactions={len(workload.attn)}")

    print("\n── min compute (FLOPs) ──")
    for key, value in label.flops.items():
        print(f"  {key:<16}{_fmt_count(value):>12}")
    print(f"  {'TOTAL':<16}{_fmt_count(label.flops_total):>12}")

    print("\n── min memory (bytes) ──")
    for key, value in label.bytes.items():
        print(f"  {key:<16}{_fmt_bytes(value):>14}")
    print(f"  {'TOTAL':<16}{_fmt_bytes(label.bytes_total):>14}")

    compute_ms, memory_ms, bound = label.roofline_ms(gpu, dtype, num_gpus)
    global_floor = max(compute_ms, memory_ms)
    segmented = label.segmented_lower_bound_ms(gpu, dtype, num_gpus)
    rows = label.segment_rows(gpu, dtype, num_gpus)
    tokens = workload.matmul_tokens

    # Each segment names its own precision; `dtype` only fills in for the ones
    # that do not, so print what was actually used rather than the fallback.
    used_dtypes = sorted({row["compute_dtype"] for row in rows})
    print(f"\n── roofline ({gpu}, {'+'.join(used_dtypes)}, {num_gpus} GPU) ──")
    print(f"  global floor (full fusion)  {global_floor:9.4f} ms   [{bound}-bound]")
    print(f"  segmented lower bound       {segmented:9.4f} ms   <- realistic (Σ per-op)")
    if segmented > 0 and tokens > 0:
        print(f"  ceiling throughput          {tokens / segmented * 1e3:>12,.0f} tok/s")

    print(f"\n    {'op':<12}{'FLOPs':>10}{'bytes':>12}{'dtype':>7}{'bound':>9}{'ms':>10}")
    for row in rows:
        print(f"    {row['name']:<12}{_fmt_count(row['flops']):>10}"
              f"{_fmt_bytes(row['bytes']):>12}{row['compute_dtype']:>7}"
              f"{row['bound']:>9}{row['ms']:>10.4f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m model.work", description=__doc__)
    parser.add_argument("config", help="path to a HuggingFace config.json")
    parser.add_argument("--decode", help="BxS: B decode tokens each at kv_len S")
    parser.add_argument("--prefill", help="P or P@C: append P tokens onto C cached")
    parser.add_argument("--batch-file", help="JSON {prefill:[[app,pre]], decode:[kv], sampled}")
    parser.add_argument("--gpu", default="H200", help="GPU name for the roofline (default H200)")
    parser.add_argument("--dtype", default=None, help="roofline dtype (default: config's)")
    parser.add_argument("--num-gpus", type=int, default=1, help="GPUs the work is spread over")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    with open(args.config) as handle:
        raw_config = json.load(handle)
    model = build_model(raw_config)
    workload = _workload(args)
    label = model.label(workload)
    dtype = args.dtype or raw_config.get("torch_dtype", "bf16")

    if args.json:
        compute_ms, memory_ms, bound = label.roofline_ms(args.gpu, dtype, args.num_gpus)
        json.dump(
            {
                "model": model.name,
                "params": label.params,
                "flops": label.flops,
                "bytes": label.bytes,
                "roofline_ms": {
                    "global_floor": max(compute_ms, memory_ms),
                    "compute": compute_ms,
                    "memory": memory_ms,
                    "bound": bound,
                    "segmented_lower_bound": label.segmented_lower_bound_ms(
                        args.gpu, dtype, args.num_gpus
                    ),
                },
                "segments": label.segment_rows(args.gpu, dtype, args.num_gpus),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return

    _print_human(raw_config, model, workload, label, args.gpu, dtype, args.num_gpus)


if __name__ == "__main__":
    main()
