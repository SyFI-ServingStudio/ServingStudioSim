"""Materialize the eight frozen museum prompts twice through req-frontend's pool.

The independent executor seeds a circular TokenProvider at ordinal*9973.
Each subject occupies one 9973-token segment. Only its first C-8 tokens are
consumed (504 at C512); the remainder is neutral filler excluded from corpus claims.
Request ordinal i replays subject i % 8, exactly as the stock profiler's batch does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from alignment.neuron.vllm_records import token_hash
from profiling.runners.neuron.vllm_forward_engine import cases_for_plan

STRIDE = 9973
POOL_SIZE = 8 * STRIDE
OUTPUT_TOKENS = 8


def prompts(tokenizer, context: int = 512, count: int = 16) -> list[list[int]]:
    """The profiler's frozen batch prompts: ordinal i is subject i % 8, C-8 tokens."""

    class Encoder:
        def encode(self, text):
            return tokenizer.encode(text, add_special_tokens=True).ids

    if type(count) is not int or count < 1:
        raise ValueError("stock corpus requires a positive request count")
    # One profiler batch of max(count, 8) covers every subject; its first
    # `count` prompts are the replay. cases_for_plan caps a batch at KV residency.
    batch = max(count, 8)
    cases = cases_for_plan(
        {"context": context, "specs": [{"phase": "decode", "token_bucket": batch}]}, Encoder()
    )
    if len(cases[0]["prompts"]) != batch:
        raise ValueError(f"C{context} KV pool cannot hold {count} resident requests")
    return cases[0]["prompts"][:count]


def verify_corpus(config, model_path: Path) -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    pool = []
    for line in Path(config.text_file).read_text().splitlines():
        if line.strip():
            pool.extend(tokenizer.encode(line, add_special_tokens=False).ids)
    pool = pool[: config.token_pool_limit]
    if len(pool) != POOL_SIZE or config.token_pool_limit != POOL_SIZE:
        raise ValueError("stock corpus must have exactly eight 9973-token pool segments")
    count, prompt_len = config.max_items, config.max_model_len - OUTPUT_TOKENS
    expected = prompts(tokenizer, config.max_model_len, count)
    with Path(config.frontend.path).open() as source:
        rows = list(csv.DictReader(source))
    if (
        len(rows) != count
        or len({r["id"] for r in rows}) != count
        or any(
            int(r["input_len"]) != prompt_len
            or int(r["output_len"]) != OUTPUT_TOKENS
            or float(r["arrival_time"]) != 0
            for r in rows
        )
    ):
        raise ValueError(
            f"stock trace requires {count} simultaneous {prompt_len}+{OUTPUT_TOKENS} requests"
        )
    hashes = {}
    for ordinal, row in enumerate(rows):
        start = ordinal * STRIDE % len(pool)
        actual = [pool[(start + i) % len(pool)] for i in range(prompt_len)]
        if actual != expected[ordinal]:
            raise ValueError("req-frontend pool does not produce the frozen museum prompt")
        hashes["independent_" + row["id"]] = token_hash(actual)
    paths = {
        "text": Path(config.text_file),
        "trace": Path(config.frontend.path),
        "tokenizer": model_path / "tokenizer.json",
    }
    return {
        "pool_tokens": POOL_SIZE,
        "ordinal_stride": STRIDE,
        "prompt_hash_encoding": "little-endian uint32 token IDs",
        "prompt_hashes": hashes,
        "consumed_prompt_tokens": count * prompt_len,
        "unconsumed_filler_is_corpus": False,
        "sources": {
            k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in paths.items()
        },
    }


def materialize(model_path: Path, output: Path, context: int = 512, requests: int = 16) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    prompt_len = context - OUTPUT_TOKENS
    prompts(tokenizer, context, requests)  # Reject a request count the KV pool cannot hold.
    subjects = prompts(tokenizer, context, 8)
    filler = " item" * (STRIDE - prompt_len)
    if len(tokenizer.encode(filler, add_special_tokens=False).ids) != STRIDE - prompt_len:
        raise ValueError("neutral filler tokenization changed")
    lines = []
    for ids in subjects:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=False).ids != ids:
            raise ValueError("museum prompt text does not round-trip exact token IDs")
        lines.extend([text, filler])
    output.mkdir(parents=True, exist_ok=True)
    (output / "museum-pool.txt").write_text("\n".join(lines) + "\n")
    with (output / "requests.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "arrival_time", "input_len", "output_len"])
        for i in range(requests):
            writer.writerow([f"museum-{i:02d}", 0, prompt_len, OUTPUT_TOKENS])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--requests", type=int, default=16)
    args = parser.parse_args()
    materialize(args.model, args.output, args.context, args.requests)
