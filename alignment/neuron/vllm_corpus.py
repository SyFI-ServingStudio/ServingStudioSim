"""Materialize the eight frozen museum prompts twice through req-frontend's pool.

The independent executor seeds a circular TokenProvider at ordinal*9973.
Each subject occupies one 9973-token segment. Only its first 504 tokens are
consumed; the remainder is neutral filler and is excluded from corpus claims.
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


def prompts(tokenizer) -> list[list[int]]:
    class Encoder:
        def encode(self, text):
            return tokenizer.encode(text, add_special_tokens=True).ids

    cases = cases_for_plan(
        {"context": 512, "specs": [{"phase": "decode", "token_bucket": 16}]}, Encoder()
    )
    return cases[0]["prompts"]


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
    expected = prompts(tokenizer)
    with Path(config.frontend.path).open() as source:
        rows = list(csv.DictReader(source))
    if (
        len(rows) != 16
        or len({r["id"] for r in rows}) != 16
        or any(
            int(r["input_len"]) != 504 or int(r["output_len"]) != 8 or float(r["arrival_time"]) != 0
            for r in rows
        )
    ):
        raise ValueError("initial stock trace requires 16 simultaneous 504+8 requests")
    hashes = {}
    for ordinal, row in enumerate(rows):
        start = ordinal * STRIDE % len(pool)
        actual = [pool[(start + i) % len(pool)] for i in range(504)]
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
        "consumed_prompt_tokens": 16 * 504,
        "unconsumed_filler_is_corpus": False,
        "sources": {
            k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for k, p in paths.items()
        },
    }


def materialize(model_path: Path, output: Path) -> None:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    expected = prompts(tokenizer)
    filler = " item" * (STRIDE - 504)
    if len(tokenizer.encode(filler, add_special_tokens=False).ids) != STRIDE - 504:
        raise ValueError("neutral filler tokenization changed")
    lines = []
    for ids in expected[:8]:
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=False).ids != ids:
            raise ValueError("museum prompt text does not round-trip exact token IDs")
        lines.extend([text, filler])
    output.mkdir(parents=True, exist_ok=True)
    (output / "museum-pool.txt").write_text("\n".join(lines) + "\n")
    with (output / "requests.csv").open("w") as target:
        writer = csv.writer(target)
        writer.writerow(["id", "arrival_time", "input_len", "output_len"])
        for i in range(16):
            writer.writerow([f"museum-{i:02d}", 0, 504, 8])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.model, args.output)
