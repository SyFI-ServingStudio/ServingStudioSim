"""Write the verified allocation-only patch to a new file; never install it."""

import argparse
import hashlib
import json
from pathlib import Path

STOCK_SHA256 = "aa4f45c977bb6cc3e3a3e838c1b06fed643eb36ebfda4e04942df0a30a3b2562"
PATCHED_SHA256 = "61af4bdaa2bd8f99b80e07c684837a6b784a247a5d64d0e80a9db6e6bf4e6c41"
OLD = b"""        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
"""
NEW = b"""        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            # Allocation capacity must cover both prefill tokens and padded decode rows.
            # Keep prefill/segmentation/scheduler budgets unchanged. This trial is
            # restricted to ordinary one-token decode, without speculative decoding.
            max_num_batched_tokens=max(
                self.max_num_batched_tokens,
                max(self.neuron_config.num_seqs_buckets),
            ),
"""


def replay(source: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refuse to overwrite output: {output}")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != STOCK_SHA256:
        raise ValueError("source differs from the exact validated stock Neuron runner")
    if data.count(OLD) != 1:
        raise ValueError("source allocation site is missing or ambiguous")
    patched = data.replace(OLD, NEW)
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise ValueError("patched bytes differ from the validated allocation-only candidate")
    with output.open("xb") as stream:
        stream.write(patched)
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "source_sha256": STOCK_SHA256,
        "output_sha256": PATCHED_SHA256,
        "installed": False,
        "scope": "InputBatch allocation only; ordinary nonspeculative one-token decode",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original pinned neuron_model_runner.py")
    parser.add_argument("output", type=Path, help="new candidate file; must not already exist")
    args = parser.parse_args(argv)
    try:
        receipt = replay(args.source, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
