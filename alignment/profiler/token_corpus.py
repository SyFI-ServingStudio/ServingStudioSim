"""Pack a capture's per-request expert routes into one token corpus.

The corpus is what the simulator's `routing: corpus` reads. It keeps whole
tokens rather than a per-expert marginal, because a grouped MoE GEMM is billed
partly by how many of a rank's expert groups are non-empty -- a statement about
which experts a step's tokens *jointly* select, which a marginal has already
summed away. Sampling contiguous runs of it reproduces the correlation a verify
block has; resampling a marginal cannot, by construction.

Input is one `.npz` per replayed request, written by the capture pass: an
`expert_ids` array of shape `(tokens, layers, top_k)` plus the token ids,
absolute token positions, and the model layer indices those layers are.
Output is `routes.u16` (little-endian, C order `[token, layer, slot]`), the
`manifest.json` the Rust `TokenCorpusConfig` deserializes, and a
`provenance.json` that keeps request identity *outside* the sampler -- the
sampler must not be able to condition on which request a window came from.

Validation is deliberate and eager. A capture costs GPU hours; a corpus that
silently disagrees with the model it will price is far more expensive than a
pack that refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DATA_FILE = "routes.u16"

_FNV_OFFSET = 0xCBF2_9CE4_8422_2325
_FNV_PRIME = 0x0000_0100_0000_01B3
_MASK64 = (1 << 64) - 1


def fnv1a64(payload: bytes) -> int:
    """FNV-1a over the payload bytes, matching the simulator's loader.

    The manifest carries this so a corpus that changed under a built config
    fails loudly instead of quietly shifting every profiled shape.
    """
    digest = _FNV_OFFSET
    for byte in payload:
        digest = ((digest ^ byte) * _FNV_PRIME) & _MASK64
    return digest


def _request_sort_key(path: Path) -> tuple[int, str]:
    """Order by the numeric request suffix when there is one, else by name.

    Concatenation order decides where the seams between requests fall, and a
    sampled window may cross one. Making it deterministic is what lets two packs
    of the same capture produce the same bytes.
    """
    stem = path.stem
    _, _, suffix = stem.rpartition("_")
    return (int(suffix), stem) if suffix.isdigit() else (1 << 62, stem)


def pack_token_corpus(
    source: Path,
    out_dir: Path,
    *,
    num_experts: int,
    scope: str = "accepted generated tokens; no prompt or rejected draft routes",
) -> dict:
    """Concatenate every captured request into one corpus under `out_dir`.

    Returns the manifest. Raises when the capture is empty or internally
    inconsistent; nothing is written in that case.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if num_experts > 65536:
        raise ValueError("a token corpus stores expert ids as u16")

    paths = sorted(Path(source).glob("request_*.npz"), key=_request_sort_key)
    if not paths:
        raise ValueError(f"no captured request routes under {source}")

    arrays: list[np.ndarray] = []
    segments: list[dict] = []
    shape: tuple[int, int] | None = None
    layer_indices: list[int] | None = None
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            ids = data["expert_ids"]
            layers = data["model_layer_indices"].tolist()
            if ids.ndim != 3:
                raise ValueError(f"{path.name}: expected a (tokens, layers, top_k) array")
            if shape is None:
                shape, layer_indices = ids.shape[1:], layers
            elif ids.shape[1:] != shape:
                raise ValueError(
                    f"{path.name}: routes are {ids.shape[1:]}, the capture's first request "
                    f"was {shape}; one corpus describes one model"
                )
            elif layers != layer_indices:
                raise ValueError(
                    f"{path.name}: covers model layers {layers[:3]}..., "
                    f"the capture's first request covered {layer_indices[:3]}..."
                )
            # A request whose generation was one token long contributes no
            # *accepted* routes: the last generated token never runs a forward.
            # That is a real capture, not a broken one, so it is kept as an
            # empty segment rather than rejected.
            if ids.shape[0]:
                if ids.min() < 0 or ids.max() >= num_experts:
                    raise ValueError(f"{path.name}: expert id outside 0..{num_experts - 1}")
                # The router selects top-k *distinct* experts. A repeat means
                # the capture recorded padding or an uninitialized row, which
                # would show up downstream only as an implausibly concentrated
                # fold.
                if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
                    raise ValueError(f"{path.name}: a token selected the same expert twice")
            arrays.append(ids.astype("<u2"))
            segments.append({"request": path.stem, "start": offset, "length": int(ids.shape[0])})
            offset += int(ids.shape[0])

    num_layers, top_k = shape
    payload = np.concatenate(arrays).tobytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DATA_FILE).write_bytes(payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_file": DATA_FILE,
        "num_tokens": offset,
        "num_layers": int(num_layers),
        "num_experts": num_experts,
        "top_k": int(top_k),
        "checksum_fnv1a64": fnv1a64(payload),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Kept beside the corpus, not inside the manifest: the sampler binds the
    # manifest and must not be able to condition a draw on request identity.
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(Path(source).resolve()),
                "encoding": "little-endian u16, C order [token, layer, top_k]",
                "scope": scope,
                "model_layer_indices": layer_indices,
                "request_segments": segments,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
