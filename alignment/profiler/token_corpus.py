"""Pack a capture's per-request expert routes into one token corpus.

The corpus is what the simulator's `routing: corpus` reads. It keeps whole
tokens rather than a per-expert marginal, because a grouped MoE GEMM is billed
partly by how many of a rank's expert groups are non-empty -- a statement about
which experts a step's tokens *jointly* select, which a marginal has already
summed away. Sampling contiguous runs of it reproduces the correlation a verify
block has; resampling a marginal cannot, by construction.

Input is one `.npy` per replayed request, written verbatim by the load
generator from what the instrumented server returned: shape
`(rows, model_layers, top_k)` over *every* model layer, whose first row is the
last prompt token's forward. This module owns turning that into the corpus:
dropping the prompt row, and reducing the layer axis to the layers that
actually route. Dense layers never call the capture hook, so they stay zero,
and a routed row always holds top-k *distinct* experts -- which makes all-zero
an unambiguous "not routed" rather than a threshold.

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
import re
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


def _request_sort_key(path: Path) -> tuple:
    """Natural order over the request id, so `s10` sorts after `s2`.

    Concatenation order decides where the seams between requests fall, and a
    sampled window may cross one. Making it deterministic is what lets two packs
    of the same capture produce the same bytes.
    """
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part) for part in re.split(r"(\d+)", path.stem)
    )


def _routed_layers(arrays: list[np.ndarray], num_layers: int, top_k: int) -> range:
    """The contiguous span of layers that recorded routes.

    A dense layer never calls the capture hook, so its rows stay zero for the
    whole capture, and a routed row holds top-k *distinct* experts -- which is
    what makes all-zero unambiguous, and why it requires top-k above one. The
    span is required to be contiguous because the simulator addresses the corpus
    by layer *range*: a gap would make "the first N layers" mean something other
    than what the model's first N routed layers are.
    """
    if top_k < 2:
        # With one slot a token routed to expert 0 is indistinguishable from a
        # layer that never routed, so this rule cannot separate them. The
        # capture is fine; this inference is what does not hold.
        raise ValueError(
            "a top-1 router cannot be told apart from a dense layer by an "
            "all-zero row; the routed layer span must be supplied instead"
        )
    routed = np.zeros(num_layers, dtype=bool)
    for ids in arrays:
        if ids.shape[0]:
            routed |= ids.any(axis=(0, 2))
    covered = np.flatnonzero(routed)
    if not covered.size:
        raise ValueError("the capture recorded no routed layers at all")
    span = range(int(covered[0]), int(covered[-1]) + 1)
    if covered.size != len(span):
        missing = sorted(set(span) - set(covered.tolist()))
        raise ValueError(
            f"model layers {missing[:3]} route nowhere but sit inside the routed "
            f"span {span.start}..{span.stop}; the corpus layer axis must be contiguous"
        )
    return span


def _drop_undrafted_tokens(path: Path, ids: np.ndarray, drafted: bool) -> tuple[np.ndarray, int]:
    """Remove the tokens whose step ran no drafter, and count them.

    The engine skips drafting for a step whose sequences no longer fit the
    drafter, so the MTP slot -- the span's last layer in a drafted capture --
    stays zero for those tokens while every body layer routed. The token is
    real but only partly recorded, and a corpus row must be whole, so it is left
    out. Any other unrouted layer is a capture defect and is refused; without a
    drafter the last layer is a body layer like the rest.
    """
    routed = ids.any(axis=2)
    if routed.all():
        return ids, 0
    undrafted = np.zeros(routed.shape[0], dtype=bool)
    if drafted and routed.shape[1] > 1:
        undrafted = routed[:, :-1].all(axis=1) & ~routed[:, -1]
    broken = ~routed.all(axis=1) & ~undrafted
    if broken.any():
        token = int(np.flatnonzero(broken)[0])
        raise ValueError(
            f"{path.name}: token {token} recorded no routes for a layer inside the routed "
            "span; only a skipped draft step may leave one empty, and only the MTP slot"
        )
    return ids[~undrafted], int(undrafted.sum())


def pack_token_corpus(
    source: Path,
    out_dir: Path,
    *,
    num_experts: int,
    num_target_layers: int | None = None,
    scope: str = "accepted generated tokens; no prompt or rejected draft routes",
) -> dict:
    """Concatenate every captured request into one corpus under `out_dir`.

    `num_target_layers` is the checkpoint's `num_hidden_layers`. The capture
    buffer appends a slot past them only for an MTP drafter, so a capture wider
    than the target drafted, and only that last slot may be empty for a token.
    Without it every routed layer must be whole.

    Returns the manifest. Raises when the capture is empty or internally
    inconsistent; nothing is written in that case.
    """
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if num_experts > 65536:
        raise ValueError("a token corpus stores expert ids as u16")

    paths = sorted(Path(source).glob("*.npy"), key=_request_sort_key)
    if not paths:
        raise ValueError(f"no captured request routes under {source}")

    arrays: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for path in paths:
        ids = np.load(path, allow_pickle=False)
        if ids.ndim != 3:
            raise ValueError(f"{path.name}: expected a (tokens, layers, top_k) array")
        if shape is None:
            shape = ids.shape[1:]
        elif ids.shape[1:] != shape:
            raise ValueError(
                f"{path.name}: routes are {ids.shape[1:]}, the capture's first request "
                f"was {shape}; one corpus describes one model"
            )
        # Row 0 is the last prompt token's forward -- the capture is anchored
        # there so that every row after it is provably a forward some generated
        # token caused. The corpus keeps only those.
        arrays.append(ids[1:])

    num_model_layers, top_k = shape
    drafted = False
    if num_target_layers is not None:
        if num_model_layers < num_target_layers:
            raise ValueError(
                f"routes cover {num_model_layers} layers but the checkpoint has "
                f"{num_target_layers}; the capture was not of this model"
            )
        drafted = num_model_layers > num_target_layers
    layers = _routed_layers(arrays, num_model_layers, top_k)
    if drafted and layers.stop != num_model_layers:
        # The capture buffer appends the drafter's slot after the target's
        # layers, so a drafted capture whose span stops short never recorded it.
        raise ValueError(
            f"a drafted capture must record the MTP slot, model layer {num_model_layers - 1}, "
            f"but routes stop at layer {layers.stop - 1}"
        )

    segments: list[dict] = []
    payload_parts: list[np.ndarray] = []
    offset = 0
    undrafted_tokens = 0
    for path, ids in zip(paths, arrays, strict=True):
        ids = ids[:, layers.start : layers.stop, :]
        ids, undrafted = _drop_undrafted_tokens(path, ids, drafted)
        undrafted_tokens += undrafted
        # A request whose generation was one token long contributes no
        # *accepted* routes: the last generated token never runs a forward.
        # That is a real capture, not a broken one, so it is kept as an empty
        # segment rather than rejected.
        if ids.shape[0]:
            if ids.min() < 0 or ids.max() >= num_experts:
                raise ValueError(f"{path.name}: expert id outside 0..{num_experts - 1}")
            # The router selects top-k *distinct* experts. A repeat means the
            # capture recorded padding or an uninitialized row, which would show
            # up downstream only as an implausibly concentrated fold.
            if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
                raise ValueError(f"{path.name}: a token selected the same expert twice")
        payload_parts.append(ids.astype("<u2"))
        segments.append({"request": path.stem, "start": offset, "length": int(ids.shape[0])})
        offset += int(ids.shape[0])

    payload = np.concatenate(payload_parts).tobytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DATA_FILE).write_bytes(payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_file": DATA_FILE,
        "num_tokens": offset,
        "num_layers": len(layers),
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
                "num_model_layers": int(num_model_layers),
                "model_layer_indices": list(layers),
                "tokens_dropped_without_draft_routes": undrafted_tokens,
                "request_segments": segments,
            },
            indent=2,
        )
        + "\n"
    )
    return manifest
