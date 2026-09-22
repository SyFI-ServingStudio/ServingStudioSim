"""Packing captured routes into the corpus the simulator samples."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alignment.profiler.token_corpus import fnv1a64, pack_token_corpus


def write_request(directory, name, ids, *, layers=None):
    tokens, num_layers, top_k = ids.shape
    np.savez(
        directory / f"{name}.npz",
        expert_ids=ids.astype(np.uint16),
        token_ids=np.arange(tokens, dtype=np.int32),
        token_positions=np.arange(tokens, dtype=np.int32),
        model_layer_indices=np.array(
            layers if layers is not None else range(3, 3 + num_layers), dtype=np.int32
        ),
    )


def routes(tokens, num_layers=4, top_k=8, num_experts=64, seed=0):
    rng = np.random.default_rng(seed)
    selected = np.empty((tokens, num_layers, top_k), dtype=np.uint16)
    for token in range(tokens):
        for layer in range(num_layers):
            selected[token, layer] = rng.choice(num_experts, size=top_k, replace=False)
    return selected


def test_pack_concatenates_in_request_order_and_checksums_the_payload(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    first, second = routes(5, seed=1), routes(3, seed=2)
    # Written out of order on purpose: concatenation order decides where the
    # seams between requests fall, so it must come from the id, not the listing.
    write_request(source, "request_10", second)
    write_request(source, "request_2", first)

    manifest = pack_token_corpus(source, tmp_path / "corpus", num_experts=64)

    assert manifest["num_tokens"] == 8
    assert (manifest["num_layers"], manifest["top_k"]) == (4, 8)
    payload = (tmp_path / "corpus" / "routes.u16").read_bytes()
    assert payload == np.concatenate([first, second]).astype("<u2").tobytes()
    assert manifest["checksum_fnv1a64"] == fnv1a64(payload)
    assert json.loads((tmp_path / "corpus" / "manifest.json").read_text()) == manifest

    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert [segment["request"] for segment in provenance["request_segments"]] == [
        "request_2",
        "request_10",
    ]
    assert provenance["request_segments"][1]["start"] == 5
    # Request identity stays out of the manifest: the sampler binds the
    # manifest, and a draw must not be able to condition on which request a
    # window came from.
    assert "request_segments" not in manifest


def test_a_single_token_generation_contributes_an_empty_segment(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    write_request(source, "request_0", routes(4))
    # The last generated token never runs a forward, so a one-token generation
    # yields no accepted routes at all. That is a real capture.
    write_request(source, "request_1", routes(0))

    manifest = pack_token_corpus(source, tmp_path / "corpus", num_experts=64)

    assert manifest["num_tokens"] == 4
    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert provenance["request_segments"][1]["length"] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda ids: ids[:, :3, :],
            "one corpus describes one model",
            id="layer_count_changed",
        ),
        pytest.param(lambda ids: np.full_like(ids, 99), "outside 0..63", id="expert_out_of_range"),
        pytest.param(
            lambda ids: np.concatenate([ids[..., :1], ids[..., :1], ids[..., 2:]], axis=-1),
            "selected the same expert twice",
            id="repeated_expert",
        ),
    ],
)
def test_a_capture_that_disagrees_with_itself_is_refused(tmp_path, mutate, message):
    source = tmp_path / "capture"
    source.mkdir()
    write_request(source, "request_0", routes(4))
    write_request(source, "request_1", mutate(routes(4, seed=3)))

    with pytest.raises(ValueError, match=message):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)


def test_a_capture_that_recorded_different_model_layers_is_refused(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    write_request(source, "request_0", routes(4), layers=range(3, 7))
    write_request(source, "request_1", routes(4, seed=3), layers=range(4, 8))

    with pytest.raises(ValueError, match="covers model layers"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)


def test_an_empty_capture_is_refused_rather_than_packed(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()

    with pytest.raises(ValueError, match="no captured request routes"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)
