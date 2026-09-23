"""Packing captured routes into the corpus the simulator samples."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alignment.profiler.token_corpus import fnv1a64, pack_token_corpus

DENSE_LAYERS = 3


def routes(tokens, num_layers=4, top_k=8, num_experts=64, seed=0):
    """What the server returns: the prompt row, then the routed layers in place.

    The array covers every model layer, so the leading dense ones stay zero and
    the packer has to find the routed span itself.
    """
    rng = np.random.default_rng(seed)
    selected = np.zeros((tokens + 1, DENSE_LAYERS + num_layers, top_k), dtype=np.uint16)
    for token in range(tokens + 1):
        for layer in range(DENSE_LAYERS, DENSE_LAYERS + num_layers):
            selected[token, layer] = rng.choice(num_experts, size=top_k, replace=False)
    return selected


def write_request(directory, name, ids):
    np.save(directory / f"{name}.npy", ids.astype(np.uint16))


def test_pack_drops_the_prompt_row_and_the_dense_layers(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(5)
    write_request(source, "session_s1_round_000000", captured)

    manifest = pack_token_corpus(source, tmp_path / "corpus", num_experts=64)

    assert manifest["num_tokens"] == 5
    assert (manifest["num_layers"], manifest["top_k"]) == (4, 8)
    payload = (tmp_path / "corpus" / "routes.u16").read_bytes()
    assert payload == captured[1:, DENSE_LAYERS:, :].astype("<u2").tobytes()
    assert manifest["checksum_fnv1a64"] == fnv1a64(payload)
    assert json.loads((tmp_path / "corpus" / "manifest.json").read_text()) == manifest

    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert provenance["model_layer_indices"] == [3, 4, 5, 6]
    assert provenance["num_model_layers"] == 7


def test_requests_concatenate_in_natural_id_order(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    first, second = routes(5, seed=1), routes(3, seed=2)
    # Written out of order, and named so that lexicographic order would put
    # session 10 first. Concatenation order decides where the seams between
    # requests fall, so it must come from the id read as a number.
    write_request(source, "session_s10_round_000000", second)
    write_request(source, "session_s2_round_000000", first)

    manifest = pack_token_corpus(source, tmp_path / "corpus", num_experts=64)

    assert manifest["num_tokens"] == 8
    payload = (tmp_path / "corpus" / "routes.u16").read_bytes()
    expected = np.concatenate([first[1:], second[1:]])[:, DENSE_LAYERS:, :]
    assert payload == expected.astype("<u2").tobytes()

    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert [segment["request"] for segment in provenance["request_segments"]] == [
        "session_s2_round_000000",
        "session_s10_round_000000",
    ]
    assert provenance["request_segments"][1]["start"] == 5
    # Request identity stays out of the manifest: the sampler binds the
    # manifest, and a draw must not be able to condition on which request a
    # window came from.
    assert "request_segments" not in manifest


def test_a_single_token_generation_contributes_an_empty_segment(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    write_request(source, "session_s1_round_000000", routes(4))
    # The last generated token never runs a forward, so a one-token generation
    # leaves only the prompt row. That is a real capture.
    write_request(source, "session_s2_round_000000", routes(0))

    manifest = pack_token_corpus(source, tmp_path / "corpus", num_experts=64)

    assert manifest["num_tokens"] == 4
    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert provenance["request_segments"][1]["length"] == 0


def test_a_layer_that_routes_nowhere_inside_the_span_is_refused(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(4)
    # A routed layer that recorded nothing is how a mis-bound capture hook
    # shows up. Packing it would silently shift every later layer's identity.
    captured[:, DENSE_LAYERS + 1, :] = 0
    write_request(source, "session_s1_round_000000", captured)

    with pytest.raises(ValueError, match="must be contiguous"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)


def test_a_step_that_skipped_drafting_drops_its_tokens_not_the_pack(tmp_path):
    """A sequence past the drafter's limit still decodes, with no MTP routes."""
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(5)
    captured[4:, -1, :] = 0  # the MTP slot of the last two generated tokens
    write_request(source, "session_s1_round_000000", captured)

    # The capture is one slot wider than the target: the MTP drafter's.
    target_layers = DENSE_LAYERS + 3
    manifest = pack_token_corpus(
        source, tmp_path / "corpus", num_experts=64, num_target_layers=target_layers
    )

    assert (manifest["num_tokens"], manifest["num_layers"]) == (3, 4)
    payload = (tmp_path / "corpus" / "routes.u16").read_bytes()
    assert payload == captured[1:4, DENSE_LAYERS:, :].astype("<u2").tobytes()
    provenance = json.loads((tmp_path / "corpus" / "provenance.json").read_text())
    assert provenance["tokens_dropped_without_draft_routes"] == 2
    assert provenance["request_segments"][0]["length"] == 3


def test_without_a_drafter_an_empty_last_layer_is_a_defect(tmp_path):
    """The same hole in an undrafted capture is a body layer that did not record."""
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(5)
    captured[4, -1, :] = 0
    write_request(source, "session_s1_round_000000", captured)

    with pytest.raises(ValueError, match="token 3 recorded no routes"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)
    # A speculative capture whose drafter was not MTP has no extra slot either.
    with pytest.raises(ValueError, match="token 3 recorded no routes"):
        pack_token_corpus(
            source, tmp_path / "corpus", num_experts=64, num_target_layers=DENSE_LAYERS + 4
        )


def test_a_drafted_capture_that_never_recorded_its_slot_is_refused(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(4)
    captured[:, -1, :] = 0
    write_request(source, "session_s1_round_000000", captured)

    with pytest.raises(ValueError, match="must record the MTP slot"):
        pack_token_corpus(
            source, tmp_path / "corpus", num_experts=64, num_target_layers=DENSE_LAYERS + 3
        )


def test_a_token_missing_a_body_layer_is_refused(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()
    captured = routes(4)
    captured[2, DENSE_LAYERS + 1, :] = 0
    write_request(source, "session_s1_round_000000", captured)

    with pytest.raises(ValueError, match="token 1 recorded no routes"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda ids: ids[:, :-1, :],
            "one corpus describes one model",
            id="layer_count_changed",
        ),
        pytest.param(
            lambda ids: np.where(ids > 0, np.uint16(99), ids),
            "outside 0..63",
            id="expert_out_of_range",
        ),
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
    write_request(source, "session_s1_round_000000", routes(4))
    write_request(source, "session_s2_round_000000", mutate(routes(4, seed=3)))

    with pytest.raises(ValueError, match=message):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)


def test_an_empty_capture_is_refused_rather_than_packed(tmp_path):
    source = tmp_path / "capture"
    source.mkdir()

    with pytest.raises(ValueError, match="no captured request routes"):
        pack_token_corpus(source, tmp_path / "corpus", num_experts=64)
