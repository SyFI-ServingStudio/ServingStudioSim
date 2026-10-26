"""Independent necessary work maps once onto the stock full-forward boundary."""

import json
from collections import Counter
from pathlib import Path

import pytest

from model.work import Workload, load_model

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "model/work/location_maps/llama3_vllm_neuron_unified.json"


@pytest.mark.parametrize(
    "phase,batch,expected_flops,expected_bytes",
    [
        ("prefill", 1, 7_102_927_994_880, 15_080_038_400),
        ("decode", 1, 15_274_082_304, 15_076_048_896),
        ("decode", 9, 137_466_740_736, 15_605_645_312),
        ("decode", 16, 244_385_316_864, 16_069_042_176),
    ],
)
def test_stock_forward_map_reconciles_independent_logical_work(
    phase, batch, expected_flops, expected_bytes
):
    """Goldens use checkpoint math; no profiler, TP partition or bucket shapes.

    Body FLOPs: 13,958,643,712/token; attention: 524,288/pair;
    sampled head: 1,050,673,152/row. Weights/norms: 15,009,849,344 bytes
    plus 8192/embedding row. Persistent KV: 131,072/token. Prefill504
    has 504*505/2 causal pairs; decode505 includes the current token.
    """
    mapping = json.loads(MAP.read_text())
    assert mapping["schema_version"] == 1
    assert mapping["mapping_id"] == "llama3-vllm-neuron-unified-v1"
    assert mapping["arch_types"] == ["llama3_vllm_neuron"]
    assert [rule["location"] for rule in mapping["locations"]] == ["unified.forward"]
    workload = (
        Workload.causal_lm(prefill=[(504, 0)], sampled=1)
        if phase == "prefill"
        else Workload.causal_lm(decode=[505] * batch, sampled=batch)
    )
    label = load_model(ROOT / "model/config/llama3_8b.json").label(workload)
    rows = {segment.name: segment for segment in label.segments}
    semantics = [name for rule in mapping["locations"] for name in rule["semantics"]]
    assert len(rows) == len(label.segments) == 12
    assert Counter(semantics) == Counter({name: 1 for name in rows})
    assert sum(rows[name].flops_total for name in semantics) == expected_flops
    assert sum(rows[name].bytes_total for name in semantics) == expected_bytes
    assert label.flops_total == expected_flops
    assert label.bytes_total == expected_bytes
    assert label.params["total"] == 8_030_261_248


REGIONS_MAP = ROOT / "model/work/location_maps/llama3_vllm_neuron_regions.json"


def test_region_map_partitions_the_whole_forward_semantics():
    """The region split moves only the vocabulary projection to the head."""
    whole = json.loads(MAP.read_text())
    regions = json.loads(REGIONS_MAP.read_text())
    assert regions["schema_version"] == 1
    assert regions["mapping_id"] == "llama3-vllm-neuron-regions-v1"
    assert regions["arch_types"] == whole["arch_types"]
    by_location = {rule["location"]: rule["semantics"] for rule in regions["locations"]}
    assert set(by_location) == {"unified.model", "unified.head"}
    assert by_location["unified.head"] == ["lm_head"]
    assert Counter(name for names in by_location.values() for name in names) == Counter(
        whole["locations"][0]["semantics"]
    )


SEGMENTS_MAP = ROOT / "model/work/location_maps/llama3_vllm_neuron_segments.json"


def test_segment_map_partitions_the_whole_forward_semantics():
    """Collective-delimited blocks own their sublayer's semantics, once each."""
    whole = json.loads(MAP.read_text())
    segments = json.loads(SEGMENTS_MAP.read_text())
    assert segments["mapping_id"] == "llama3-vllm-neuron-segments-v1"
    assert segments["arch_types"] == whole["arch_types"]
    by_location = {rule["location"]: rule["semantics"] for rule in segments["locations"]}
    assert set(by_location) == {
        "unified.embedding",
        "unified.attention_block",
        "unified.mlp_block",
        "unified.head",
    }
    assert by_location["unified.mlp_block"] == ["post_norm", "gate_up", "down"]
    assert Counter(name for names in by_location.values() for name in names) == Counter(
        whole["locations"][0]["semantics"]
    )


@pytest.mark.parametrize(
    "prior,chunk,sampled,expected_flops,expected_bytes",
    [
        (2048, 2048, 0, 31_886_374_076_416, 15_563_497_472),
        (14336, 2040, 1, 44_901_190_336_512, 17_172_996_096),
    ],
)
def test_segmented_prefill_chunk_binds_to_the_same_forward_location(
    prior, chunk, sampled, expected_flops, expected_bytes
):
    """`prefill_segment_size` keeps the single `unified.forward` slot, so the
    whole-forward map also attributes one chunk of a segmented prompt.

    Hand-derived from the per-unit costs above: a chunk of q tokens after p
    cached ones has q*p + q*(q+1)/2 causal pairs and reads/writes p+q KV rows.
    Only the final chunk samples. C16384/S2048: chunk 1 is (p=2048, q=2048);
    the final chunk of a 16376-token prompt is (p=14336, q=2040).
    """
    mapping = json.loads(MAP.read_text())
    assert [rule["location"] for rule in mapping["locations"]] == ["unified.forward"]
    pairs = chunk * prior + chunk * (chunk + 1) // 2
    assert expected_flops == (13_958_643_712 * chunk + 524_288 * pairs + 1_050_673_152 * sampled)
    assert expected_bytes == 15_009_849_344 + 8192 * chunk + 131_072 * (prior + chunk)
    label = load_model(ROOT / "model/config/llama3_8b.json").label(
        Workload.causal_lm(prefill=[(chunk, prior)], sampled=sampled)
    )
    semantics = mapping["locations"][0]["semantics"]
    assert Counter(semantics) == Counter(segment.name for segment in label.segments)
    assert label.flops_total == expected_flops
    assert label.bytes_total == expected_bytes
