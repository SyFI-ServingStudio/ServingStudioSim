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
