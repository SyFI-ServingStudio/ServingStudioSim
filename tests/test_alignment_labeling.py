"""Contract tests for the labeling tools.

The fixture is a miniature of the structure the real inventory has and the
tests are the defects that actually reached a published number: a tile reused by
two layers, a chain that can only be labeled if the first link is labeled first,
and an operation whose kernels ended up with two different label bodies.
"""

import copy
import json

import pytest

from alignment.labeling import (
    Rule,
    apply_rules,
    check,
    load_rules,
    read_coverage,
    slots_ending,
    transfer_labels,
    walk_kernels,
)


def kernel(name, label=None):
    return {"name": name, "label": {} if label is None else label}


MAPPED_NORM = {
    "status": "mapped",
    "operation": "attention.q_a_rms_norm",
    "type": "attention",
    "role": "q_a_rms_norm",
    "simulated_slots": ["unified.body.layer.attention.q_a_rms_norm"],
    "cross_rank": "independent",
}


@pytest.fixture
def document():
    """One flat segment and one repeat body, both in the forward phase, plus a
    postprocess sequence that reuses the decoder's GEMM tile for the lm_head."""
    return {
        "phases": {
            "forward": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_aaaa1111",
                        "program": [
                            {"kernels": [kernel("norm_kernel", dict(MAPPED_NORM))]},
                            {
                                "repeat": {
                                    "count": 18,
                                    "body": {
                                        "kernels": [
                                            kernel("nvjet_tile_TNT"),
                                            kernel("splitKreduce_kernel"),
                                            kernel("cudaMemsetAsync_filler"),
                                        ]
                                    },
                                }
                            },
                        ],
                    }
                ]
            },
            "postprocess": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_bbbb2222",
                        "program": [{"kernels": [kernel("nvjet_tile_TNT")]}],
                    }
                ]
            },
        }
    }


SLOTS = [
    ("unified.body.layer.attention.q_a_rms_norm", "rms_norm"),
    ("unified.body.layer.moe.router.router_gemm_bf16_proxy", "single_gemm"),
    ("unified.body.dense.moe.router.router_gemm_bf16_proxy", "single_gemm"),
    ("unified.postprocess.lm_head", "single_gemm"),
]


def test_walk_yields_program_order_and_stops_predecessors_at_a_body_edge(document):
    positions = list(walk_kernels(document))
    assert [position.name for position in positions] == [
        "norm_kernel",
        "nvjet_tile_TNT",
        "splitKreduce_kernel",
        "cudaMemsetAsync_filler",
        "nvjet_tile_TNT",
    ]
    # The repeat body is a new body, so the flat segment's mapped kernel is not
    # its predecessor: a rule keyed on `after` must not fire across the edge.
    assert positions[1].previous_operation is None
    assert positions[1].previous_name is None
    assert positions[1].repeat == 18
    # The postprocess sequence likewise starts clean.
    assert positions[4].previous_operation is None


def test_slots_ending_collects_every_layer_variant_and_rejects_a_stale_suffix():
    assert slots_ending(SLOTS, "router.router_gemm_bf16_proxy") == [
        "unified.body.dense.moe.router.router_gemm_bf16_proxy",
        "unified.body.layer.moe.router.router_gemm_bf16_proxy",
    ]
    with pytest.raises(ValueError, match="no simulated slot"):
        slots_ending(SLOTS, "attention.renamed_away")


def test_a_rule_chain_labels_the_reduce_that_follows_the_gemm_it_reduces(document):
    rules = [
        Rule(
            name="nvjet_tile_TNT",
            operation="moe.router.router_gemm",
            type="moe",
            role="moe router gemm",
            slot_suffixes=("router.router_gemm_bf16_proxy",),
        ),
        # Only fires because the GEMM above was labeled in the same pass.
        Rule(
            name="splitKreduce_kernel",
            operation="moe.router.router_gemm",
            type="moe",
            role="moe router gemm",
            slot_suffixes=("router.router_gemm_bf16_proxy",),
            after="moe.router.router_gemm",
        ),
    ]
    report = apply_rules(document, rules, SLOTS)
    assert report.applied["moe.router.router_gemm"] == 3  # 2 forward + 1 postprocess
    assert report.unfired == []
    labels = [position.label for position in walk_kernels(document)]
    assert labels[2]["operation"] == "moe.router.router_gemm"
    assert labels[2]["simulated_slots"] == [
        "unified.body.dense.moe.router.router_gemm_bf16_proxy",
        "unified.body.layer.moe.router.router_gemm_bf16_proxy",
    ]
    # Bookkeeping stays unmapped: no rule claims it and none invents coverage.
    assert labels[3] == {}


def test_a_phase_rule_separates_the_lm_head_from_the_decoder_tile(document):
    """The defect that produced a fake `model under-predicts 2x` finding."""
    rules = [
        Rule(
            name="nvjet_tile_TNT",
            operation="main.lm_head",
            type="ffn",
            role="lm_head",
            slot_suffixes=("lm_head",),
            phase="postprocess",
        ),
        Rule(
            name="nvjet_tile_TNT",
            operation="attention.q_absorb",
            type="attention",
            role="q_absorb",
            slot_suffixes=("router.router_gemm_bf16_proxy",),
        ),
    ]
    apply_rules(document, rules, SLOTS)
    positions = list(walk_kernels(document))
    assert positions[1].operation == "attention.q_absorb"
    assert positions[4].operation == "main.lm_head"
    assert check(document) == []


def test_an_already_mapped_position_is_kept_and_reported_rather_than_overwritten(document):
    rules = [
        Rule(
            name="norm_kernel",
            operation="attention.something_else",
            type="attention",
            role="something else",
            slot_suffixes=("attention.q_a_rms_norm",),
        )
    ]
    report = apply_rules(document, rules, SLOTS)
    assert report.applied == {}
    assert len(report.conflicts) == 1
    assert "attention.q_a_rms_norm kept" in report.conflicts[0]
    assert list(walk_kernels(document))[0].operation == "attention.q_a_rms_norm"

    overwriting = [
        Rule(
            name="norm_kernel",
            operation="attention.something_else",
            type="attention",
            role="something else",
            slot_suffixes=("attention.q_a_rms_norm",),
            overwrite=True,
        )
    ]
    assert apply_rules(document, overwriting, SLOTS).applied == {"attention.something_else": 1}


def test_check_reports_one_operation_that_ended_up_with_two_label_bodies(document):
    positions = list(walk_kernels(document))
    for index, position in enumerate((positions[1], positions[4])):
        position.label.update(
            {
                "status": "mapped",
                "operation": "attention.q_absorb",
                "type": "attention",
                "role": "q_absorb" if index == 0 else "q absorb",
                "simulated_slots": ["unified.postprocess.lm_head"],
                "cross_rank": "independent",
            }
        )
    findings = check(document)
    errors = [finding for finding in findings if finding.severity == "error"]
    assert len(errors) == 1
    assert "attention.q_absorb" in errors[0].summary
    assert "2 different labels" in errors[0].summary


def test_check_reports_a_label_with_no_cross_rank(document):
    list(walk_kernels(document))[1].label.update(
        {
            "status": "mapped",
            "operation": "attention.q_absorb",
            "type": "attention",
            "role": "q_absorb",
            "simulated_slots": ["unified.postprocess.lm_head"],
        }
    )
    errors = [finding for finding in check(document) if finding.severity == "error"]
    assert len(errors) == 1
    assert "no cross_rank" in errors[0].summary


def test_check_flags_one_operation_spread_across_two_phases(document):
    """The lm_head defect's signature before anyone knows it is a defect."""
    body = {
        "status": "mapped",
        "operation": "attention.q_absorb",
        "type": "attention",
        "role": "q_absorb",
        "simulated_slots": ["unified.body.layer.moe.router.router_gemm_bf16_proxy"],
        "cross_rank": "independent",
    }
    for position in walk_kernels(document):
        if position.name == "nvjet_tile_TNT":
            position.label.update(body)
    findings = check(document)
    assert [finding.severity for finding in findings] == ["warning"]
    assert "one operation across 2 phases" in findings[0].summary


def test_rules_load_from_a_file_and_reject_an_unknown_key(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "nvjet_tile_TNT",
                        "operation": "main.lm_head",
                        "type": "ffn",
                        "role": "lm_head",
                        "slot_suffixes": ["lm_head"],
                        "phase": "postprocess",
                        "note": "same tile as the decoder's q_absorb",
                    }
                ]
            }
        )
    )
    (rule,) = load_rules(path)
    assert rule.phase == "postprocess"
    assert rule.cross_rank == "independent"

    path.write_text(json.dumps([{"name": "x", "operation": "y", "typo_key": 1}]))
    with pytest.raises(ValueError, match="unknown keys"):
        load_rules(path)


def test_coverage_aggregates_the_per_row_unmapped_lists_by_name(tmp_path):
    """The analyzer reports one row per folded position, so a kernel that runs
    in 225 rows never shows its total anywhere."""
    report = tmp_path / "alignment_iteration_report.json"
    report.write_text(
        json.dumps(
            {
                "mapping": {
                    "coverage": {
                        "measured_duration_fraction": 0.9548,
                        "simulated_workload_fraction": 0.992,
                        "measured_total_kernel_ms": 19719.0,
                        "measured_mapped_ms": 18828.0,
                    },
                    "unmapped_measured_kernels": [
                        {"name": "computeCountAndIndiceDevice", "total_ms": 200.0, "calls": 100},
                        {"name": "computeCountAndIndiceDevice", "total_ms": 103.1, "calls": 66},
                        {"name": "triton_poi_fused_2", "total_ms": 56.3, "calls": 12},
                    ],
                    "unmapped_simulated_slots": [
                        {"slot": "a.kv_a_rms_norm", "total_ms": 232.0},
                        {"slot": "b.router_fp32_cast", "total_ms": 222.3},
                    ],
                }
            }
        )
    )
    coverage = read_coverage(report)
    assert [(kernel.name, kernel.rows, kernel.calls) for kernel in coverage.kernels] == [
        ("computeCountAndIndiceDevice", 2, 166),
        ("triton_poi_fused_2", 1, 12),
    ]
    assert coverage.kernels[0].total_ms == pytest.approx(303.1)
    assert coverage.measured_unmapped_ms == pytest.approx(891.0)
    assert coverage.slots[0] == ("a.kv_a_rms_norm", 232.0)


def test_transfer_moves_labels_onto_a_re_parsed_inventory_of_the_same_program(document):
    """A re-parse rebuilds the same kernels with new occurrence bookkeeping."""
    destination = copy.deepcopy(document)
    for phase in destination["phases"].values():
        for sequence in phase["unique_sequences"]:
            sequence["occurrences"] = [{"device_id": 0, "iterations": [8]}]
            for segment in sequence["program"]:
                body = segment.get("kernels") or segment["repeat"]["body"]["kernels"]
                for kernel_entry in body:
                    kernel_entry.pop("label", None)

    report = transfer_labels(document, destination)

    assert report.transferred == 1
    assert report.unlabeled == 4
    forward = destination["phases"]["forward"]["unique_sequences"][0]
    assert forward["program"][0]["kernels"][0]["label"] == MAPPED_NORM
    assert forward["occurrences"] == [{"device_id": 0, "iterations": [8]}]


def test_transfer_refuses_an_inventory_whose_program_differs(document):
    destination = copy.deepcopy(document)
    destination["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0][
        "name"
    ] = "some_other_kernel"

    with pytest.raises(ValueError, match="not the same program"):
        transfer_labels(document, destination)
