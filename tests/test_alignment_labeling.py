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
    format_coverage,
    load_rules,
    read_coverage,
    slots_ending,
    subsumptions,
    transfer_labels,
    walk_kernels,
)
from alignment.labeling import cli as labeling_cli
from alignment.labeling.rules import NO_NEIGHBOUR, label_body
from launcher.alignment_config import load_labeled_kernel_sequences


def test_unmapped_collective_rule_preserves_missing_simulator_owner():
    from launcher.alignment_config import _validate_labeled_kernel

    rule = Rule.from_mapping({
        "name": "allreduce", "operation": "draft.embedding",
        "status": "unmapped", "cross_rank": "synchronizing",
        "phase": "draft", "after_name": "embedding", "before_name": "norm",
    })
    label = label_body(rule, [])
    assert label == {
        "status": "unmapped", "cross_rank": "synchronizing",
        "collective": "draft.embedding",
    }
    operations, slots = {}, {}
    kernel = {"name": "allreduce", "suggested_category": "all_reduce", "label": label}
    _validate_labeled_kernel(kernel, "test", operations, slots)
    assert operations == slots == {}
    label["cross_rank"] = "independent"
    with pytest.raises(ValueError, match="collective identity requires"):
        _validate_labeled_kernel(kernel, "test", {}, {})


@pytest.mark.parametrize("extra", [
    {"slot_suffixes": ["embedding"]}, {"cross_rank": "independent"},
])
def test_unmapped_collective_rule_rejects_invented_mapping(extra):
    with pytest.raises(ValueError, match="unmapped collective rules require"):
        Rule.from_mapping({
            "name": "allreduce", "operation": "draft.embedding",
            "status": "unmapped", "cross_rank": "synchronizing", **extra,
        })


def _tracks(*programs: list[dict]) -> list[dict]:
    """Wrap folded programs as a sequence's concurrent tracks.

    Every fixture here is one execution stream unless it says otherwise, so the
    common case is one primary track holding what used to be `program`.
    """
    return [
        {
            "track_index": index,
            "stream_role": "primary" if index == 0 else "concurrent",
            "kernel_count": sum(
                len(segment["kernels"])
                if "kernels" in segment
                else segment["repeat"]["count"] * len(segment["repeat"]["body"]["kernels"])
                for segment in program
            ),
            "program": program,
        }
        for index, program in enumerate(programs)
    ]


def test_initialize_writes_explicit_unmapped_cross_rank_labels(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "labeled.json"
    source.write_text(
        json.dumps(
            {
                "phases": {
                    "forward": {
                        "unique_sequences": [
                            {
                                "sequence_id": "sequence_test",
                                "tracks": _tracks(
                                    [
                                        {
                                            "kernels": [
                                                {
                                                    "name": "kernel",
                                                    "suggested_category": "other",
                                                }
                                            ]
                                        }
                                    ]
                                ),
                            }
                        ]
                    }
                }
            }
        )
    )

    assert labeling_cli.main(["initialize", str(source), str(output)]) == 0

    initialized = json.loads(output.read_text())
    assert next(walk_kernels(initialized)).label == {
        "status": "unmapped",
        "cross_rank": "independent",
    }
    assert check(initialized) == []


def test_initialize_refuses_to_overwrite_existing_label_decisions(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "labeled.json"
    source.write_text(
        json.dumps(
            {
                "phases": {
                    "forward": {
                        "unique_sequences": [
                            {
                                "sequence_id": "sequence_test",
                                "tracks": _tracks(
                                    [
                                        {
                                            "kernels": [
                                                {
                                                    "name": "kernel",
                                                    "suggested_category": "other",
                                                    "label": {
                                                        "status": "unmapped",
                                                        "cross_rank": "independent",
                                                    },
                                                }
                                            ]
                                        }
                                    ]
                                ),
                            }
                        ]
                    }
                }
            }
        )
    )

    with pytest.raises(SystemExit):
        labeling_cli.main(["initialize", str(source), str(output)])

    assert not output.exists()


def test_initialize_unfolds_repeats_for_occurrence_specific_boundaries(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "labeled.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "encoding": "folded-v2",
                "source_parsed": "profile/parsed.json",
                "device_ids": [0],
                "folding_policy": {"kind": "exact_contiguous_repeat"},
                "phases": {
                    "forward": {
                        "unique_sequences": [
                            {
                                "sequence_id": "sequence_test",
                                "occurrences": [{"device_id": 0, "iterations": [7]}],
                                "expanded_kernel_count": 4,
                                "tracks": _tracks(
                                    [
                                        {
                                            "repeat": {
                                                "count": 2,
                                                "body": {
                                                    "kernels": [
                                                        {
                                                            "name": "norm",
                                                            "suggested_category": "other",
                                                        },
                                                        {
                                                            "name": "projection",
                                                            "suggested_category": "other",
                                                        },
                                                    ]
                                                },
                                            }
                                        }
                                    ]
                                ),
                            }
                        ]
                    }
                },
            }
        )
    )

    assert labeling_cli.main(["initialize", str(source), str(output), "--unfold"]) == 0

    unfolded = json.loads(output.read_text())
    assert unfolded["encoding"] == "literal-v1"
    positions = list(walk_kernels(unfolded))
    assert len(positions) == 4
    assert all(position.repeat == 1 for position in positions)
    assert positions[0].next_name == "projection"
    assert positions[2].next_name == "projection"
    assert load_labeled_kernel_sequences(output)["encoding"] == "literal-v1"


def test_rule_before_name_distinguishes_last_identical_boundary():
    rule = Rule.from_mapping(
        {
            "name": "norm",
            "before_name": "lm_head",
            "operation": "main.final_norm",
            "type": "model",
            "role": "final norm",
            "slot_suffixes": ["final_norm"],
        }
    )
    document = {
        "phases": {
            "forward": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_boundary",
                        "tracks": _tracks(
                            [
                                {
                                    "kernels": [
                                        kernel("norm"),
                                        kernel("qkv"),
                                        kernel("norm"),
                                        kernel("lm_head"),
                                    ]
                                }
                            ]
                        ),
                    }
                ]
            }
        }
    }

    positions = list(walk_kernels(document))
    assert not rule.matches(positions[0])
    assert rule.matches(positions[2])


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
                        "tracks": _tracks(
                            [
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
                            ]
                        ),
                    }
                ]
            },
            "postprocess": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_bbbb2222",
                        "tracks": _tracks([{"kernels": [kernel("nvjet_tile_TNT")]}]),
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


def test_walk_stops_predecessors_at_a_track_edge():
    """A concurrent track's first kernel has no predecessor at all.

    Two bodies at least ran one after the other, so "what ran before" is a
    weaker but real fact across them. Two tracks ran at the same time: there is
    no "before" between them, and a rule keyed on `after`/`after_name` that
    fired across the boundary would charge one stream's kernel to the other
    stream's neighbour.
    """
    document = {
        "phases": {
            "forward": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_two_tracks",
                        "tracks": _tracks(
                            [{"kernels": [kernel("routed_gemm"), kernel("routed_down")]}],
                            [{"kernels": [kernel("shared_gemm")]}],
                        ),
                    }
                ]
            }
        }
    }

    positions = list(walk_kernels(document))
    positions[0].label.update(MAPPED_NORM)
    positions = list(walk_kernels(document))

    assert [position.track_index for position in positions] == [0, 0, 1]
    # Inside one track the chain still works.
    assert positions[1].previous_name == "routed_gemm"
    assert positions[1].previous_operation == MAPPED_NORM["operation"]
    # Across the track boundary nothing carries over, in either direction.
    assert positions[2].previous_name is None
    assert positions[2].previous_operation is None
    assert positions[1].next_name is None
    assert "track1" in positions[2].coordinate


def test_slots_ending_collects_every_layer_variant_and_rejects_a_stale_suffix():
    assert slots_ending(SLOTS, "router.router_gemm_bf16_proxy") == [
        "unified.body.dense.moe.router.router_gemm_bf16_proxy",
        "unified.body.layer.moe.router.router_gemm_bf16_proxy",
    ]
    with pytest.raises(ValueError, match="no simulated slot"):
        slots_ending(SLOTS, "attention.renamed_away")


def test_target_rule_does_not_absorb_matching_mtp_slots():
    rule = Rule.from_mapping({
        "name": "gemm", "operation": "target.head", "type": "model", "role": "target head",
        "slot_suffixes": ["lm_head"], "excluded_slot_prefixes": ["unified.mtp."],
    })
    slots = [("unified.lm_head", "gemm"), ("unified.mtp.step_0.lm_head", "gemm")]
    assert label_body(rule, slots)["simulated_slots"] == ["unified.lm_head"]


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


def test_a_stream_role_rule_separates_reused_kernels_on_concurrent_tracks():
    document = {
        "phases": {
            "forward": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_two_roles",
                        "tracks": _tracks(
                            [{"kernels": [kernel("reused_tile")]}],
                            [{"kernels": [kernel("reused_tile")]}],
                        ),
                    }
                ]
            }
        }
    }
    rules = [
        Rule(
            name="reused_tile",
            operation="moe.shared_expert.gate_up_proj",
            type="model",
            role="concurrent projection",
            slot_suffixes=("router.router_gemm_bf16_proxy",),
            stream_role="concurrent",
        )
    ]

    report = apply_rules(document, rules, SLOTS)
    positions = list(walk_kernels(document))
    assert report.applied == {"moe.shared_expert.gate_up_proj": 1}
    assert positions[0].operation is None
    assert positions[1].operation == "moe.shared_expert.gate_up_proj"
    assert check(document) == []


def test_a_before_rule_uses_a_mapped_successor_from_an_earlier_pass(document):
    reduction = Rule(
        name="splitKreduce_kernel",
        operation="moe.router.router_gemm",
        type="model",
        role="router reduction",
        slot_suffixes=("router.router_gemm_bf16_proxy",),
    )
    producer = Rule(
        name="nvjet_tile_TNT",
        operation="moe.router.router_gemm",
        type="model",
        role="router producer",
        slot_suffixes=("router.router_gemm_bf16_proxy",),
        before="moe.router.router_gemm",
    )

    assert apply_rules(document, [reduction], SLOTS).applied == {"moe.router.router_gemm": 1}
    assert apply_rules(document, [producer], SLOTS).applied == {"moe.router.router_gemm": 1}
    positions = list(walk_kernels(document))
    assert positions[1].operation == "moe.router.router_gemm"
    assert positions[1].next_operation == "moe.router.router_gemm"


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

    invalid_stream_role = {
        "name": "x",
        "operation": "y",
        "type": "model",
        "role": "test",
        "slot_suffixes": ["lm_head"],
        "stream_role": "side",
    }
    path.write_text(json.dumps([invalid_stream_role]))
    with pytest.raises(ValueError, match="stream_role must be one of"):
        load_rules(path)


def test_coverage_aggregates_the_per_row_unmapped_lists_by_name(tmp_path):
    """The analyzer reports one row per folded position, so a kernel that runs
    in 225 rows never shows its total anywhere."""
    report = tmp_path / "alignment_iteration_report.json"
    report.write_text(
        json.dumps(
            {
                "meta": {"iterations": 3},
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
                },
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
    assert coverage.iteration_count == 3
    assert coverage.slots[0] == ("a.kv_a_rms_norm", 232.0)

    rendered = format_coverage(coverage)
    assert "891.000 / 19719.000 ms total across 3 iterations" in rendered
    assert "per-iteration average 297.000 / 6573.000 ms" in rendered


def test_transfer_moves_labels_onto_a_re_parsed_inventory_of_the_same_program(document):
    """A re-parse rebuilds the same kernels with new occurrence bookkeeping."""
    destination = copy.deepcopy(document)
    for phase in destination["phases"].values():
        for sequence in phase["unique_sequences"]:
            sequence["occurrences"] = [{"device_id": 0, "iterations": [8]}]
            for track in sequence["tracks"]:
                for segment in track["program"]:
                    body = segment.get("kernels") or segment["repeat"]["body"]["kernels"]
                    for kernel_entry in body:
                        kernel_entry.pop("label", None)

    report = transfer_labels(document, destination)

    assert report.transferred == 1
    assert report.unlabeled == 4
    forward = destination["phases"]["forward"]["unique_sequences"][0]
    assert forward["tracks"][0]["program"][0]["kernels"][0]["label"] == MAPPED_NORM
    assert forward["occurrences"] == [{"device_id": 0, "iterations": [8]}]


def test_transfer_refuses_an_inventory_whose_program_differs(document):
    destination = copy.deepcopy(document)
    destination["phases"]["forward"]["unique_sequences"][0]["tracks"][0]["program"][0]["kernels"][
        0
    ]["name"] = "some_other_kernel"

    with pytest.raises(ValueError, match="not the same program"):
        transfer_labels(document, destination)


def _rule(operation: str, **keys) -> Rule:
    return Rule.from_mapping(
        {
            "operation": operation,
            "type": "ffn",
            "role": "gemm",
            "slot_suffixes": ["gate_up_proj"],
            **keys,
        }
    )


def test_subsumptions_proves_the_wider_matcher_decides_nothing_on_its_own():
    """The real defect this was written for: a later file's wider matcher was
    survived rather than corrected, so the label came from the file order.

    `nvjet_sm100_tst_` is a family prefix — every position the narrow rule can
    ever see, the wide one sees too, and they disagree about the operation.
    """
    wide = _rule("ffn.gate_up_proj", name="nvjet_sm100_tst_", phase="forward")
    narrow = _rule(
        "moe.shared_expert.gate_up_proj",
        name="nvjet_sm100_tst_64x32_64x16",
        phase="forward",
    )

    (finding,) = subsumptions([wide, narrow])
    assert "ffn.gate_up_proj" in finding and "moe.shared_expert.gate_up_proj" in finding
    # Order-independent itself: swapping the two reports the same pair.
    assert subsumptions([narrow, wide]) == [finding]
    # Same matcher, same answer, no ambiguity to report.
    assert subsumptions([wide, _rule("ffn.gate_up_proj", name="nvjet", phase="forward")]) == []


def test_subsumptions_is_silent_when_a_key_can_separate_the_two_rules():
    """An omitted key is no constraint, an exact key must agree, and the empty
    slot is structural — it subsumes only itself. Each is enough to separate."""
    base = {"name": "cublasLt::splitKreduce_kernel", "phase": "forward"}
    wide = _rule("attention.q_b_proj", **base)

    # Distinguished by an exact key both constrain, to different values. Both
    # must constrain it: omitting a key widens a rule, so `wide` without a
    # `stream_role` still claims everything a rule that pins one claims.
    assert subsumptions(
        [
            _rule("attention.q_b_proj", **base, stream_role="primary"),
            _rule("moe.router", **base, stream_role="concurrent"),
        ]
    ) == []
    assert len(subsumptions([wide, _rule("moe.router", **base, stream_role="concurrent")])) == 1
    # Distinguished by the neighbour sentinel: "no predecessor" is not "any".
    assert subsumptions(
        [
            _rule("attention.q_b_proj", **base, after=NO_NEIGHBOUR),
            _rule("moe.router", **base, after="moe.router"),
        ]
    ) == []
    # Narrow on different axes: each constrains a key the other leaves open, so
    # neither claims everything the other does and the file order settles
    # nothing. They may still overlap on some position — that is `disagreements`
    # to find against a capture, not something the rule text can decide.
    assert subsumptions(
        [
            _rule("attention.q_b_proj", **base, after="attention.q_b_proj"),
            _rule("moe.router", **base, stream_role="concurrent"),
        ]
    ) == []
