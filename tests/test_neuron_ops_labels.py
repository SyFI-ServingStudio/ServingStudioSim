"""CPU checks for conservative static Neuron label evidence."""

import copy
import hashlib
import io
import json
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from profiling.profilers import neuron_ops_labels as labels


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _identity():
    return {"source_hashes": dict(labels.SUPPORTED_SOURCE_HASHES)}


def _map(instructions, graph="stock", core=0, function="layer"):
    return {
        "schema_version": 1,
        "instruction_size": 64,
        "decoder_version": labels.SUPPORTED_COMPILER_VERSION,
        "neff_sha256": _sha(graph),
        "source_identity": _identity(),
        "instructions": [
            {
                "pcore_idx": core,
                "engine": "GpSimd",
                "compiler_pc": i,
                "function": function,
                "instruction_sha256": _sha(value),
                "parents": [{"id": i, "kernel_file": f"{graph}/{i}.py"}],
            }
            for i, value in enumerate(instructions)
        ],
    }


def _transfer(stock, debug, **kwargs):
    return labels.transfer_debug_labels(
        stock,
        debug,
        source_identity=_identity(),
        expected_stock_neff_sha256=_sha("stock"),
        expected_debug_neff_sha256=_sha("debug"),
        **kwargs,
    )


def test_equal_function_proves_repeated_instruction_correspondence():
    stock, debug = _map(["same", "same"]), _map(["same", "same"], "debug")
    before = copy.deepcopy(stock)
    out = _transfer(stock, debug)
    assert stock == before
    assert out["instructions"][1]["parents"] == stock["instructions"][1]["parents"]
    for pc, row in enumerate(out["instructions"]):
        proof = row["debug_match"]
        assert proof["method"] == "identical_function_stream"
        assert proof["stock_function_stream_sha256"] == proof["debug_function_stream_sha256"]
        assert row["matched_debug_parents"] == debug["instructions"][pc]["parents"]


def test_ambiguous_duplicate_instructions_remain_unmapped():
    out = _transfer(_map(["a"] * 3), _map(["a"] * 4, "debug"))
    assert out["debug_transfer"]["totals"]["unmatched"] == 3
    assert all("matched_debug_parents" not in r for r in out["instructions"])


def test_changed_schedule_never_transfers_changed_instruction():
    out = _transfer(_map(["a", "b", "c"]), _map(["c", "changed", "a"], "debug"))
    rows = out["instructions"]
    assert all("matched_debug_parents" not in row for row in rows)
    assert not out["debug_transfer"]["functions"][0]["whole_function_equal"]


def test_long_unique_homogeneous_context_is_explicitly_inferred():
    values = [str(i) for i in range(25)]
    stock, debug = _map(values), _map(["inserted"] + values, "debug")
    for row in debug["instructions"]:
        row["parents"][0]["kernel_name"] = "same_kernel"
    out = _transfer(stock, debug)
    proof = out["instructions"][12]["debug_match"]
    assert proof["label_confidence"] == "inferred_context"
    assert proof["context_radius"] >= 8
    assert proof["homogeneous_context_owner"] == "same_kernel"
    assert proof["debug_compiler_pc"] == 13
    assert "debug_match" not in out["instructions"][0]


def test_mixed_semantic_context_stays_unresolved():
    values = [str(i) for i in range(25)]
    stock, debug = _map(values), _map(["inserted"] + values, "debug")
    for i, row in enumerate(debug["instructions"]):
        row["parents"][0]["kernel_name"] = f"kernel_{i % 2}"
    out = _transfer(stock, debug)
    assert all("debug_match" not in row for row in out["instructions"])


def test_projection_lowered_at_transpose_retains_composite_owner():
    filename = "vllm_neuron/functional/attention/attention_decode.py"
    hashes = dict(labels.SOURCE_HASHES)
    assert labels.source_operation(f"{filename}:809", hashes) == "qkv_projection"
    assert labels.source_operation(f"{filename}:823", hashes) == "qkv_projection_and_layout"
    assert labels.source_operation(f"{filename}:829", hashes) == "qkv_projection_and_layout"
    assert labels.source_operation(f"{filename}:820", hashes) == "qkv_layout"
    assert labels.source_operation(f"{filename}:804", hashes) == "rmsnorm"
    hashes[filename] = _sha("other source")
    assert labels.source_operation(f"{filename}:823", hashes) is None


def test_duplicate_pc_across_physical_cores_does_not_collide():
    stock, debug = _map(["same"]), _map(["same"], "debug")
    stock["instructions"] += _map(["same"], core=1)["instructions"]
    debug["instructions"] += _map(["same"], "debug", core=1)["instructions"]
    debug["instructions"][1]["parents"][0]["kernel_file"] = "core1.py"
    out = _transfer(stock, debug)
    assert out["instructions"][1]["matched_debug_parents"][0]["kernel_file"] == "core1.py"
    assert out["instructions"][0]["matched_debug_parents"][0]["kernel_file"] != "core1.py"


def test_no_cross_function_match():
    out = _transfer(_map(["same"], function="a"), _map(["same"], "debug", function="b"))
    assert out["debug_transfer"]["totals"]["matched"] == 0


@pytest.mark.parametrize("which", ["stock", "debug"])
def test_wrong_neff_identity_rejected(which):
    stock, debug = _map(["a"]), _map(["a"], "debug")
    (stock if which == "stock" else debug)["neff_sha256"] = _sha("other")
    with pytest.raises(ValueError, match="NEFF SHA256 differs"):
        _transfer(stock, debug)


def test_wrong_source_identity_rejected():
    stock, debug = _map(["a"]), _map(["a"], "debug")
    key = next(iter(labels.SUPPORTED_SOURCE_HASHES))
    debug["source_identity"]["source_hashes"][key] = _sha("changed source")
    with pytest.raises(ValueError, match="semantic source identity"):
        _transfer(stock, debug)


def test_unknown_compiler_abi_rejected():
    stock, debug = _map(["a"]), _map(["a"], "debug")
    debug["decoder_version"] = "unknown"
    with pytest.raises(ValueError, match="compiler ABI"):
        _transfer(stock, debug)


def test_duplicate_same_core_instruction_identity_rejected():
    stock, debug = _map(["a"]), _map(["a"], "debug")
    stock["instructions"] *= 2
    with pytest.raises(ValueError, match="Duplicate static instruction identity"):
        _transfer(stock, debug)


def test_extractor_rejects_unknown_abi_before_loading_sdk(tmp_path, monkeypatch):
    neff = tmp_path / "x.neff"
    neff.write_bytes(b"ignored")
    monkeypatch.setattr(labels.importlib.metadata, "version", lambda _: "future")
    with pytest.raises(ValueError, match="compiler ABI"):
        labels.export_static_map(neff)


@pytest.mark.parametrize(
    "parents,expected",
    [
        ([{"instruction_type": "TRANSPOSE"}], "transpose"),
        ([{"instruction_type": "TRANSPOSE"}] * 2, "transpose"),
        ([{"instruction_type": "TRANSPOSE"}, {"instruction_type": "REGULAR"}], None),
        ([{"instruction_type": "TRANSPOSE"}, {"missing": True}], None),
        ([{"instruction_type": "TRANSPOSE"}, {}], None),
        ([], None),
    ],
)
def test_compiler_primitive_requires_all_parent_types(parents, expected):
    assert labels.compiler_operation_from_parents(parents) == expected


def test_debug_transfer_does_not_replace_stock_primitive_proof():
    stock, debug = _map(["a"]), _map(["a"], "debug")
    stock["instructions"][0]["compiler_operation"] = None
    stock["instructions"][0]["parents"][0]["instruction_type"] = "REGULAR"
    debug["instructions"][0]["compiler_operation"] = "transpose"
    debug["instructions"][0]["parents"][0]["instruction_type"] = "TRANSPOSE"
    row = _transfer(stock, debug)["instructions"][0]
    assert row["compiler_operation"] is None
    assert labels.compiler_operation_from_parents(row["parents"]) is None


def test_extractor_preserves_duplicate_and_missing_backend_parents(tmp_path, monkeypatch):
    class FakeProto:
        def ParseFromString(self, value):
            data = json.loads(value)
            self.instructions = [SimpleNamespace(**row) for row in data["instructions"]]
            for ins in self.instructions:
                ins.instruction_type = 3
                ins.parent_ids = getattr(ins, "parent_ids", [12, 13])
                ins.DESCRIPTOR = SimpleNamespace(
                    fields_by_name={
                        "instruction_type": SimpleNamespace(
                            enum_type=SimpleNamespace(
                                values_by_number={3: SimpleNamespace(name="TRANSPOSE")}
                            )
                        )
                    }
                )
            self.kernel_functions = ["kernel"]
            self.kernel_filenames = ["source.py"]

    for name in ["neuronxcc", "neuronxcc.proto", "neuronxcc.proto.ir_debug_info_pb2"]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["neuronxcc.proto.ir_debug_info_pb2"].ir_debug_info = FakeProto
    monkeypatch.setattr(
        labels.importlib.metadata, "version", lambda _: labels.SUPPORTED_COMPILER_VERSION
    )
    monkeypatch.setattr(labels, "collect_source_identity", _identity)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:

        def add(name, value):
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            tar.addfile(entry, io.BytesIO(value))

        for core in ["sg00", "sg01"]:
            for engine, (base, _) in labels.ENGINES.items():
                add(f"{core}/{base}.bin", bytes([212]) + bytes(63))
                add(
                    f"{core}/debug_info_asm_{engine}.dbg",
                    json.dumps(
                        {
                            "instructions": [
                                {"id": 0, "name": "", "attrs": {}, "parent_ids": [7, 99]}
                            ]
                        }
                    ).encode(),
                )
                add(
                    f"{core}/debug_info_backend_{engine}.dbg",
                    json.dumps(
                        {
                            "instructions": [
                                {
                                    "id": 7,
                                    "name": "first",
                                    "attrs": {"kernel_file": "0", "kernel_line": "1"},
                                },
                                {
                                    "id": 7,
                                    "name": "second",
                                    "attrs": {"kernel_file": "0", "kernel_line": "2"},
                                },
                            ]
                        }
                    ).encode(),
                )
    neff = tmp_path / "x.neff"
    neff.write_bytes(bytes(1024) + stream.getvalue())
    out = labels.export_static_map(neff)
    assert len(out["instructions"]) == 10
    for row in out["instructions"]:
        assert [p.get("kernel_line") for p in row["parents"]] == ["1", "2", None]
        assert row["parents"][-1] == {"id": 99, "missing": True}
        assert row["parents"][0]["kernel_file"] == "source.py"
        assert row["parents"][0]["instruction_type"] == "TRANSPOSE"
        assert row["parents"][0]["parent_ids"] == [12, 13]
        assert row["compiler_operation"] is None  # Missing parent99 prevents agreement.
    assert all(s["duplicate_backend_id_count"] == 1 for s in out["engine_summaries"])
    assert all(s["missing_backend_parent_count"] == 1 for s in out["engine_summaries"])
