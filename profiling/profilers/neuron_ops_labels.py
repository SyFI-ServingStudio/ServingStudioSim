"""Export an exact cached NEFF PC map using its own embedded debug records.

Optional offline helper for the pinned Neuron SDK, requiring no device. Output
labels are compiler metadata evidence, not measured timings or boundary claims.
All assembly parents are retained; ambiguities and missing parents are explicit.
"""

import collections
import copy
import hashlib
import importlib.metadata
import importlib.util
import io
import re
import tarfile
from pathlib import Path

SUPPORTED_COMPILER_VERSION = "2.27.5334.0+f702b353"
SUPPORTED_SOURCE_HASHES = {
    "vllm_neuron/model/llama3/model.py": (
        "0bab88d1c07e279550ef669e753ba7a7b541370a51ef8a58b07bfd0ed01b0823"
    ),
    "vllm_neuron/functional/attention/attention_decode.py": (
        "a98e0fb751c0a6f70bd2ea8b9ba7d3dcf7c2d56acdafac1b11d57449c4d64ba1"
    ),
    "vllm_neuron/functional/mlp.py": (
        "86db8b306f5dec83ab8bceb24f8e725f0b176dcd675ea8cfdd0f60cdd344d104"
    ),
    "vllm_neuron/functional/sampling.py": (
        "5be28b1101e8380141a069a5794f5154bd9a3433e5e062559d18229adbaced08"
    ),
}


def collect_source_identity():
    """Hash the four installed source files used by the pinned semantic map."""
    spec = importlib.util.find_spec("vllm_neuron")
    if spec is None or spec.origin is None:
        raise ValueError("The pinned vllm_neuron source package is unavailable")
    package_root = Path(spec.origin).parent
    hashes = {}
    for source in SUPPORTED_SOURCE_HASHES:
        path = package_root / source.split("/", 1)[1]
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"source_hashes": hashes}


def _validate_source_identity(identity):
    hashes = identity.get("source_hashes", {})
    for source, expected in SUPPORTED_SOURCE_HASHES.items():
        if hashes.get(source) != expected:
            raise ValueError(f"Unsupported semantic source identity: {source}")


def _validate_map(data, expected_neff_sha256, source_identity):
    if data.get("schema_version") != 1 or data.get("instruction_size") != 64:
        raise ValueError("Unsupported static instruction map schema")
    if data.get("decoder_version") != SUPPORTED_COMPILER_VERSION:
        raise ValueError("Unsupported static map compiler ABI")
    digest = data.get("neff_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Static map lacks an exact NEFF SHA256")
    if expected_neff_sha256 is not None and digest != expected_neff_sha256:
        raise ValueError("Static map NEFF SHA256 differs from the requested graph")
    _validate_source_identity(data.get("source_identity", {}))
    if data["source_identity"]["source_hashes"] != source_identity["source_hashes"]:
        raise ValueError("Static map source identity differs from the installed source")
    seen = set()
    for row in data["instructions"]:
        key = row["pcore_idx"], row["engine"], row["compiler_pc"]
        if key in seen:
            raise ValueError(f"Duplicate static instruction identity: {key}")
        seen.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("instruction_sha256", "")):
            raise ValueError("Static instruction lacks an exact binary SHA256")


ENGINES = {
    "PE": ("PE0", "Tensor"),
    "Activation": ("Activation0", "Scalar"),
    "DVE": ("DVE0", "Vector"),
    "Pool": ("Pool0", "GpSimd"),
    "SP": ("SP0", "Sync"),
}


def compiler_operation_from_parents(parents):
    """Classify a compiler primitive only when every embedded candidate agrees.

    This establishes a primitive kind, not model ownership or whether the
    compiler introduced the operation. Missing parents and mixed types remain
    unknown. Debug-build parents never replace the stock proof.
    """
    if parents and all(
        not parent.get("missing") and parent.get("instruction_type") == "TRANSPOSE"
        for parent in parents
    ):
        return "transpose"
    return None


def export_static_map(neff_path, *, expected_compiler_version=SUPPORTED_COMPILER_VERSION):
    """Read the pinned SDK debug schema without modifying or executing the NEFF."""
    data = Path(neff_path).read_bytes()
    try:
        version = importlib.metadata.version("neuronx-cc")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    if version != SUPPORTED_COMPILER_VERSION or expected_compiler_version != version:
        raise ValueError(
            f"Unsupported Neuron compiler ABI: {version!r}; expected {expected_compiler_version!r}"
        )
    import neuronxcc.proto.ir_debug_info_pb2 as pb

    source_identity = collect_source_identity()
    result = {
        "schema_version": 1,
        "source_identity": source_identity,
        "neff_sha256": hashlib.sha256(data).hexdigest(),
        "neff_bytes": len(data),
        "decoder_distribution": "neuronx-cc",
        "decoder_version": version,
        "method": (
            "Exact embedded assembly index joined to all embedded backend parents; "
            "no cross-NEFF label transfer."
        ),
        "instruction_size": 64,
        "engine_summaries": [],
        "instructions": [],
    }
    with tarfile.open(fileobj=io.BytesIO(data[1024:]), mode="r:*") as tar:
        names = set(tar.getnames())

        def read(name):
            return tar.extractfile(name).read()

        cores = sorted({x.split("/")[0] for x in names if re.match(r"sg\d+/PE0.bin$", x)})
        if not cores:
            raise ValueError("NEFF has no supported physical-core instruction streams")
        for core in cores:
            for engine, (base, capture_engine) in ENGINES.items():
                binary = read(f"{core}/{base}.bin")
                if len(binary) % 64:
                    raise ValueError(f"{core}/{base}: partial instruction")
                asm = pb.ir_debug_info()
                asm.ParseFromString(read(f"{core}/debug_info_asm_{engine}.dbg"))
                backend = pb.ir_debug_info()
                backend.ParseFromString(read(f"{core}/debug_info_backend_{engine}.dbg"))
                if len(asm.instructions) != len(binary) // 64:
                    raise ValueError(
                        f"{core}/{engine}: assembly record count differs from instruction count"
                    )
                backend_by_id = collections.defaultdict(list)
                for ins in backend.instructions:
                    attrs = dict(ins.attrs)
                    type_enum = ins.DESCRIPTOR.fields_by_name["instruction_type"].enum_type
                    type_value = type_enum.values_by_number.get(ins.instruction_type)

                    def lookup(value, entries):
                        if value.isdigit():
                            return entries[int(value)] if int(value) < len(entries) else ""
                        return value

                    backend_by_id[ins.id].append(
                        {
                            "id": ins.id,
                            "name": ins.name,
                            "attrs": attrs,
                            "instruction_type": type_value.name if type_value else "UNKNOWN",
                            "instruction_type_code": ins.instruction_type,
                            "parent_ids": list(ins.parent_ids),
                            "kernel_name": lookup(
                                attrs.get("kernel_name", ""), backend.kernel_functions
                            ),
                            "kernel_file": lookup(
                                attrs.get("kernel_file", ""), backend.kernel_filenames
                            ),
                            "kernel_line": attrs.get("kernel_line", ""),
                        }
                    )
                fn = "main"
                cardinalities = collections.Counter()
                missing = 0
                source_count = 0
                for pc, asm_ins in enumerate(asm.instructions):
                    raw = binary[64 * pc : 64 * (pc + 1)]
                    opcode = raw[0]
                    if opcode == 209:
                        fn = raw[12:32].split(b"\0")[0].decode()
                    parent_ids = list(asm_ins.parent_ids)
                    cardinalities[len(parent_ids)] += 1
                    parents = [
                        record
                        for x in parent_ids
                        for record in backend_by_id.get(x, [{"id": x, "missing": True}])
                    ]
                    missing += sum(p.get("missing", False) for p in parents)
                    source_count += any(p.get("kernel_file") for p in parents)
                    compiler_operation = compiler_operation_from_parents(parents)
                    row = {
                        "pcore_idx": int(core[2:]),
                        "engine": capture_engine,
                        "compiler_pc": pc,
                        "function": fn,
                        "opcode_byte": opcode,
                        "instruction_sha256": hashlib.sha256(raw).hexdigest(),
                        "asm_id": asm_ins.id,
                        "asm_name": asm_ins.name,
                        "asm_attrs": dict(asm_ins.attrs),
                        "asm_parent_ids": parent_ids,
                        "parents": parents,
                        "compiler_operation": compiler_operation,
                        "compiler_operation_evidence": (
                            "unanimous_backend_instruction_type" if compiler_operation else None
                        ),
                    }
                    if opcode == 212:
                        row["dge_subopcode_byte"] = raw[15]
                    result["instructions"].append(row)
                    if opcode == 210:
                        fn = "main"
                result["engine_summaries"].append(
                    {
                        "pcore_idx": int(core[2:]),
                        "engine": capture_engine,
                        "binary_sha256": hashlib.sha256(binary).hexdigest(),
                        "instruction_count": len(binary) // 64,
                        "asm_parent_cardinality": dict(cardinalities),
                        "missing_backend_parent_count": missing,
                        "duplicate_backend_id_count": sum(
                            len(records) > 1 for records in backend_by_id.values()
                        ),
                        "instructions_with_kernel_file": source_count,
                    }
                )
    return result


def _groups(data):
    result = collections.defaultdict(list)
    for row in data["instructions"]:
        result[row["pcore_idx"], row["engine"], row["function"]].append(row)
    for rows in result.values():
        rows.sort(key=lambda r: r["compiler_pc"])
    return result


def _digest(hashes):
    return hashlib.sha256(b"".join(bytes.fromhex(h) for h in hashes)).hexdigest()


def _debug_owner(row, source_hashes):
    """Return an owner only when every backend candidate identifies that owner."""
    owners = set()
    for parent in row["parents"]:
        operation = source_operation(
            f"{parent.get('kernel_file', '')}:{parent.get('kernel_line', '')}",
            source_hashes,
        )
        owner = operation or parent.get("kernel_name")
        if parent.get("missing") or not owner:
            return None
        owners.add(owner)
    return next(iter(owners)) if len(owners) == 1 else None


def transfer_debug_labels(
    stock,
    debug,
    *,
    source_identity,
    expected_stock_neff_sha256=None,
    expected_debug_neff_sha256=None,
):
    """Return a stock map supplemented only by exact, unique debug evidence.

    Timings always belong to the stock capture. Matching identical instruction
    bytes does not prove additive operation timing or equal complete schedules.
    """
    _validate_source_identity(source_identity)
    _validate_map(stock, expected_stock_neff_sha256, source_identity)
    _validate_map(debug, expected_debug_neff_sha256, source_identity)
    result = copy.deepcopy(stock)
    a, b = _groups(result), _groups(debug)
    receipts = []
    totals = collections.Counter()
    for key, target in a.items():
        source = b.get(key, [])
        ah = [r["instruction_sha256"] for r in target]
        bh = [r["instruction_sha256"] for r in source]
        matches = {}
        whole_equal = bool(source) and ah == bh
        if whole_equal:
            matches = {i: (i, "identical_function_stream", None) for i in range(len(target))}
        elif source:
            source_owners = [_debug_owner(row, source_identity["source_hashes"]) for row in source]
            for radius in [8, 16, 32]:

                def contexts(hashes):
                    out = collections.defaultdict(list)
                    for i in range(radius, len(hashes) - radius):
                        out[tuple(hashes[i - radius : i + radius + 1])].append(i)
                    return out

                ac, bc = contexts(ah), contexts(bh)
                for context, stock_indices in ac.items():
                    source_indices = bc.get(context, [])
                    if len(stock_indices) == 1 and len(source_indices) == 1:
                        i, j = stock_indices[0], source_indices[0]
                        owner = source_owners[j]
                        context_owners = {
                            value
                            for value in source_owners[j - radius : j + radius + 1]
                            if value is not None
                        }
                        if owner is None or context_owners != {owner}:
                            continue
                        matches.setdefault(i, (j, "unique_binary_context", radius))
        used = collections.Counter(j for j, _, _ in matches.values())
        counts = collections.Counter({"matched": 0, "matched_with_source": 0})
        ah_digest, bh_digest = (_digest(ah), _digest(bh)) if whole_equal else (None, None)
        for i, (j, method, radius) in matches.items():
            if used[j] != 1:
                counts["rejected_non_bijective"] += 1
                continue
            assert ah[i] == bh[j]
            target[i]["matched_debug_parents"] = copy.deepcopy(source[j]["parents"])
            target[i]["debug_match"] = {
                "method": method,
                "debug_neff_sha256": debug["neff_sha256"],
                "debug_compiler_pc": source[j]["compiler_pc"],
                "instruction_sha256": ah[i],
                "context_radius": radius,
                "label_confidence": "exact_function" if whole_equal else "inferred_context",
                "homogeneous_context_owner": None if whole_equal else source_owners[j],
                "stock_function_stream_sha256": ah_digest,
                "debug_function_stream_sha256": bh_digest,
            }
            counts["matched"] += 1
            counts[method] += 1
            counts["matched_with_source"] += any(p.get("kernel_file") for p in source[j]["parents"])
        counts["stock_instructions"] = len(target)
        counts["debug_instructions"] = len(source)
        counts["unmatched"] = len(target) - counts["matched"]
        totals.update(counts)
        receipts.append(
            {
                "pcore_idx": key[0],
                "engine": key[1],
                "function": key[2],
                "whole_function_equal": whole_equal,
                **dict(counts),
            }
        )
    result["debug_transfer"] = {
        "schema_version": 1,
        "source_identity": copy.deepcopy(source_identity),
        "stock_neff_sha256": stock["neff_sha256"],
        "debug_neff_sha256": debug["neff_sha256"],
        "policy": (
            "Whole function equality proves relative PCs. Context inference requires at least "
            "17 equal instructions unique in both functions and one consistent known source "
            "owner; unmatched retains only stock evidence."
        ),
        "totals": dict(totals),
        "functions": receipts,
    }
    return result


# Source-line classifications are enabled only for the verified package files.
SOURCE_HASHES = SUPPORTED_SOURCE_HASHES


def source_operation(source: str, source_hashes: dict) -> str | None:
    match = re.search(r"(vllm_neuron/.*):(\d+)$", source)
    if not match:
        return None
    filename, line = match.group(1), int(match.group(2))
    if filename not in SOURCE_HASHES or source_hashes.get(filename) != SOURCE_HASHES[filename]:
        return None
    if filename.endswith("attention_decode.py"):
        for lower, upper, operation in (
            (418, 451, "rmsnorm"),
            (454, 509, "rope"),
            (801, 804, "rmsnorm"),
            (809, 811, "qkv_projection"),
            (816, 822, "qkv_layout"),
            # The pinned compiler assigns projection MATMUL/LDWEIGHTS and its
            # fused head-layout work to the transpose at line 823. Source
            # metadata cannot separate those costs into independent ops.
            (823, 829, "qkv_projection_and_layout"),
            (834, 836, "rmsnorm"),
            (884, 887, "rope"),
            (891, 893, "rmsnorm"),
            (905, 990, "kv_gather"),
            (991, 1040, "attention"),
            (1080, 1162, "kv_update"),
            (1163, 1180, "output_projection"),
        ):
            if lower <= line <= upper:
                return operation
    if filename.endswith("model/llama3/model.py"):
        if line in (1270, 1293, 1323, 1341):
            return "residual_add"
        for lower, upper, operation in (
            (159, 165, "rmsnorm"),
            (230, 281, "rope"),
            (855, 870, "rope"),
            (590, 624, "kv_update"),
            (790, 820, "attention_layout"),
        ):
            if lower <= line <= upper:
                return operation
    if filename.endswith("functional/mlp.py"):
        return "mlp"
    if filename.endswith("functional/sampling.py"):
        return "sampling"
    return None
