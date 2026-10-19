"""Pinned lite-Neuron cleanup capability adapter for the stock EngineCore.

The lite allocator directly allocates/frees NRT tensors. It has no PyTorch
caching allocator and is not a DeviceAllocator, so Torch's generic empty_cache
query asserts. Preserve vLLM cleanup except that unsupported cache flush.
"""

import ast
import copy
import hashlib
import importlib.metadata
import inspect
from pathlib import Path

PARALLEL_STATE_SHA256 = "1feda9ac98c1dd14cbad60a9205194d224ad7d75656c6e94cffbe3bdbcbf4003"
CLEANUP_SOURCE_SHA256 = "f9ccebcac1ef771ac5f16bd130b9e1a31e7e3ef0c716326655c0a858a5380347"
LITE_INIT_SHA256 = "b70d1fec2b880db1c7c611837f0c9d70251493297a3e1ce8f0306787ee654a16"
LITE_CONFIG_SHA256 = "5e5d071eb33c5410ddd888120988f73131404490354fed2673a76a41cf821cf6"
ALLOCATOR_SHA256 = "93f44b4203bfce5b5515dfc168325c53167fd96a6fa4c2bf01fe0ca490a652f3"
LITE_VERSION = "2.11.0.1.0.1284+f49d8626"


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapted_cleanup(original, skip_device_cache_flush):
    """Replace only the pinned cache call; retain its other operations verbatim."""
    source = inspect.getsource(original)
    if hashlib.sha256(source.encode()).hexdigest() != CLEANUP_SOURCE_SHA256:
        raise RuntimeError("stock cleanup function differs from pinned noncaching adapter")
    tree = ast.parse(source)
    changed = copy.deepcopy(tree)
    calls = [
        node
        for node in ast.walk(changed)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.accelerator.empty_cache"
    ]
    if len(calls) != 1 or calls[0].args or calls[0].keywords:
        raise RuntimeError("stock cleanup cache-call ABI differs")
    calls[0].func = ast.Name(id="_flush_supported_device_cache", ctx=ast.Load())
    ast.fix_missing_locations(changed)

    def flush_supported_device_cache():
        if not skip_device_cache_flush():
            original.__globals__["torch"].accelerator.empty_cache()

    namespace = {
        **original.__globals__,
        "_flush_supported_device_cache": flush_supported_device_cache,
    }
    exec(compile(changed, original.__code__.co_filename, "exec", dont_inherit=True), namespace)
    result = namespace[original.__name__]
    if inspect.signature(result) != inspect.signature(original):
        raise RuntimeError("stock cleanup signature changed")
    return result


def install_cleanup_adapter():
    """Bind the exact EngineCore alias after verifying backend and allocator bytes."""
    from vllm.platforms import current_platform

    if current_platform.device_type != "neuron":
        return None
    import libtorch_neuronx_lite as lite
    from vllm.distributed import parallel_state
    from vllm.v1.engine import core

    installed = getattr(core.cleanup_dist_env_and_memory, "_stock_neuron_cleanup_receipt", None)
    if installed is not None:
        return installed
    if core.cleanup_dist_env_and_memory is not parallel_state.cleanup_dist_env_and_memory:
        raise RuntimeError("stock EngineCore cleanup alias is already modified")
    if importlib.metadata.version("libtorch-neuronx-lite") != LITE_VERSION:
        raise RuntimeError("stock cleanup requires the pinned lite allocator version")
    if not lite.owns_privateuse1():
        raise RuntimeError("stock cleanup requires lite to own Neuron PrivateUse1")
    root = Path(lite.__file__).parent
    files = {
        Path(parallel_state.__file__): PARALLEL_STATE_SHA256,
        root / "__init__.py": LITE_INIT_SHA256,
        Path(inspect.getfile(lite.owns_privateuse1)): LITE_CONFIG_SHA256,
        root / "lib/libtorchneuron.so": ALLOCATOR_SHA256,
    }
    if any(_sha(path) != expected for path, expected in files.items()):
        raise RuntimeError("stock cleanup/allocator differs from pinned noncaching authority")

    def skip_flush():
        from vllm.platforms import current_platform

        return current_platform.device_type == "neuron" and lite.owns_privateuse1()

    compatible = adapted_cleanup(parallel_state.cleanup_dist_env_and_memory, skip_flush)
    receipt = {
        "scope": "EngineCore cleanup; generic caching-allocator flush only",
        "allocator": "pinned lite Neuron noncaching; lifetime frees NRT tensors",
        "files": {str(path): expected for path, expected in files.items()},
        "cleanup_source_sha256": CLEANUP_SOURCE_SHA256,
        "torch_global_behavior_modified": False,
    }
    compatible._stock_neuron_cleanup_receipt = receipt
    core.cleanup_dist_env_and_memory = compatible
    return receipt
