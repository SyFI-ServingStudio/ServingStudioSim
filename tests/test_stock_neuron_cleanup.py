"""Execute the exact installed cleanup body with CPU-only lifecycle doubles."""

import hashlib
import importlib.util
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from alignment.neuron import vllm_cleanup as cleanup

FIXTURE = Path(__file__).parent / "fixtures/stock_neuron_cleanup.txt"


@pytest.fixture
def original(monkeypatch):
    spec = importlib.util.spec_from_loader(
        "pinned_cleanup_fixture", SourceFileLoader("pinned_cleanup_fixture", str(FIXTURE))
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def record(name):
        return lambda *args, **kwargs: calls.append(name)

    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = NS(
        device_type="neuron", is_cpu=lambda: False, is_rocm=lambda: False
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    monkeypatch.setitem(sys.modules, "ray", NS(shutdown=record("ray")))
    module.logger = NS(
        debug=record("start"), debug_once=record("complete"), warning=record("warning")
    )
    module.envs = NS(disable_envs_cache=record("env"))
    module.gc = NS(unfreeze=record("unfreeze"), collect=record("gc"))
    module.destroy_model_parallel = record("model_groups")
    module.destroy_distributed_environment = record("world_group")
    module.torch = NS(
        accelerator=NS(empty_cache=record("device_cache")),
        _C=NS(_host_emptyCache=record("host_cache")),
    )
    return module, calls, platforms.current_platform


def test_fixture_is_exact_pinned_function():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == cleanup.CLEANUP_SOURCE_SHA256


@pytest.mark.parametrize("skip", [True, False])
def test_actual_cleanup_order_and_only_device_flush_difference(original, skip):
    module, calls, _ = original
    flush = module.torch.accelerator.empty_cache
    adapted = cleanup.adapted_cleanup(module.cleanup_dist_env_and_memory, lambda: skip)
    adapted(shutdown_ray=True)
    expected = ["start", "env", "unfreeze", "model_groups", "world_group", "ray", "gc"]
    if not skip:
        expected.append("device_cache")
    assert calls == expected + ["host_cache", "complete"]
    assert module.torch.accelerator.empty_cache is flush


def test_cpu_backend_still_skips_both_cache_operations(original):
    module, calls, platform = original
    platform.is_cpu = lambda: True
    cleanup.adapted_cleanup(module.cleanup_dist_env_and_memory, lambda: False)()
    assert calls == ["start", "env", "unfreeze", "model_groups", "world_group", "gc", "complete"]


@pytest.mark.parametrize("operation", ["destroy_model_parallel", "device_cache", "host_cache"])
def test_cleanup_errors_propagate(original, operation):
    module, calls, _ = original

    def fail():
        raise RuntimeError("cleanup failure")

    if operation == "destroy_model_parallel":
        module.destroy_model_parallel = fail
    elif operation == "device_cache":
        module.torch.accelerator.empty_cache = fail
    else:
        module.torch._C._host_emptyCache = fail
    adapted = cleanup.adapted_cleanup(
        module.cleanup_dist_env_and_memory, lambda: operation != "device_cache"
    )
    with pytest.raises(RuntimeError, match="cleanup failure"):
        adapted()
    assert "complete" not in calls


def test_unknown_cleanup_source_rejected():
    with pytest.raises(RuntimeError, match="differs from pinned"):
        cleanup.adapted_cleanup(test_unknown_cleanup_source_rejected, lambda: True)


def runtime_modules(monkeypatch, module):
    package = Path("/pinned/lite")

    def owner():
        return True

    lite = NS(__file__=str(package / "__init__.py"), owns_privateuse1=owner)
    core = NS(cleanup_dist_env_and_memory=module.cleanup_dist_env_and_memory)
    parallel = NS(
        __file__="/pinned/parallel_state.py",
        cleanup_dist_env_and_memory=module.cleanup_dist_env_and_memory,
    )
    monkeypatch.setitem(sys.modules, "libtorch_neuronx_lite", lite)
    monkeypatch.setitem(sys.modules, "vllm.distributed", NS(parallel_state=parallel))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", NS(core=core))
    monkeypatch.setattr(cleanup.importlib.metadata, "version", lambda _: cleanup.LITE_VERSION)
    original_getfile = cleanup.inspect.getfile
    monkeypatch.setattr(
        cleanup.inspect,
        "getfile",
        lambda fn: "/pinned/lite/_config.py" if fn is owner else original_getfile(fn),
    )
    hashes = {
        Path(parallel.__file__): cleanup.PARALLEL_STATE_SHA256,
        package / "__init__.py": cleanup.LITE_INIT_SHA256,
        package / "_config.py": cleanup.LITE_CONFIG_SHA256,
        package / "lib/libtorchneuron.so": cleanup.ALLOCATOR_SHA256,
    }
    monkeypatch.setattr(cleanup, "_sha", lambda p: hashes[Path(p)])
    return lite, core, parallel, hashes


def test_install_binds_only_engine_core_and_is_idempotent(monkeypatch, original):
    module, calls, platform = original
    lite, core, parallel, _ = runtime_modules(monkeypatch, module)
    receipt = cleanup.install_cleanup_adapter()
    assert cleanup.install_cleanup_adapter() is receipt
    assert parallel.cleanup_dist_env_and_memory is module.cleanup_dist_env_and_memory
    core.cleanup_dist_env_and_memory()
    assert "host_cache" in calls and "device_cache" not in calls
    calls.clear()
    platform.device_type = "cuda"
    core.cleanup_dist_env_and_memory()
    assert "device_cache" in calls  # Other backend keeps the original operation.
    calls.clear()
    platform.device_type = "neuron"
    lite.owns_privateuse1 = lambda: False
    core.cleanup_dist_env_and_memory()
    assert "device_cache" in calls  # Native ownership receives no lite exemption.


def test_install_leaves_cpu_platform_unchanged(original):
    _, _, platform = original
    platform.device_type = "cpu"
    assert cleanup.install_cleanup_adapter() is None


@pytest.mark.parametrize("mutation", ["alias", "version", "owner", "binary"])
def test_unknown_runtime_capabilities_fail_closed(monkeypatch, original, mutation):
    module, _, _ = original
    lite, core, _, hashes = runtime_modules(monkeypatch, module)
    if mutation == "alias":
        core.cleanup_dist_env_and_memory = lambda: None
    elif mutation == "version":
        monkeypatch.setattr(cleanup.importlib.metadata, "version", lambda _: "other")
    elif mutation == "owner":
        lite.owns_privateuse1 = lambda: False
    else:
        hashes[Path("/pinned/lite/lib/libtorchneuron.so")] = "0" * 64
    with pytest.raises(RuntimeError):
        cleanup.install_cleanup_adapter()
