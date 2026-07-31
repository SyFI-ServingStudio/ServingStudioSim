"""Lazy build support for the pinned vLLM persistent top-k CUDA op.

Importing this package is deliberately inert. The worker-only loader imports
Torch, validates the vendored source, and builds the extension only when the
production profiler entry point is invoked.
"""

from .loader import (
    NativeExtensionBuildError,
    NativeExtensionLoadError,
    NativeExtensionUnsupported,
    load_persistent_topk_op,
    verify_vendored_sources,
)

__all__ = [
    "NativeExtensionBuildError",
    "NativeExtensionLoadError",
    "NativeExtensionUnsupported",
    "load_persistent_topk_op",
    "verify_vendored_sources",
]
