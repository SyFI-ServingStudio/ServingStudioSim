"""Lazy build support for the corrected vLLM persistent top-k CUDA op.

Importing this package is deliberately inert. The worker-only loader imports
Torch, validates the pinned assets and correction overlay, and derives/builds
the extension only when the production profiler entry point is invoked.
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
