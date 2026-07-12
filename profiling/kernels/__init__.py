"""Per-kernel modules. Each module's import triggers a side-effect
``register(KernelProfilerSpec(...))`` call that wires the kind into the shared
registry.

Add a kernel by creating ``profiling/kernels/<kind>.py`` with a ``KIND``
constant, the ``<Kind>Args`` dataclass, and a ``register(...)`` call, then add
``from . import <kind>`` to this barrel.

Symmetric with Rust ``simulator/src/timing/kernels/<kind>.rs``: one file per
kernel kind owns its Python wire format and registry presence.
"""

from profiling.kernels import (
    all_reduce,  # noqa: F401
    elementwise,  # noqa: F401
    flashinfer_attn_decode,  # noqa: F401
    flashinfer_attn_prefill,  # noqa: F401
    flashinfer_attn_rect,  # noqa: F401
    grouped_gemm,  # noqa: F401
    kv_cache_append,  # noqa: F401
    p2p_inter,  # noqa: F401
    p2p_intra,  # noqa: F401
    rms_norm,  # noqa: F401
    single_gemm,  # noqa: F401
)
