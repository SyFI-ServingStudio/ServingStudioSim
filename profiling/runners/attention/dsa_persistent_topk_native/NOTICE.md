# Corrected vLLM v0.23-derived persistent top-k source

The files under `upstream/` are copied byte-for-byte from vLLM v0.23.0 commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` and remain licensed under
Apache-2.0. The precise upstream paths and SHA-256 digests are recorded in
`source_manifest.json` and verified before every build or load.

The executable is not byte-identical vLLM v0.23 production. The worker loader
applies the machine-verifiable `correction_manifest.json` overlay to
cache-local build copies while leaving `upstream/` untouched. Natural-index and
`-1` handling remains unchanged for lengths through `top_k`; all longer rows
use v0.23's bundled cooperative radix implementation. The fixed-capacity short,
medium, and FilteredTopK selectors are unreachable.

The overlay also gives each CTA group a `radix_iter` that advances only after a
radix row. This preserves triple-buffer rotation when trivial and radix rows
alternate across resident-group iterations. Exact replacements, input hashes,
result hashes, and the correction identity are verified before build and
included in the cache fingerprint and completion marker.

`binding.cpp` is ServingStudio Sim-owned compatibility scaffolding. It exposes the
corrected v0.23-derived kernel through the private `_C_pinned_topk` Torch
namespace needed by the available Torch 2.10 stable ABI.
