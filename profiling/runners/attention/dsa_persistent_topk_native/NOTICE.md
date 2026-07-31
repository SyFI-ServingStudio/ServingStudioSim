# Pinned vLLM persistent top-k source

The files under `upstream/` are copied byte-for-byte from vLLM v0.23.0 commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` and remain licensed under
Apache-2.0. The precise upstream paths and SHA-256 digests are recorded in
`source_manifest.json` and verified before every build or load.

`binding.cpp` is VibeSim-owned compatibility scaffolding. It exposes the pinned
kernel through the private `_C_pinned_topk` Torch namespace needed by the
available Torch 2.10 stable ABI. It does not change the upstream selection math,
workspace reset, dispatch threshold, or launch sequence.
