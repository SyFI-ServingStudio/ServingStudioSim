# Trainium2 source replay tools

`replay_decode_buffer_capacity.py` writes the verified allocation-only candidate
for the pinned stock vLLM Neuron runner to a **new file**. It never installs a
patch, edits the source or writes profiling rows.

```bash
uv run python tools/trainium2/replay_decode_buffer_capacity.py \
  /path/to/original/neuron_model_runner.py /path/to/new/neuron_model_runner.py
```

The exact source SHA is `aa4f45c977bb6cc3e3a3e838c1b06fed643eb36ebfda4e04942df0a30a3b2562`;
the verified output SHA is `61af4bdaa2bd8f99b80e07c684837a6b784a247a5d64d0e80a9db6e6bf4e6c41`.
Unknown, already patched and existing output files are rejected, including under
`python -O`.

The change sizes `InputBatch` allocation to the greater of the prefill token
budget and the largest configured decode bucket. It preserves the scheduler,
prefill budget, model math and compiler settings. Its scope is ordinary
nonspeculative one-token decode. The B256 experiment passed 2048 native full-logit
vectors under the original vendor precision criterion; this does not extend the
registered C512/[1,16] profiling or campaign domain. Larger-capacity experiments
and their numerical results remain separate evidence.

The human-readable patch is `decode-buffer-capacity.patch`. The replay CLI
validates both complete source and candidate hashes instead of applying a fuzzy
patch to arbitrary framework versions. Preserve original installed sources and
run any candidate in an isolated copy or environment.
