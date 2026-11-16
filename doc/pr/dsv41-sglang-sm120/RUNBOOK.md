# Rental runbook: profile and align deepseek_v41_sglang(_dspark) on RTX PRO 6000

Inputs ready on kisnis (branch `yilegu/dsv41-sglang-sm120`, worktree `wt-dsv41-sglang-sm120`):
- the runners and kinds (`SPLIT.md`);
- the arch and presets (`DESIGN.md`);
- the row list (`ROWS_TO_FILL.md`: 4,168 rows, 52 configs);
- the campaign pack `presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4`;
- the alignment fork `vibesim-alignment-v0.5.23` (worktree `wt-sglang-alignment-v0523`).

## One runtime for everything: the SGLang image

Run the ServingStudio launcher **inside** `lmsysorg/sglang:v0.5.23-cu130`, so kernels are timed on the exact FlashInfer and
sgl-kernel builds the server uses:

```bash
# on the rental, after cloning/rsyncing the worktree to ~/ss and the fork's python/ to ~/ss/alignment/profiler/sglang/python
ln -sfn /opt/sglang ~/ss/alignment/profiler/sglang/python/.venv-sglang   # the image's venv; resolved inside the container
sudo docker run -d --name ss --gpus all --ipc=host --network host --shm-size 64g \
  -v ~/ss:/ss -v ~/models:/models -e VIBESIM_PROFILE_RUNTIME=host \
  --entrypoint sleep lmsysorg/sglang:v0.5.23-cu130 infinity
sudo docker exec ss bash -lc 'git -C /sgl-workspace/sglang rev-parse HEAD'   # expect 297e54fd10...
sudo docker exec ss bash -lc 'apt-get update -qq && apt-get install -y -qq mold binutils clang'   # simulator build
sudo docker exec ss bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh; curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal'
sudo docker exec ss bash -lc 'cd /ss && ~/.local/bin/uv sync'                # project env for the launcher
```

`sglang_env` (`profiling/exec/env.py`) puts the fork's `python/` first on the import path, so the fork's
`sglang` shadows the image's own, and the image's native extensions and FlashInfer are used. Record
`pip show flashinfer-python flashinfer-cubin sgl-kernel torch` in each run's provenance.

Host settings, as root, **before** anything starts:
- `echo advise > /sys/kernel/mm/transparent_hugepage/shmem_enabled` (Engram host table);
- `swapoff -a`;
- check `free -g` (B1 needs about 110 GB free for one Engram table).

## B1: 1x RTX PRO 6000 -- DONE 2026-11-06 (4,149/4,168 rows, $2.79; see ROWS_TO_FILL.md), about 2-3 h (RunPod / Massed Compute 1x, about $1.7-2.5/h, so about $5-8)

No model download is needed.

1. **Boot and checks** (10 min): provenance (`nvidia-smi -q`, `lscpu`, `free -g`), host settings, image pull (40 GB,
   about 3 min), `uv sync`.
2. **Runner bring-up** (30-45 min). Run each new backend once at one shape and fix anything that fails its reference
   check:
   - `single_gemm/sglang_triton_block32_fp8`;
   - `nvfp4_fused_moe/flashinfer_cutlass_sm120_mxfp4`. Watch the W13 half order and the 8% bound;
   - `paged_sparse_mla` at rows 1 / 7 / 65 / 2048. Check FlashInfer's 16-head decode dispatch;
   - indexer decode and prefill;
   - the three mHC kinds;
   - `engram_gate`;
   - `engram_lookup`, on `host_shared_huge`.

   Commit fixes on the branch.
3. **JIT fill** (60-90 min): run both Decycle presets with JIT profiling on, so every missing non-comm row is
   measured. Then rerun `--cache-report`: only the 19 `all_reduce` rows may remain missing.
4. **Save back to kisnis:** `profiling/profile.db` (merge by row later), the runner logs, the version manifest.
   Delete the instance.

## B2: 4x RTX PRO 6000, about 3-4 h (bare metal with GPU P2P preferred; Massed Compute 4x VM $8.76/h, so about $26-35)

1. **Boot** (10 min). Provenance plus `nvidia-smi topo -m`, and a P2P probe (NCCL transport line `via P2P` /
   `via SHM`). Host settings as root. Start the checkpoint download in the background at once: 510 GB with
   `huggingface_hub<0.36` `snapshot_download`, about 45 min at 200 MB/s. Pin the snapshot hash in
   `recorded_on.checkpoint_snapshots`.
2. **Comm rows** (15-20 min, during the download): `all_reduce` / `nccl` at the dry-run shapes on 4 ranks. The rows
   need the topology identity (P2P vs SHM).
3. **Alignment campaign**, after the download (about 2 h for 14 cases; trim to the 7 DSpark cases plus 2 no-spec
   cases if time is short):
   - `alignment-campaign run --pack dsv41_flash_rtxpro6000_sglang_tp4_ep4 --host <rental host profile>`;
   - per case, the passes `profile_popularity`, `profile_workload` and `profile_nsys`;
   - host profile: `hf_hub_root=/models`, `nsys_executable=/usr/local/bin/nsys` (image), `fork_python` = the symlink
     above.
4. **Read back per case:** `max_total_num_tokens` (fills `attn_gpu_memory_gb`) and the DSpark acceptance per
   position from `spec_decode_metrics.json` (fills `speculative_acceptance` and the Decycle preset's acceptance).
5. **Save** captures, logs and `profile.db` to kisnis. Delete the instance and verify the deletion.

## After (kisnis, no GPU)

1. Write the label rules from the first `profile_nsys` capture (folded launch programs), not from the B200 pack.
2. Run `alignment-campaign label / extract / compare`, giving kernel-align and `gpu_time_multiplier`, then e2e-align.
3. Validate against the 2026-11-05 Decycle replay windows on this hardware:
   - typical: TTFT p90 2.0 / 14.0 s, TPOT p99 135 ms;
   - busy dense: 19.6 / 34.2 s, 121 ms.
4. Promote placeholders that rank in the comparison. First candidates: the indexer bookkeeping and top-k, `wo_a`,
   and the compressor.
5. Rerun the Decycle presets at 4 and 8 cards, and update the client cost comparison.

## Stop rules

- **B1:** stop if a runner cannot pass its reference check within about 30 min of debugging; leave it as a
  placeholder and note it.
- **Both:** hard cap of instance time plus 30 min. The instance is always deleted before reporting.
