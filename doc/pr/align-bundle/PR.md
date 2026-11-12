# alignment-campaign: bundle / verify-bundle to hand profile passes to contributors

## Purpose

The GPU profile passes of an alignment pack need the hardware; everything after them does not. This lets
someone with the hardware run a pack's profile passes from an issue and return the captures, while the
maintainer recomputes every number locally.

- **`alignment-campaign bundle`** collects each case's profile pass directories into one directory that can
  be published.
  - It includes the `.nsys-rep` and the trace each pass replayed. It leaves out the nsys `.sqlite`, which is
    re-exported from the `.nsys-rep`.
  - `bundle.json` records each file's sha256 and size, each pass's state, and the pack's `campaign.yaml`
    sha256 and commit.
  - Text files have host paths, the user name, the host name and GPU UUIDs replaced by `${...}` placeholders.
    The host paths include the capture, case and out-root dirs, the checkout, home, the host profile's paths
    and checkpoints, and every mount the capture recorded.
  - Every file, binaries included, is then searched for the user name, host name, home and every non-shared
    host path. Any hit refuses the bundle.
  - `--upload` opens a PR on a Hugging Face dataset (`UW-SyFI/servingstudio-alignment-captures`), so a
    contributor needs no write access.
- **`alignment-campaign verify-bundle`** reads a local directory, or `--from-hf` at a pinned revision (a
  commit sha or `refs/pr/<n>`).
  - It accepts only the case's own profile pass names and safe relative paths.
  - It re-hashes every file and rejects a changed, missing or unlisted one.
  - It re-reads the capture conditions from the hashed files (engine image, model snapshot, replayed trace,
    GPU) and requires each to equal what the local checkout renders. A pass marked complete must contain its
    phase's required artifacts.
  - Only then does it place each pass: staged beside the case directory, the completion marker written last,
    then renamed into place, with the capture's `log_dir` restored to its new location.
- **Source fix.** `VIBESIM_AUTOTUNE_PREFLIGHT` now carries the warm-caches manifest's container path
  (`/cache/...`) instead of the host path. nsys records the environment in the `.nsys-rep`, so every capture
  named the host user and cache layout. Captures taken before this fix are refused by `bundle`.
- **Published as is.** nsys stores each GPU's bare UUID in the `.nsys-rep`. It cannot be scrubbed there, so
  it is published; text files have it replaced.
- **Docs.** `launcher/alignment_campaign/README.md` ("Handing captures off") and
  `skills/operate-run-alignment` (contributor and maintainer commands).

## Test Plan

- `CUDA_VISIBLE_DEVICES= just test-cpu`.
- `tests/test_alignment_campaign_bundle.py` covers:
  - scrubbing, and refusal of a leak in a binary;
  - pass-name and path validation;
  - conditions re-read from the files;
  - atomic placement;
  - pinned revisions, and the `--from-hf` download filter.
- Round trip on a real capture (the DSv4.1-Flash DP4 alignment Phase 1 case from #125, without its
  `.nsys-rep`, which predates the source fix): `bundle`, then `verify-bundle` into a freshly rendered
  campaign, then timing_predict, label, analysis_kernel, simulation, analysis_e2e, extract and compare.

## Test Result

- `just test-cpu`: Rust `--lib` 1583 passed. pytest 5204 passed and 4 failed; the 4 are the
  `tests/test_l1_single_gemm.py::test_energy_*` tests, which fail the same way on `main`.
- `tests/test_alignment_campaign_bundle.py`: 24 passed.
- Round trip: the recomputed DP4 metrics equal the original `alignment_phase1.md` row: kernel_abs +1.58%,
  e2e −5.15%, TTFT −10.29%, and the rest.
- The real DSv4.1 DP4 `.nsys-rep` held the host user only through `VIBESIM_AUTOTUNE_PREFLIGHT`, 5 times;
  `bundle` refuses it.
- `verify-bundle` on that case rejected the workload pass: it ran in image `cbe2f57bc603b683` while the
  checkout renders `fe6a3d9a6af96e67`. It is being recaptured for #125.

Not run: a real upload to and download from the Hugging Face dataset.

## Dependencies

None.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
