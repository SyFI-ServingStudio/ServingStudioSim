# neuron-bench-import

Seeds `profiling/profile.db` with the Neuron (AWS Trainium2 LNC2) rows of GLM-5.3-Flash FP8 from the
glm5-trn2 engine's benchmark JSONLs, plus the trn2-flashnext branch's in-graph collective rows, and
fills the rest of the arch's grids with derived rows. Contract: `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`.

```bash
uv run --no-sync python tools/neuron-bench-import/import_bench.py \
    --bench-dir /home/ec2-user/glm5-trn2/logs/bench --fill-grid \
    --report doc/pr/trn2-glm53-flash/l1_coverage.json
```

Re-run it whenever the bench JSONLs grow: measured rows replace derived rows of the same key, and
every derived row is recomputed from the current measured ratios.

| File | Role |
|---|---|
| `import_bench.py` | CLI: collect rows, optional grid fill, insert, coverage report |
| `families/glm53.py` | bench row -> (kind, `neuron_<impl>`, per-rank args) |
| `families/collectives.py` | the trn2-flashnext `all_gather` / `reduce_scatter` rows (`data/`) |
| `geometry.py` | model geometry, per-rank args by layout, the Rust grids, default impls |
| `analytic.py` | Trainium2 roofline per region (shapes derived rows only) |
| `grid.py` | the derived fill: `analytic x` interpolated measured/analytic ratio |
