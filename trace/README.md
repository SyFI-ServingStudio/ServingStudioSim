# Trace generators and samples

VibeSim and TraceLab's `vibesim` frontend consume the same independent-request
CSV schema:

```text
id,input_len,output_len,arrival_time
```

`arrival_time` is measured in milliseconds. Generate a deterministic fixed-shape
capacity workload with:

```bash
uv run python trace/generate_fixed_shape.py "$TMPDIR/fixed.csv" \
  --requests 1024 --input-len 256 --output-len 256 --interarrival-ms 0
```

Zero interarrival time is an intentional burst. Use a positive spacing when the
experiment needs a controlled offered rate rather than immediate saturation.
