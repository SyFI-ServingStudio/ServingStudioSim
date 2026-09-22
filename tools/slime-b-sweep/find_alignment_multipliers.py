import json, sys
from pathlib import Path

root = Path(sys.argv[1])
for f in sorted(root.rglob("alignment_iteration_report.json")):
    try:
        meta = (json.loads(f.read_text()).get("meta") or {})
    except Exception as exc:
        print(f"  ERR {f}: {exc}")
        continue
    v = meta.get("recommended_gpu_time_multiplier")
    label = str(f.relative_to(root)).replace("/reports/alignment_iteration_report.json", "")
    print(f"{'-' if v is None else round(v, 4):>8}  iters={meta.get('iterations')!s:<7} {label}")
