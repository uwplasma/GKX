"""Print one compact row per harness JSON line on stdin (window_vjp / e2e / window_fwd)."""
import json
import sys

for line in sys.stdin:
    name, _, rest = line.partition(" ")
    if name not in ("window_vjp", "e2e", "window_fwd"):
        continue
    d = json.loads(rest)
    keep = {k: d[k] for k in ("steady_median", "steady", "compile_s", "temp_bytes", "value", "grad") if k in d}
    if isinstance(keep.get("compile_s"), float):
        keep["compile_s"] = round(keep["compile_s"], 2)
    keep["steady_median"] = round(keep["steady_median"], 4)
    keep["steady"] = [round(x, 4) for x in keep["steady"]]
    print(name, json.dumps(keep), flush=True)
