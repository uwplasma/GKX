"""Print the top XLA ops of a jax.profiler trace directory (xprof trace_viewer)."""

import collections
import glob
import json
import sys

from xprof.convert import raw_to_tool_data as r

path = sorted(glob.glob(f"{sys.argv[1]}/**/*.xplane.pb", recursive=True))[-1]
data, _ = r.xspace_to_tool_data([path], "trace_viewer", {})
events = json.loads(data)["traceEvents"]
total = collections.Counter()
count = collections.Counter()
for e in events:
    if e.get("ph") == "X" and "dur" in e:
        total[e["name"]] += e["dur"]
        count[e["name"]] += 1
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
for name, dur in total.most_common(top):
    print(f"{dur/1e3:10.2f} ms  x{count[name]:5d}  {name[:110]}")

skip = ("ThreadpoolListener", "ThunkExecutor", "$", "while")
groups = collections.Counter()
for name, dur in total.items():
    if name.startswith(skip):
        continue
    key = name.split(".")[0]
    key = "fft" if key.startswith("fft") else key
    groups[key] += dur
print("--- by kind (ms):", round(sum(groups.values()) / 1e3, 1))
for key, dur in groups.most_common(15):
    print(f"{dur/1e3:10.2f} ms  {key}")
