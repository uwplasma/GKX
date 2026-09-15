import json
import sys

a, b = sys.argv[1], sys.argv[2]
keys = ("fft", "concatenate", "copy", "transpose", "gather", "reverse", "bytes_written")
for route in ("diagnostics", "runtime"):
    for grid in ("32", "64"):
        A = json.load(open(f"{a}/{route}_{grid}.json"))
        B = json.load(open(f"{b}/{route}_{grid}.json"))
        if route == "diagnostics":
            print(
                f"rhs {grid}: "
                + "  ".join(f"{k} {A['rhs'][k]}->{B['rhs'][k]}" for k in keys)
            )
        for m in ("rk3", "rk4"):
            ca, cb = A["steps"][m]["counts"], B["steps"][m]["counts"]
            print(
                f"{route:11s} {grid} {m}: "
                + "  ".join(f"{k} {ca[k]}->{cb[k]}" for k in keys)
            )
