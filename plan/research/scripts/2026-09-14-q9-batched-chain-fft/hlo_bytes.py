import re, sys, collections
import numpy as np

RX = re.compile(
    r"^\s*(?:ROOT\s+)?%[\w.\-]+ = (?:(?P<dtype>[a-z]+\d+)\[(?P<shape>[\d,]*)\](?:\{[\d,]*\})?|\(.*?\)) (?P<op>[a-z][\w\-]*)\((?P<args>[^)]*)\)",
    re.M,
)


def agg(path):
    out = collections.Counter()
    n = collections.Counter()
    for m in RX.finditer(open(path).read()):
        if m.group("op") in ("concatenate", "copy") and m.group("dtype"):
            bits = int(re.sub(r"\D", "", m.group("dtype")))
            dims = [int(d) for d in m.group("shape").split(",") if d]
            key = (m.group("op"), m.group("dtype"), m.group("shape"))
            out[key] += int(np.prod(dims, dtype=np.int64)) * max(1, bits // 8)
            n[key] += 1
    return out, n


a, na = agg(sys.argv[1])
b, nb = agg(sys.argv[2])
for k in sorted(set(a) | set(b)):
    if a[k] != b[k]:
        print(
            f"{k[0]:12s} {k[1]}[{k[2]}]  n {na[k]}->{nb[k]}  bytes {a[k]}->{b[k]}  delta {b[k] - a[k]}"
        )
print(
    "total",
    sum(a.values()),
    "->",
    sum(b.values()),
    "delta",
    sum(b.values()) - sum(a.values()),
)
