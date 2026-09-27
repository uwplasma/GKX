#!/usr/bin/env python3
"""Stella 1.4x gap: term-knockout variants on circular Miller, ky_gx = 0.30 (Lref = a).

    python gap.py write <root>   -> <root>/{gs2,stella}/<case>/<case>.in and prints a queue

Variants (same physics otherwise as cases.py M): base, nodrift (both drifts off),
nomirror (stella only; GS2 has no mirror knob), nodrift_nomirror (stella only).
"""
import sys
from pathlib import Path
import cases

SUB = {
    "gs2": {"nodrift": [(" gridfac = 1.0\n", " gridfac = 1.0\n driftknob = 0.0\n")]},
    "stella": {
        "nodrift": [("  include_xdrift = .true.", "  include_xdrift = .false."), ("  include_ydrift = .true.", "  include_ydrift = .false.")],
        "nomirror": [("  include_mirror = .true.", "  include_mirror = .false.")],
    },
}
SUB["stella"]["nodrift_nomirror"] = SUB["stella"]["nodrift"] + SUB["stella"]["nomirror"]
RUNG = {"gs2": "r3", "stella": "r1"}

root = Path(sys.argv[2])
for code in ("gs2", "stella"):
    for var in ["base", *SUB[code]]:
        c = dict(code=code, geom="M", phys="a", rung=RUNG[code], ky_gx=0.30, ky_code=cases.SQRT2 * 0.30, t_max=300.0,
                 name=f"gap_{code}_M_ky0.30_{RUNG[code]}_{var}")
        text = cases.gs2_input(c) if code == "gs2" else cases.stella_input(c)
        for a, b in SUB[code].get(var, []):
            assert text.count(a) == 1, (code, var, a)
            text = text.replace(a, b)
        d = root / code / c["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / (c["name"] + ".in")).write_text(text)
        print(code, c["name"], 7200)
