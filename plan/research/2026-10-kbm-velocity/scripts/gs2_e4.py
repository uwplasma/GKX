#!/usr/bin/env python3
"""GS2 velocity-grid rungs for the KBM (A_par) and KBMB (A_par + B_par) circular-Miller cases.

    python gs2_e4.py <root>  -> queue lines "gs2 <case> <timeout>"
Reuses the 2026-09-27 xcode generators (cases.py, em.py's em_text) unchanged; run from that bench.
e4v: the e3 spatial grid (ntheta 48, nperiod 3) with negrid 24, ngauss 12 (velocity only).
e4:  ntheta 64, nperiod 3, negrid 24, ngauss 12, delt .005.
"""

import sys
from pathlib import Path

sys.argv = [sys.argv[0], "write", "/dev/null/unused"] + sys.argv[1:]
import cases  # noqa: E402

cases.GS2_RUNGS.update(
    e4v=dict(ntheta=48, nperiod=3, negrid=24, ngauss=12, delt=0.01),
    e4=dict(ntheta=64, nperiod=3, negrid=24, ngauss=12, delt=0.005),
)
src = Path("em.py").read_text().split("\nroot = Path")[0]
exec(compile(src, "em.py", "exec"))  # defines em_text without writing the old queue
root = Path(sys.argv[3])
for fam, fbpar in (("KBM", 0.0), ("KBMB", 1.0)):
    for rung in ("e4v", "e4"):
        for ky in (0.1, 0.3, 0.5):
            c = dict(
                code="gs2",
                geom="M",
                phys="a",
                rung=rung,
                ky_gx=ky,
                ky_code=cases.SQRT2 * ky,
                t_max=100.0,
                name=f"{fam}_gs2_M_ky{ky:.2f}_{rung}",
            )
            d = root / "gs2" / c["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / (c["name"] + ".in")).write_text(em_text(c, 0.015, 1.0, fbpar))  # noqa: F821 (defined by the exec above)
            print("gs2 " + c["name"] + " 21600")
