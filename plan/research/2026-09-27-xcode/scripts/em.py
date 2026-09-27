#!/usr/bin/env python3
"""GS2 inputs for the kinetic-electron (VAL-KE), KBM (VAL-REF) and three-field KBM (EM-B-PAR)
circular-Miller cases of the GX benchmark decks, plus the s-alpha ky=0.55 energy-grid extension.

    python em.py write <root>   -> prints queue lines "<code> <case> <timeout>"
Physics (GX units, Lref = a): rhoc .5, q 1.4, shat .8, R 2.77778, a/LT 2.49, a/Ln .8 for both
species, m_e = 2.7e-4 m_i, Te = Ti; KE: beta 1e-5 with A_par (as the GX deck); KBM: beta .015,
betaprim 0, A_par only (fbpar 0) and with B_par (fbpar 1). ky_gs2 = sqrt(2) ky_gx.
"""

import sys
from pathlib import Path
import cases

ELEC = """&species_parameters_2
 z = -1.0
 mass = 2.7e-4
 dens = 1.0
 temp = 1.0
 tprim = 2.49
 fprim = 0.8
 uprim = 0.0
 vnewk = 0.0
 type = "electron"
/
&dist_fn_species_knobs_2
 fexpr = 0.48
 bakdif = 0.05
/
"""
RUNGS = {
    "r2": dict(ntheta=32, nperiod=2, negrid=12, ngauss=6, delt=0.02),
    "r3": dict(ntheta=48, nperiod=3, negrid=16, ngauss=8, delt=0.01),
}
cases.GS2_RUNGS.update(
    {
        "e2": RUNGS["r2"],
        "e3": RUNGS["r3"],
        "r5": dict(ntheta=64, nperiod=4, negrid=32, ngauss=12, delt=0.025),
        "r6": dict(ntheta=64, nperiod=4, negrid=48, ngauss=16, delt=0.025),
    }
)


def em_text(c, beta, fapar, fbpar):
    t = cases.gs2_input(c)
    for a, b in (
        (" nspec = 1\n", " nspec = 2\n"),
        (" beta = 0.0\n", f" beta = {beta}\n"),
        (" fapar = 0.0\n", f" fapar = {fapar}\n"),
        (" fbpar = 0.0\n", f" fbpar = {fbpar}\n"),
        ("&init_g_knobs", ELEC + "&init_g_knobs"),
    ):
        assert t.count(a) == 1, a
        t = t.replace(a, b)
    return t


root = Path(sys.argv[2])
out = []
for fam, beta, fapar, fbpar in (
    ("KE", 1e-5, 1.0, 0.0),
    ("KBM", 0.015, 1.0, 0.0),
    ("KBMB", 0.015, 1.0, 1.0),
):
    for rung in ("e2", "e3"):
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
            (d / (c["name"] + ".in")).write_text(em_text(c, beta, fapar, fbpar))
            out.append("gs2 " + c["name"] + " 14400")
for rung in ("r5", "r6"):
    c = dict(
        code="gs2",
        geom="S",
        phys="a",
        rung=rung,
        ky_gx=0.55,
        ky_code=cases.SQRT2 * 0.55,
        t_max=1200.0,
        name=f"gs2_S_ky0.55_{rung}",
    )
    d = root / "gs2" / c["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / (c["name"] + ".in")).write_text(cases.gs2_input(c))
    out.append("gs2 " + c["name"] + " 28800")
print("\n".join(out))
