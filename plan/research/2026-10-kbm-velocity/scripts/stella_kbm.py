#!/usr/bin/env python3
"""stella v1.0 inputs for the circular-Miller KBM (beta .015, kinetic electrons, A_par [+ B_par]).

    python stella_kbm.py <root>  -> queue lines "stella <case> <timeout>"
Built from the Q20 cases.stella_input (run from the xcode bench) with two kinetic species and
the &electromagnetic namelist. v-grid ladder at fixed z grid (nzed 48, nperiod 2): v1 (nvgrid 24,
nmu 12), v2 (48, 24). ky_stella = sqrt(2) ky_gx; delt .005 (electron streaming is implicit).
"""

import sys
from pathlib import Path

import cases

ELEC = """&species_parameters_2
  dens = 1.0
  fprim = 0.8
  mass = 2.7e-4
  temp = 1.0
  tprim = 2.49
  type = "electron"
  z = -1.0
/
&electromagnetic
  include_apar = .true.
  include_bpar = {bpar}
  beta = 0.015
/
"""
cases.STELLA_RUNGS.update(
    v1=dict(nzed=48, nperiod=2, nvgrid=24, nmu=12, delt=0.005),
    v2=dict(nzed=48, nperiod=2, nvgrid=48, nmu=24, delt=0.005),
)
root = Path(sys.argv[1])
for fam, bpar in (("KBM", ".false."), ("KBMB", ".true.")):
    for rung in ("v1", "v2"):
        name = f"{fam}_stella_M_ky0.30_{rung}"
        t = cases.stella_input(
            dict(
                code="stella",
                geom="M",
                phys="a",
                rung=rung,
                ky_gx=0.3,
                ky_code=cases.SQRT2 * 0.3,
                t_max=60.0 * cases.SQRT2,
                name=name,
            )
        )
        for a, b in (
            ("  nspec = 1\n", "  nspec = 2\n"),
            # stella v1.0 ignores &electromagnetic unless this switch is on
            (
                "  include_nonlinear = .false.\n",
                "  include_nonlinear = .false.\n  include_electromagnetic = .true.\n",
            ),
            ("&kxky_grid_option", ELEC.format(bpar=bpar) + "&kxky_grid_option"),
        ):
            assert t.count(a) == 1, a
            t = t.replace(a, b)
        d = root / "stella" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.in").write_text(t)
        print(f"stella {name} 28800")
