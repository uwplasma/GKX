#!/usr/bin/env python3
"""Collisional Cyclone ITG nu-scan inputs for GS2 8.2.1 and stella (ky_gx = 0.3, Lref = a).

    python itg_coll_cases.py write <root>    write <root>/<code>/<case>/<case>.in, print the queue

Reuses the Q20 writers (plan/research/scripts/2026-09-14-cross-code-cyclone/cases.py),
rung r3, adiabatic electrons, and switches on each code's collision operator:
GS2 "default" (pitch-angle + energy diffusion with conserving terms), stella
"fokker-planck" and "dougherty". vnewk = vnew_ref = nu_gkx / sqrt(2) (GS2/stella
vref = sqrt(2T/m); docs/conventions.rst).
"""

import math
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "2026-09-14-cross-code-cyclone"
    ),
)
import cases as q20  # noqa: E402

NU_GKX = (0.0, 0.01, 0.03, 0.1)
KY = 0.30


def gs2(name, geom, nu):
    case = dict(
        name=name,
        code="gs2",
        geom=geom,
        phys="a",
        rung="r3",
        ky_gx=KY,
        ky_code=math.sqrt(2) * KY,
        t_max=q20.T_MAX_A[geom][KY],
    )
    text = q20.gs2_input(case)
    if nu:
        text = text.replace('collision_model = "none"', 'collision_model = "default"')
        text = text.replace("vnewk = 0.0", f"vnewk = {nu / math.sqrt(2):.10e}")
    return text


ELECTRON = """&species_parameters_2
 z = -1.0
 mass = 2.7e-4
 dens = 1.0
 temp = 1.0
 tprim = 2.49
 fprim = 0.8
 uprim = 0.0
 vnewk = {vnewk:.10e}
 type = "electron"
/
&dist_fn_species_knobs_2
 fexpr = 0.48
 bakdif = 0.05
/
"""


def gs2_kinetic_electrons(name, nu):
    """s-alpha kinetic-electron Cyclone (examples/05, beta 1e-5 with A_par)."""

    q20.GS2_RUNGS["e3"] = dict(ntheta=48, nperiod=3, negrid=16, ngauss=8, delt=0.01)
    case = dict(
        name=name,
        code="gs2",
        geom="S",
        phys="a",
        rung="e3",
        ky_gx=KY,
        ky_code=math.sqrt(2) * KY,
        t_max=100.0,
    )
    text = q20.gs2_input(case)
    vnewk_e = nu * (1.0 / 2.7e-4) ** 0.5 / math.sqrt(2)
    for a, b in (
        (" nspec = 1\n", " nspec = 2\n"),
        (" beta = 0.0\n", " beta = 1e-05\n"),
        (" fapar = 0.0\n", " fapar = 1.0\n"),
        (
            "&init_g_knobs",
            ELECTRON.format(vnewk=vnewk_e if nu else 0.0) + "&init_g_knobs",
        ),
    ):
        assert text.count(a) == 1, a
        text = text.replace(a, b)
    if nu:
        text = text.replace('collision_model = "none"', 'collision_model = "default"')
        text = text.replace(
            'vnewk = 0.0\n type = "ion"',
            f'vnewk = {nu / math.sqrt(2):.10e}\n type = "ion"',
        )
    return text


def stella(name, nu, model):
    case = dict(
        name=name,
        code="stella",
        geom="M",
        phys="a",
        rung="r3",
        ky_gx=KY,
        ky_code=math.sqrt(2) * KY,
        t_max=q20.T_MAX_A["M"][KY],
    )
    text = q20.stella_input(case)
    if nu:
        text += (
            "&dissipation_and_collisions_options\n  include_collisions = .true.\n"
            f'  collision_model = "{model}"\n  collisions_implicit = .true.\n'
            f"  vnew_ref = {nu / math.sqrt(2):.10e}\n/\n"
        )
    return text


if __name__ == "__main__":
    root = Path(sys.argv[2])
    for nu in NU_GKX:
        jobs = [
            (f"gs2_{g}_nu{nu:g}", "gs2", gs2(f"gs2_{g}_nu{nu:g}", g, nu))
            for g in ("S", "M")
        ]
        for model in ("fokker-planck", "dougherty") if nu else ("none",):
            name = f"stella_M_{model}_nu{nu:g}"
            jobs.append((name, "stella", stella(name, nu, model)))
        name = f"gs2_KE_nu{nu:g}"
        jobs.append((name, "gs2", gs2_kinetic_electrons(name, nu)))
        for name, code, text in jobs:
            d = root / code / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.in").write_text(text)
            print(code, name)
