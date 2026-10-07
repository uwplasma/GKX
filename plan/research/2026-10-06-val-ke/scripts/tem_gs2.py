#!/usr/bin/env python3
"""GS2 8.2.1 inputs and fits for the Dannert & Jenko (2005) TEM case.

    python tem_gs2.py write <root>        write <root>/gs2/<case>/<case>.in and print the queue
    python tem_gs2.py fit <root>          CSV of final-window gamma/omega in GKX and c_s/R units

Case definition: tools/comparison/fixtures/parity/tem_dannert_jenko_2005.toml.
GS2 units: Lref = R, reference species = ions, vref = sqrt(2 T_i/m_i). So
aky = sqrt(2) ky_gkx = sqrt(2/3) ky rho_s, gamma_gkx = sqrt(2) gamma_gs2,
gamma [c_s/R] = sqrt(3) gamma_gkx.  beta (GS2, ion reference) = beta_e/3.
"""

import glob
import math
import sys
from pathlib import Path

S2, S3 = math.sqrt(2.0), math.sqrt(3.0)
KY_RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
RUNGS = {
    "t1": dict(ntheta=32, nperiod=2, negrid=12, ngauss=6, delt=0.01),
    "t2": dict(ntheta=48, nperiod=3, negrid=16, ngauss=8, delt=0.005),
}
T_MAX = 100.0
BETA_E = 1.0e-3


def gs2_input(name, ky_rhos, r, beta_e=BETA_E):
    nstep = int(round(T_MAX / r["delt"]))
    sp = ""
    for i, (z, m, t, tp, kind) in enumerate(
        [(1.0, 1.0, 1.0, 0.0, "ion"), (-1.0, 1.0 / 1836.0, 3.0, 6.0, "electron")], 1
    ):
        sp += f"""&species_parameters_{i}
 z = {z}
 mass = {m:.12e}
 dens = 1.0
 temp = {t}
 tprim = {tp}
 fprim = 3.0
 uprim = 0.0
 vnewk = 0.0
 type = "{kind}"
/
&dist_fn_species_knobs_{i}
 fexpr = 0.48
 bakdif = 0.05
/
"""
    return f"""! {name}: Dannert & Jenko, Phys. Plasmas 12, 072309 (2005) nominal TEM
&kt_grids_knobs
 grid_option = "single"
/
&kt_grids_single_parameters
 aky = {S2 * ky_rhos / S3:.10f}
 theta0 = 0.0
/
&theta_grid_parameters
 ntheta = {r["ntheta"]}
 nperiod = {r["nperiod"]}
 eps = 0.16
 epsl = 2.0
 shat = 0.8
 pk = {2.0 / 1.4:.10f}
 shift = 0.0
/
&theta_grid_knobs
 equilibrium_option = "s-alpha"
/
&theta_grid_salpha_knobs
 model_option = "default"
/
&le_grids_knobs
 ngauss = {r["ngauss"]}
 negrid = {r["negrid"]}
/
&dist_fn_knobs
 gridfac = 1.0
 omprimfac = 1.0
 boundary_option = "linked"
 adiabatic_option = "iphi00=2"
 g_exb = 0.0
 nonad_zero = .true.
/
&fields_knobs
 field_option = "local"
/
&layouts_knobs
 layout = "lexys"
/
&knobs
 beta = {beta_e / 3.0:.10e}
 zeff = 1.0
 wstar_units = .false.
 fphi = 1.0
 fapar = 1.0
 fbpar = 0.0
 delt = {r["delt"]}
 nstep = {nstep}
/
&reinit_knobs
 delt_adj = 2.0
 delt_minimum = 1.0e-06
/
&collisions_knobs
 collision_model = "none"
/
&nonlinear_terms_knobs
 nonlinear_mode = "off"
/
&species_knobs
 nspec = 2
/
{sp}&init_g_knobs
 chop_side = .false.
 phiinit = 0.001
 ginit_option = "noise"
 constant_random_flag = .true.
/
&gs2_diagnostics_knobs
 write_ascii = .false.
 print_line = .false.
 write_omega = .true.
 write_final_fields = .true.
 nsave = -1
 nwrite = {max(1, nstep // 600)}
 navg = 10
 omegatol = -0.001
 omegatinst = 500.0
 save_for_restart = .false.
/
"""


def cases():
    for rung in RUNGS:
        for k in KY_RHOS:
            yield f"TEM_ky{k:.2f}_{rung}", k, RUNGS[rung], BETA_E
    yield "TEM_ky0.30_t1_beta2", 0.3, RUNGS["t1"], 2.0 * BETA_E


def fit(root):
    import numpy as np
    from netCDF4 import Dataset

    print(
        "case,ky_rhos,t_end,gamma_gs2,omega_gs2,drift,gamma_gkx,omega_gkx,gamma_csR,omega_csR"
    )
    for d in sorted(glob.glob(str(Path(root) / "gs2" / "TEM_*"))):
        nc = sorted(Path(d).glob("*.out.nc"))
        if not nc or not (Path(d) / "DONE").exists():
            continue
        ds = Dataset(nc[0])
        t = np.asarray(ds.variables["t"][:], float)
        om = np.asarray(ds.variables["omega"][:], float)
        o, g = om[:, 0, -1, 0], om[:, 0, -1, 1]
        ok = np.isfinite(o) & np.isfinite(g)
        t, o, g = t[ok], o[ok], g[ok]
        w1, w0 = t >= 0.8 * t[-1], (t >= 0.6 * t[-1]) & (t < 0.8 * t[-1])
        g1, o1, g0 = g[w1].mean(), o[w1].mean(), g[w0].mean()
        k = float(Path(d).name.split("_")[1][2:])
        print(
            f"{Path(d).name},{k},{t[-1]:.1f},{g1:.6f},{o1:.6f},{(g1 - g0) / g1:+.2e},"
            f"{S2 * g1:.6f},{S2 * o1:.6f},{S2 * S3 * g1:.6f},{S2 * S3 * o1:.6f}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "write":
        for name, k, r, b in cases():
            d = Path(sys.argv[2]) / "gs2" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.in").write_text(gs2_input(name, k, r, b))
            print(f"gs2 {name} 14400")
    else:
        fit(sys.argv[2])
