#!/usr/bin/env python3
"""Assemble stella vs GKX CBC linear ITG comparison table.

Conventions:
  stella (GS2/GX family): a_ref, vth = sqrt(2T/m), rho = vth/Omega.
  GKX (source-verified,   a_ref, cs  = sqrt(T/m),  rho_s = cs/Omega.
       operators/linear/params.py: vth=sqrt(T/m), rho=sqrt(Tm)/q)
  Mapping at fixed physical mode:
     ky_stella = sqrt(2) * ky_gkx_input
     (gamma, omega)[vth/a] = (gamma, omega)[cs/a] / sqrt(2)
"""
import re
import numpy as np

SQRT2 = np.sqrt(2.0)
BASE = "/private/tmp/claude-501/-Users-rogeriojorge-local/1e858e4f-6438-4dbd-8d3c-60502cd814ab/scratchpad/stella_vs_gkx"

# --- stella: parse final-time rows of the 6-point scan .omega file ---
stella = {}
with open(f"{BASE}/stella_scan/cbc_scan6.omega") as f:
    lines = [line for line in f if line.strip() and not line.strip().startswith("#")]
rows = [list(map(float, line.split())) for line in lines]
tmax = max(r[0] for r in rows)
for r in rows:
    if r[0] == tmax:
        t, ky, kx, reom, imom, reavg, imavg = r
        stella[round(ky, 6)] = (reavg, imavg)  # time-averaged omega, gamma

# --- GKX: parse scan log, then overlay refined longer-horizon runs ---
gkx = {}
gkx_horizon = {}
pat = re.compile(r"ky=([0-9.]+)\s+gamma=(-?[0-9.]+)\s+omega=(-?[0-9.]+)")
for fname, horizon in [
    ("gkx_miller_scan_sqrt2.log", 40),   # batch scan, t=40
    ("gkx_lowky_t80.log", 80),           # ky 0.1414, 0.2121 at t=80
    ("gkx_refine2.log", 120),            # ky 0.2828 t=80; ky 0.1414 t=120 (last wins)
]:
    try:
        with open(f"{BASE}/gkx/{fname}") as f:
            for line in f:
                m = pat.search(line)
                if m and "fit complete" not in line:
                    kyg, g, w = map(float, m.groups())
                    gkx[round(kyg, 6)] = (g, w)
                    gkx_horizon[round(kyg, 6)] = horizon
    except FileNotFoundError:
        pass
if 0.2828 in gkx_horizon and gkx_horizon[0.2828] == 120:
    gkx_horizon[0.2828] = 80  # refine2 ran this point at t=80, not t=120

# --- tracked GX/GKX mismatch table (s-alpha + hypercollisional lane) ---
gxtab = {}
with open("/Users/rogeriojorge/local/GKX/docs/_static/cyclone_mismatch_table.csv") as f:
    next(f)
    for line in f:
        v = list(map(float, line.split(",")))
        gxtab[round(v[0], 6)] = tuple(v[1:5])  # gamma_ref, omega_ref, gamma_gkx, omega_gkx

out = ["ky_stella,ky_gkx_input,gamma_stella_vth,omega_stella_vth,"
       "gamma_gkx_cs,omega_gkx_cs,gamma_gkx_vth,omega_gkx_vth,"
       "rel_gamma_gkx_vs_stella,rel_omega_gkx_vs_stella,gkx_fit_horizon,"
       "gamma_gxref_raw,omega_gxref_raw,gamma_gkxtracked_raw,omega_gkxtracked_raw"]
for ky_st in sorted(stella):
    ky_g = ky_st / SQRT2
    w_st, g_st = stella[ky_st]
    # find gkx entry nearest ky_g
    match = min(gkx, key=lambda k: abs(k - ky_g)) if gkx else None
    if match is None or abs(match - ky_g) > 5e-3:
        g_cs = w_cs = g_vth = w_vth = rg = rw = float("nan")
        horizon = 0
    else:
        g_cs, w_cs = gkx[match]
        g_vth, w_vth = g_cs / SQRT2, w_cs / SQRT2
        rg = (g_vth - g_st) / g_st
        rw = (w_vth - w_st) / w_st
        horizon = gkx_horizon.get(match, 40)
    # tracked table overlap at face-value GKX ky (their input units)
    tk = min(gxtab, key=lambda k: abs(k - ky_g))
    gx_g, gx_w, gkxt_g, gkxt_w = (gxtab[tk] if abs(tk - ky_g) < 5e-3
                                  else (float("nan"),) * 4)
    out.append(
        f"{ky_st:.3f},{ky_g:.6f},{g_st:.6f},{w_st:.6f},"
        f"{g_cs:.6f},{w_cs:.6f},{g_vth:.6f},{w_vth:.6f},"
        f"{rg:+.4f},{rw:+.4f},{horizon},"
        f"{gx_g:.6f},{gx_w:.6f},{gkxt_g:.6f},{gkxt_w:.6f}"
    )

csv = "\n".join(out) + "\n"
with open(f"{BASE}/comparison.csv", "w") as f:
    f.write(csv)
print(csv)
