"""Run the analytic-benchmark cases and write docs/_static/analytic_benchmarks.json.

    python scripts/artifacts/build_analytic_benchmarks.py <case> [<case> ...]

Cases: zonal (Rosenbluth-Hinton / Xiao-Catto residual and Sugama-Watanabe GAM),
pkj (Pueschel-Kammerer-Jenko CBC KBM beta scan, ky = 0.2), az (strongly driven
KBM, Aleynikova-Zocco / Tang-Connor-Hastie), cht (ky -> 0 KBM onset against the
Connor-Hastie-Taylor ideal boundary). Each case merges its records into the JSON,
so cases can run as separate jobs. CPU minutes to hours per case; the
specification and the reference values are in
plan/research/2026-09-analytic-benchmarks/REPORT.md.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/_static/analytic_benchmarks.json"
ZONAL_DECK = ROOT / "benchmarks/cases/miller_zonal_response.toml"
EM_DECK = ROOT / "examples/06_electromagnetic/case_full.toml"
R_OVER_A = 2.77778
M_E = 1.0 / 1836.15  # hydrogen


def zonal_run(
    q: float,
    eps: float,
    kx: float,
    *,
    model: str = "s-alpha",
    kappa: float = 1.0,
    Nm: int = 128,
    Nz: int = 32,
    t_max_R0: float = 22.0,
    dt: float = 2.5e-3,
    hyper: bool = False,
) -> dict:
    """ky = 0, Boltzmann electrons with the zonal correction, rhoc = 0.5 a.

    Returns the z-averaged zonal potential over its t = 0 value, t in R0/v_ti.
    `hyper` switches on the GX |k_z| Hermite hypercollisions, which absorb the
    free-streaming recurrence of a low-Nm run.
    """
    from gkx.runtime import run_runtime_linear
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, _ = load_runtime_from_toml(ZONAL_DECK)
    r0 = 0.5 / eps
    geo = replace(
        cfg.geometry, model=model, q=q, s_hat=0.8, epsilon=eps, R0=r0, rhoc=0.5
    )
    if model == "miller":
        geo = replace(
            geo, R_geo=r0, shift=0.0, akappa=kappa, akappri=0.0, tri=0.0, tripri=0.0
        )
    cfg = replace(cfg, geometry=geo, grid=replace(cfg.grid, Nz=Nz, Lx=2 * np.pi / kx))
    if hyper:
        cfg = replace(
            cfg,
            physics=replace(cfg.physics, hypercollisions=True),
            terms=replace(cfg.terms, hypercollisions=1.0),
            collisions=replace(cfg.collisions, hypercollisions_kz=1.0),
        )
    stride = max(1, int(round(0.05 / dt)))
    res = run_runtime_linear(
        cfg,
        ky_target=0.0,
        kx_target=kx,
        Nl=4,
        Nm=Nm,
        solver="time",
        dt=dt,
        steps=stride * int(np.ceil(t_max_R0 * r0 / dt / stride)),
        sample_stride=stride,
        fit_signal="phi",
        require_positive=False,
        min_points=4,
    )
    field = np.asarray(res.field_history)[:, 0]
    k = int(np.argmax(np.abs(field[0]).max(axis=1)))
    trace = np.real(field[:, k].mean(axis=1))
    return {
        "t": (np.asarray(res.t) / r0).round(5).tolist(),
        "phi": (trace / trace[0]).round(6).tolist(),
    }


def em_run(
    ky: float,
    beta: float,
    tprim: float,
    *,
    fprim: float = 0.8,
    geom: str = "s-alpha",
    consistent: bool = False,
    bpar: bool = False,
    q: float = 1.4,
    s_hat: float = 0.786,
    eps: float = 0.18,
    Nm: int = 16,
    Nl: int = 4,
    nperiod: int = 3,
    t_max: float = 60.0,
    dt: float = 1e-3,
) -> dict:
    """Linear EM CBC-like point, kinetic hydrogen electrons, lengths in a (R0 = 2.78 a).

    `consistent` sets the s-alpha alpha (or Miller betaprim) from beta and the
    gradients; otherwise alpha = 0 as in Pueschel-Kammerer-Jenko.
    """
    from gkx.runtime import run_runtime_linear
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, _ = load_runtime_from_toml(EM_DECK)
    species = tuple(
        replace(s, tprim=tprim, fprim=fprim, mass=M_E if s.charge < 0 else 1.0)
        for s in cfg.species
    )
    drive = 2 * (tprim + fprim)
    geo = replace(cfg.geometry, q=q, s_hat=s_hat, epsilon=eps, R0=R_OVER_A)
    if geom == "miller":
        geo = replace(
            geo,
            model="miller",
            R_geo=R_OVER_A,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=-beta * drive if consistent else 0.0,
        )
    else:
        geo = replace(geo, alpha=q**2 * beta * drive * R_OVER_A if consistent else 0.0)
    cfg = replace(
        cfg,
        species=species,
        geometry=geo,
        grid=replace(
            cfg.grid, nperiod=nperiod, ntheta=32, Nz=32 * (2 * nperiod - 1), y0=1.0 / ky
        ),
        physics=replace(cfg.physics, beta=beta, use_bpar=bpar),
        terms=replace(cfg.terms, bpar=float(bpar)),
        time=replace(cfg.time, damp_ends_rate=100.0, t_max=t_max, dt=dt),
    )
    stride = max(1, int(round(0.2 / dt)))
    res = run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=Nl,
        Nm=Nm,
        solver="time",
        dt=dt,
        steps=stride * int(np.ceil(t_max / dt / stride)),
        sample_stride=stride,
        require_positive=False,
    )
    return {
        "ky": float(res.ky),
        "beta": beta,
        "tprim": tprim,
        "fprim": fprim,
        "geom": geom,
        "consistent": consistent,
        "bpar": bpar,
        "Nm": Nm,
        "nperiod": nperiod,
        "gamma": float(res.gamma),
        "omega": float(res.omega),
        "settled": bool(res.fit_settled),
    }


def _zonal_job(args: tuple) -> tuple[str, dict]:
    name, a, kw = args
    return name, {"args": a, **kw, **zonal_run(*a, **kw)}


def _em_job(kw: dict) -> dict:
    return em_run(**kw)


ZONAL_CASES = {
    "rh_q1.4": ((1.4, 0.18, 0.05), {}),
    "rh_q1.4_kx0.025": ((1.4, 0.18, 0.025), {}),
    "rh_q1.0": ((1.0, 0.18, 0.05), {}),
    "rh_q2.0": ((2.0, 0.18, 0.05), {}),
    "rh_miller_q1.4": ((1.4, 0.18, 0.05), {"model": "miller"}),
    "xc_kappa1": ((2.0, 0.2, 0.05), {"model": "miller", "kappa": 1.0}),
    "xc_kappa3": ((2.0, 0.2, 0.05), {"model": "miller", "kappa": 3.0}),
    "sw_fig1": ((1.5, 0.1, 0.131), {}),
}


def em_cases(case: str) -> list[dict]:
    if case == "pkj":
        return [
            {"ky": 0.2, "beta": b, "tprim": 6.89 / R_OVER_A, "bpar": bp}
            for bp in (False, True)
            for b in (0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018)
        ]
    if case == "az":
        return [
            {
                "ky": ky,
                "beta": 0.015,
                "tprim": rlt / R_OVER_A,
                "geom": geom,
                "consistent": True,
                "bpar": True,
                "t_max": 20.0,
                "dt": 5e-4,
            }
            for geom in ("miller", "s-alpha")
            for rlt in (35.0, 15.0)
            for ky in (0.05, 0.1, 0.2, 0.3)
        ]
    if case == "cht":
        return [
            {
                "ky": 0.05,
                "beta": b,
                "tprim": 6.89 / R_OVER_A,
                "geom": "miller",
                "consistent": True,
                "bpar": True,
                "t_max": 150.0,
                "nperiod": 5,
            }
            for b in (0.006, 0.008, 0.010, 0.012, 0.014)
        ]
    raise SystemExit(f"unknown case {case!r}")


def main(argv: list[str]) -> int:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for case in argv:
            if case == "zonal":
                jobs = [(n, a, kw) for n, (a, kw) in ZONAL_CASES.items()]
                data["zonal"] = dict(pool.map(_zonal_job, jobs))
            else:
                data[case] = list(pool.map(_em_job, em_cases(case)))
            OUT.write_text(json.dumps(data, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
