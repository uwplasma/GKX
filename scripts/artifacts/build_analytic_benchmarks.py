"""Run the analytic-benchmark cases and write docs/_static/analytic_benchmarks.json.

    JAX_ENABLE_X64=true python scripts/artifacts/build_analytic_benchmarks.py <case> ...

Float64 is required: in float32 the Miller geometry rejects its own theta grid
at nperiod >= 5 ("theta does not match the sampled geometry grid").

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
from scipy.integrate import solve_ivp
from scipy.special import wofz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/_static/analytic_benchmarks.json"
ZONAL_DECK = ROOT / "benchmarks/cases/miller_zonal_response.toml"
EM_DECK = ROOT / "examples/06_electromagnetic/case_full.toml"
R_OVER_A = 2.77778
M_E = 1.0 / 1836.15  # hydrogen


# ---- closed-form references (numpy/scipy only)


def rosenbluth_hinton(q: float, eps: float) -> float:
    """Rosenbluth & Hinton, PRL 80, 724 (1998): 1/(1 + 1.6 q^2/sqrt(eps))."""
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))


def xiao_catto(
    q: float, eps: float, kappa: float = 1.0, delta: float = 0.0, shift: float = 0.0
) -> float:
    """Xiao & Catto, PoP 13, 082307 (2006), Eqs. (38)-(39)."""
    s = (
        3.27
        + np.sqrt(eps)
        + 0.722 * eps
        - 1.443 * delta
        - 2.945 * shift / eps
        + (0.692 * kappa**2 - 0.722) / q**2 * eps
    ) / (1.0 + kappa**2)
    return 1.0 / (1.0 + s * q**2 / np.sqrt(eps))


def _zfun(z: complex) -> complex:
    return 1j * np.sqrt(np.pi) * wofz(z)


def sw_gam_formula(
    q: float, tau_e: float = 1.0, krai: float = 0.0
) -> tuple[float, float]:
    """Sugama & Watanabe, JPP 72, 825 (2006), Eqs. (2.9)-(2.10).

    Returns (omega_G, gamma) in v_ti/R0 (GKX units); tau_e = T_e/T_i and
    krai = k_r rho_i (GKX rho_i = sqrt(T_i/m_i)/Omega_i).
    """
    w = np.sqrt(7 + 4 * tau_e) / 2
    w *= np.sqrt(
        1 + 2 * (23 + 16 * tau_e + 4 * tau_e**2) / (q**2 * (7 + 4 * tau_e) ** 2)
    )
    wh, kq2 = q * w, 2 * (krai * q) ** 2
    corr = 1 + 2 * (23 / 4 + 4 * tau_e + tau_e**2) / (q**2 * (7 / 2 + 2 * tau_e) ** 2)
    br = np.exp(-(wh**2)) * (wh**4 + (1 + 2 * tau_e) * wh**2) + 0.25 * kq2 * np.exp(
        -(wh**2) / 4
    ) * (wh**6 / 64 + (1 + 3 * tau_e / 8) * (wh**4 / 8 + 3 * wh**2 / 4))
    return float(np.sqrt(2) * w), float(-np.sqrt(2 * np.pi) / 2 * q * br / corr)


def _sw_inverse_k(wh: complex, q: float, tau_e: float, krai: float) -> complex:
    """SW Eq. (2.7); wh = R0 q omega / sqrt(2 T_i/m_i). Z_r is read as Re Z."""
    z = _zfun(wh)
    a = 2 * wh + (2 * wh**2 + 1) * z
    d = 1.0 / tau_e + 1 + wh * z
    main = 2 * wh**3 + 3 * wh + (2 * wh**4 + 2 * wh**2 + 1) * z - wh / 2 * a**2 / d
    wr = wh.real
    zr = _zfun(wr)
    ar, dr = 2 * wr + (2 * wr**2 + 1) * zr, 1.0 / tau_e + 1 + wr * zr.real
    orbit = wr**6 / 64 + (wr**4 / 8 + 3 * wr**2 / 4 + 3 + 6 / wr**2) * (
        1 - 3 * wr / 16 * ar / dr
    )
    main += 1j * np.sqrt(np.pi) / 2 * 2 * (krai * q) ** 2 * np.exp(-(wr**2) / 4) * orbit
    return -1j * wh - 1j * q**2 / 2 * main


def sw_gam_root(q: float, tau_e: float = 1.0, krai: float = 0.0) -> tuple[float, float]:
    """Root of SW Eq. (2.7), (omega_G, gamma) in v_ti/R0 (GKX units)."""
    w, g = (x / np.sqrt(2) for x in sw_gam_formula(q, tau_e, krai))
    x0, x1 = q * complex(w, g), q * complex(1.01 * w, g)
    f0, f1 = _sw_inverse_k(x0, q, tau_e, krai), _sw_inverse_k(x1, q, tau_e, krai)
    for _ in range(100):
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, _sw_inverse_k(x1, q, tau_e, krai)
        if abs(x1 - x0) < 1e-12:
            break
    return float(np.sqrt(2) * x1.real / q), float(np.sqrt(2) * x1.imag / q)


def _ballooning_crosses(s: float, a: float, tmax: float) -> bool:
    """Even solution of CHT Eq. (10) has a zero on (0, tmax] (Newcomb: unstable)."""

    def rhs(t: float, y: np.ndarray) -> list[float]:
        h = s * t - a * np.sin(t)
        return [y[1] / (1 + h * h), -a * (np.cos(t) + h * np.sin(t)) * y[0]]

    def event(t: float, y: np.ndarray) -> float:
        return y[0]

    event.terminal = True  # type: ignore[attr-defined]
    sol = solve_ivp(
        rhs, [0.0, tmax], [1.0, 0.0], events=event, rtol=1e-10, atol=1e-13, max_step=0.5
    )
    return sol.t_events[0].size > 0


def cht_alpha_crit(
    s: float,
    tmax: float = 400.0,
    tol: float = 1e-4,
    bracket: tuple[float, float] | None = None,
) -> float:
    """Ideal s-alpha first-stability boundary, Connor-Hastie-Taylor PRL 40, 396.

    The truncated domain overestimates by O(1/tmax): 400 is within 2e-3 of the
    converged values in REPORT.md section 2.6 for 0.4 <= s <= 1.5. `bracket`
    (stable, unstable) skips the coarse scan.
    """
    if bracket is None:
        grid = np.linspace(0.05, 2.0, 40)
        i = next(k for k, a in enumerate(grid) if _ballooning_crosses(s, a, tmax))
        bracket = (grid[i - 1], grid[i])
    lo, hi = bracket
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _ballooning_crosses(s, mid, tmax) else (mid, hi)
    return float(0.5 * (lo + hi))


def pkj_beta_mhd(s_hat: float, q: float, rln: float, rlti: float, rlte: float) -> float:
    """Pueschel-Kammerer-Jenko PoP 15, 102310 (2008): 0.6 s/(q^2 sum R/L)."""
    return 0.6 * s_hat / (q**2 * (2 * rln + rlti + rlte))


def fit_zonal_response(
    t: np.ndarray, phi: np.ndarray, tmin: float = 2.0
) -> tuple[float, float, float]:
    """Fit phi_inf + A cos(omega t + p) exp(gamma t) over t >= tmin.

    Returns (residual, omega_G, gamma) in the units of ``t``. Fitting the GAM
    and the residual together removes the bias of a tail mean over a few
    weakly damped GAM periods.
    """
    from scipy.optimize import curve_fit

    t, phi = np.asarray(t, float), np.asarray(phi, float)
    keep = t >= tmin
    spectrum = np.abs(np.fft.rfft(phi[keep] - phi[keep].mean()))
    omega0 = (
        2 * np.pi * np.fft.rfftfreq(keep.sum(), t[1] - t[0])[int(np.argmax(spectrum))]
    )

    def model(
        x: np.ndarray, res: float, amp: float, om: float, gam: float, ph: float
    ) -> np.ndarray:
        return res + amp * np.cos(om * x + ph) * np.exp(gam * (x - tmin))

    p0 = [float(phi[keep][-len(phi[keep]) // 3 :].mean()), 0.3, omega0, -0.05, 0.0]
    popt, _ = curve_fit(model, t[keep], phi[keep], p0=p0, maxfev=20000)
    return float(popt[0]), float(abs(popt[2])), float(popt[3])


# ---- GKX runs


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
                "dt": 2.5e-4,
                "nperiod": 6,
                "Nm": 32,
            }
            for geom in ("miller", "s-alpha")
            for rlt in (35.0, 15.0)
            for ky in (0.05, 0.1, 0.2)
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
                "t_max": 100.0,
                "nperiod": 6,
            }
            for b in (0.008, 0.010, 0.012, 0.014, 0.016)
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
