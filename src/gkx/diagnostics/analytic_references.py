"""Closed-form references for the analytic benchmarks (numpy/scipy only).

Transcribed from plan/research/2026-09-analytic-benchmarks (REPORT.md and
benchmarks_check.py). Frequencies are returned in GKX units
(v_ti = sqrt(T_i/m_i)); the papers' sqrt(2 T/m) values are converted here.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import wofz


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
