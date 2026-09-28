"""Reference numbers for the analytic-benchmark specification (numpy/scipy only).

Every value printed here is quoted in REPORT.md. Run: python benchmarks_check.py
Equation numbers refer to the original papers (see REPORT.md for citations).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import wofz, exp1


def Z(z):
    """Plasma dispersion function, analytically continued (Z = i sqrt(pi) w(z))."""
    return 1j * np.sqrt(np.pi) * wofz(z)


# ---------------------------------------------------------------- 1. CHT 1978
def _cht_crosses(s, a, tmax, g2=0.0, c=0.0):
    """s-alpha ballooning ODE, CHT Eq. (10) / Hastie-Hesketh Eq. (2.1) with
    omega^2/omega_A^2 = -g2 and Hastie-Hesketh Eq. (2.6) compression term
    c = (beta_i q^2/2)(7 + 4 tau) (omega_* -> 0):  (f F')' + a g F - g2 f F - c g^2 F = 0,
    f = 1 + h^2, g = cos t + h sin t, h = s t - a sin t.
    Even mode F(0)=1, F'(0)=0; returns True if F has a zero on (0, tmax]."""

    def rhs(t, y):
        h = s * t - a * np.sin(t)
        f = 1.0 + h * h
        g = np.cos(t) + h * np.sin(t)
        return [y[1] / f, -(a * g - g2 * f - c * g * g) * y[0]]

    def ev(t, y):
        return y[0]

    ev.terminal = True
    sol = solve_ivp(
        rhs, [0.0, tmax], [1.0, 0.0], events=ev, rtol=1e-10, atol=1e-13, max_step=0.5
    )
    return sol.t_events[0].size > 0


def cht_alpha_crit(s, tmax, lo=0.05, hi=2.0, tol=1e-7, scan=True):
    """First-stability boundary on a finite theta domain (Newcomb criterion).
    The truncated-domain value approaches the infinite-domain one from above."""
    if scan:
        a_grid = np.linspace(lo, hi, 80)
        flag = [_cht_crosses(s, x, tmax) for x in a_grid]
        i = int(np.argmax(flag))
        lo, hi = a_grid[i - 1], a_grid[i]
    while _cht_crosses(s, lo, tmax):  # widen a bracket supplied by the caller
        lo -= 0.01
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if _cht_crosses(s, m, tmax) else (m, hi)
    return 0.5 * (lo + hi)


def cht_alpha_crit_converged(s, tol=1e-4):
    """Double the theta cut-off until alpha_crit moves by < tol. The error is
    O(1/theta_max) (each doubling halves the change), so the last change also
    bounds the remaining error; 2 v_n - v_(n-1) is the Richardson estimate."""
    tmax = 50.0
    prev = cht_alpha_crit(s, tmax)
    while True:
        tmax *= 2.0
        val = cht_alpha_crit(s, tmax, lo=prev - 0.01, hi=prev + 1e-5, scan=False)
        if abs(val - prev) < tol:
            return val, 2 * val - prev, tmax, abs(val - prev)
        prev = val


# ------------------------------------------------ 2. Hastie-Hesketh MHD limit
def hh_gamma_mhd(s, a, c=0.0, tmax=60.0):
    """gamma/omega_A of the ka_i -> 0 limit of Hastie-Hesketh: Eq. (2.1)
    (incompressible, c=0) or Eq. (2.6) with omega_*/omega -> 0 (c > 0, adiabatic
    index (7/4+tau)/(1+tau)). Largest g with a decaying even eigenfunction."""

    def crosses(g):
        return _cht_crosses(s, a, tmax, g2=g * g, c=c)

    lo, hi = 1e-4, 2.0
    if not crosses(lo):
        return 0.0
    while hi - lo > 1e-6:
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if crosses(m) else (lo, m)
    return 0.5 * (lo + hi)


# --------------------------------------------------------- 3. RH / Xiao-Catto
def rosenbluth_hinton(q, eps):
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))


def xiao_catto(q, eps, kappa=1.0, delta=0.0, Delta=0.0):
    """Xiao & Catto 2006, Eqs. (38)-(39)."""
    S = (
        3.27
        + np.sqrt(eps)
        + 0.722 * eps
        - 1.443 * delta
        - 2.945 * Delta / eps
        + (0.692 * kappa**2 - 0.722) / q**2 * eps
    ) / (1.0 + kappa**2)
    return 1.0 / (1.0 + S * q**2 / np.sqrt(eps))


# ------------------------------------------------------ 4. Sugama-Watanabe GAM
def sw_gam_formula(q, tau_e=1.0, krai=0.0):
    """SW 2006 Eqs. (2.9)-(2.10). Returns (omega_G, gamma) in v_Ti/R0 with
    v_Ti = sqrt(2 T_i/m_i); krai = k_r a_i, a_i = sqrt(T_i/m_i)/Omega_i."""
    wG = (
        np.sqrt(7 + 4 * tau_e)
        / 2
        * np.sqrt(
            1 + 2 * (23 + 16 * tau_e + 4 * tau_e**2) / (q**2 * (7 + 4 * tau_e) ** 2)
        )
    )
    wh = q * wG  # hat omega_G = R0 q omega_G / v_Ti
    kq2 = (np.sqrt(2) * krai * q) ** 2  # (k_r v_Ti q / Omega_i)^2
    corr = 1 + 2 * (23 / 4 + 4 * tau_e + tau_e**2) / (q**2 * (7 / 2 + 2 * tau_e) ** 2)
    br = np.exp(-(wh**2)) * (wh**4 + (1 + 2 * tau_e) * wh**2) + 0.25 * kq2 * np.exp(
        -(wh**2) / 4
    ) * (wh**6 / 64 + (1 + 3 * tau_e / 8) * (wh**4 / 8 + 3 * wh**2 / 4))
    gam = -np.sqrt(np.pi) / 2 * q**2 / q * br / corr  # (v_Ti/(R0 q)) q^2 = q v_Ti/R0
    return wG, gam


def sw_inverse_K(wh, q, tau_e=1.0, krai=0.0):
    """SW Eq. (2.7): 1/K(omega), wh = R0 q omega / v_Ti (complex).
    Bracket nesting re-checked against the printed page (p. 827)."""
    ti_te = 1.0 / tau_e
    z = Z(wh)
    A = 2 * wh + (2 * wh**2 + 1) * z
    D = ti_te + 1 + wh * z
    main = 2 * wh**3 + 3 * wh + (2 * wh**4 + 2 * wh**2 + 1) * z - wh / 2 * A**2 / D
    fow = 0.0
    if krai > 0:
        wr = wh.real
        zr = Z(wr)
        Ar = 2 * wr + (2 * wr**2 + 1) * zr
        Dr = ti_te + 1 + wr * zr.real  # Z_r: real part of Z (subscript r in paper)
        kq2 = (np.sqrt(2) * krai * q) ** 2
        fow = (
            1j
            * np.sqrt(np.pi)
            / 2
            * kq2
            * np.exp(-(wr**2) / 4)
            * (
                wr**6 / 64
                + (wr**4 / 8 + 3 * wr**2 / 4 + 3 + 6 / wr**2)
                * (1 - 3 * wr / 16 * Ar / Dr)
            )
        )
    return -1j * wh - 1j * q**2 / 2 * (main + fow)


def sw_gam_root(q, tau_e=1.0, krai=0.0):
    """Newton/secant root of 1/K = 0 started from the (2.9)-(2.10) estimate.
    Returns (omega_G, gamma) in v_Ti/R0."""
    wG, gam = sw_gam_formula(q, tau_e, krai)
    x0, x1 = q * complex(wG, gam), q * complex(wG * 1.01, gam)
    f0, f1 = sw_inverse_K(x0, q, tau_e, krai), sw_inverse_K(x1, q, tau_e, krai)
    for _ in range(100):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1 = x1, f1, x2
        f1 = sw_inverse_K(x1, q, tau_e, krai)
        if abs(x1 - x0) < 1e-12:
            break
    return x1.real / q, x1.imag / q


# ----------------------------------------------------------- 5. ITG thresholds
def romanelli_fit(eps_n):
    """Romanelli 1989 Eq. (27) (s=0.5, q=1.5, tau=1, b_i=0.1)."""
    return 1.0 if eps_n < 0.2 else 1.0 + 2.5 * (eps_n - 0.2)


def romanelli_fluid(eps_n, tau=1.0):
    """Eq. (22): 1 + eta_i = eps_n [4 + (tau/2)(1 - 1/(2 eps_n))^2]."""
    return eps_n * (4 + tau / 2 * (1 - 1 / (2 * eps_n)) ** 2) - 1


def romanelli_gradB_D(Om, eps_n, eta, tau=1.0):
    """Eq. (19) grad-B local kinetic dispersion D(z), z = Om tau / (2 eps_n)."""
    z = Om * tau / (2 * eps_n)
    return (
        1
        + 1 / tau
        - eta / (2 * eps_n)
        + (z * (1 - eta / (2 * eps_n)) + (1 - eta) / (2 * eps_n)) * np.exp(z) * exp1(z)
    )


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("2. Hastie-Hesketh ka_i->0 growth, q=sqrt2, tau=1 (Fig. 2 / Fig. 4)")
    for s, a, bi in ((0.6, 0.8, 0.01), (0.4, 0.54, 0.00675)):
        c = bi * 2.0 / 2 * (7 + 4)
        print(
            f"   S={s}, alpha={a}, beta_i={bi}: Eq.(2.1) {hh_gamma_mhd(s, a):.4f};"
            f" Eq.(2.6) omega_*->0 {hh_gamma_mhd(s, a, c):.4f}",
            flush=True,
        )
    q, beta_ref = np.sqrt(2.0), 0.01
    print(
        f"   GX beta_ref={beta_ref}, fprim=tprim=10 (R units), 2 species: alpha ="
        f" {q**2 * beta_ref * 2 * (10 + 10):.3f}; omega_A = {np.sqrt(2 / beta_ref) / q:.3f} v_ti/R"
    )

    print("3. Pueschel-Kammerer-Jenko 2008 CBC closed form")
    s, q, wn, wT = 0.786, 1.4, 2.22, 6.89
    bmhd = 0.6 * s / (q**2 * (2 * wn + 2 * wT))
    print(f"   beta_MHD = 0.6 s/[q^2(2wn+wTi+wTe)] = {100 * bmhd:.4f} %")
    for b in (1.14, 1.26):
        print(f"   alpha_MHD(beta={b}%) = {q**2 * b / 100 * (2 * wn + 2 * wT):.4f}")

    print(
        "4. Strongly driven KBM, omega(omega - w_pi) = -g_MHD^2 (Tang 1980 / AZ 2017 Eq. 9)"
    )
    g, wpi = 0.4, 0.5
    w = 0.5 * wpi + 1j * np.sqrt(g**2 - wpi**2 / 4)
    print(
        f"   root {w:.5f}, residual {abs(w * (w - wpi) + g**2):.1e}; Re(w)/w_pi = {w.real / wpi}"
    )
    for LT in (35, 40, 45):
        a_ = 2.7775
        print(
            f"   AZ Fig.2 R/LT={LT}: w_r = ky*(a/Ln + a/LT)/2 = ky*{(2.22 + LT) / a_ / 2:.3f}"
            f" v_ti/a;  consistent alpha = {1.4**2 * 0.015 * 2 * (2.22 + LT):.3f}"
        )

    print("5. Rosenbluth-Hinton / Xiao-Catto residuals")
    print(f"   RH  q=1.4 eps=0.18: {rosenbluth_hinton(1.4, 0.18):.4f}")
    print(f"   XC  q=1.4 eps=0.18 circular: {xiao_catto(1.4, 0.18):.4f}")
    print(f"   RH  q=1.5 eps=0.1: {rosenbluth_hinton(1.5, 0.1):.4f}")
    for k in (1.0, 1.8, 3.0):
        print(
            f"   XC Fig.1 q=2 eps=0.2 Delta=0.04 kappa={k}: {xiao_catto(2, 0.2, k, 0, 0.04):.4f}"
        )
    print(f"   XC Fig.2 kappa=1.8 delta=0.4: {xiao_catto(2, 0.2, 1.8, 0.4, 0.04):.4f}")
    print(
        f"   XC Fig.3 kappa=1.8 Delta=0 / 0.1: {xiao_catto(2, 0.2, 1.8, 0, 0):.4f} /"
        f" {xiao_catto(2, 0.2, 1.8, 0, 0.1):.4f}"
    )

    print("6. Sugama-Watanabe GAM (v_Ti=sqrt(2T/m); x sqrt2 -> GX v_ti=sqrt(T/m))")
    for q, kr in ((1.4, 0.0), (1.5, 0.131), (1.5, 0.0)):
        wG, gm = sw_gam_formula(q, 1.0, kr)
        wr, gr = sw_gam_root(q, 1.0, kr)
        print(
            f"   q={q} k_r a_i={kr}: (2.9)-(2.10) wG={wG:.4f} g={gm:.4f} v_Ti/R0"
            f" = {np.sqrt(2) * wG:.4f}, {np.sqrt(2) * gm:.4f} v_ti/R0 | root of (2.7):"
            f" {np.sqrt(2) * wr:.4f}, {np.sqrt(2) * gr:.4f} v_ti/R0"
        )

    print("7. ITG thresholds")
    for en in (0.1, 0.25, 0.5, 1.0):
        print(
            f"   Romanelli eps_n={en}: fit (27) eta_ic={romanelli_fit(en):.3f};"
            f" fluid (22) eta_ic={romanelli_fluid(en):.3f}"
        )
    print(
        f"   Romanelli (26) flat density: R/LT_c = 2(1+1/tau) = 4; with 2/3 average = {8 / 3:.3f}"
    )
    print(
        f"   BDR (32) fluid flat density: L_T/R > 4/(7+tau) -> R/LT_c = {(7 + 1) / 4:.3f};"
        " resonant Fig. 2 read L_T/R ~ 0.35 -> 2.86"
    )
    kp, en, ei, ee, ky = 0.1, 0.2, 2.5, 2.0, 0.5
    full = 2 * kp**2 / (2 * en * (2 + ei + ee) - ky**2 * (1 + ei) ** 2 / 4)
    print(
        f"   KHD (24) local beta_ic (k_par L_n=0.1, eps_n=0.2, eta_i=2.5, eta_e=2, ky=0.5):"
        f" {full:.4f}; long-wavelength form {kp**2 / (en * (2 + ei + ee)):.4f}"
    )
    print(
        "8. CHT s-alpha first-stability alpha_crit(s), theta cut-off doubled to |d|<1e-4"
    )
    for s in (0.2, 0.4, 0.6, 0.786, 0.8, 1.0, 1.5):
        a, rich, tmax, d = cht_alpha_crit_converged(s)
        print(
            f"   s={s:5.3f}  alpha_crit={a:.4f}  Richardson={rich:.4f}"
            f"  (theta_max={tmax:6.0f}, last change {d:.1e})",
            flush=True,
        )

    print(f"   Z(0) = {Z(0.0)}  (i sqrt(pi) = 1.7725i)")
