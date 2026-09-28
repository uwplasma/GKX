"""Small numpy/scipy checks for the analytic-benchmark report (no JAX, no GKX)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import wofz


def crosses_zero(s, a, tmax=60.0):
    """CHT s-alpha marginal ballooning eq.; Newcomb: zero crossing => unstable.
    d/dt[(1+h^2) F'] + a (cos t + h sin t) F = 0,  h = s t - a sin t."""
    def rhs(t, y):
        h = s * t - a * np.sin(t)
        return [y[1] / (1 + h * h), -a * (np.cos(t) + h * np.sin(t)) * y[0]]
    ev = lambda t, y: y[0]
    ev.terminal = True
    sol = solve_ivp(rhs, [0, tmax], [1.0, 0.0], events=ev, rtol=1e-10, atol=1e-12)
    return sol.t_events[0].size > 0


def alpha_crit(s):
    a = np.linspace(0.02, 2.0, 100)
    flag = [crosses_zero(s, x) for x in a]
    i = int(np.argmax(flag))
    lo, hi = a[i - 1], a[i]
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (lo, m) if crosses_zero(s, m) else (m, hi)
    return 0.5 * (lo + hi)


def Z(z):
    return 1j * np.sqrt(np.pi) * wofz(z)


def rosenbluth_hinton(q, eps):
    return 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))


def gam_sugama_watanabe(q, tau=1.0):
    """Leading-order GAM freq in v_ti/R (v_ti=sqrt(2T/m)): sqrt(7/4+tau) (large q)."""
    return np.sqrt(7.0 / 4.0 + tau) * np.sqrt(1 + (46 + 32 * tau + 8 * tau**2) / (7 + 4 * tau) ** 2 / (2 * q**2))


if __name__ == "__main__":
    print("CHT s-alpha first-stability alpha_crit(s):")
    for s in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
        print(f"  s={s:4.1f}  alpha_crit={alpha_crit(s):.4f}")
    print("Z(0) =", Z(0.0), " (expect i*sqrt(pi)=1.7725i)")
    print("RH residual q=1.4, eps=0.18:", rosenbluth_hinton(1.4, 0.18))
    print("GAM omega R/v_ti tau=1, q=1.4:", gam_sugama_watanabe(1.4))
    # KBM diamagnetic modification: gamma^2 = gamma_MHD^2 - omega_*pi^2/4
    g_mhd, wpi = 0.4, 0.5
    w = 0.5 * wpi + 1j * np.sqrt(max(g_mhd**2 - wpi**2 / 4, 0))
    print("diamag-MHD root omega =", w, " residual", w * (w - wpi) + g_mhd**2)
