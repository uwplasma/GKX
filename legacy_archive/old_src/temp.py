# fh_slim.py
# Fourier–Hermite Landau damping with Diffrax or Eig, single fit: |E_k| ≈ A e^{γ t} |cos(ω t + φ)|
# Speed-ups: JAX-jitted operators, JAX-aided initial guesses for (γ, ω, A, φ), optional bootstrap SEs.
# Usage:
#   python fh_slim.py --backend diffrax
#   python fh_slim.py --backend eig

import argparse
import time
from functools import partial

# ---- JAX setup ----
import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Optional Diffrax (for --backend diffrax)
try:
    from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
    HAS_DIFFRAX = True
except Exception:
    HAS_DIFFRAX = False

# SciPy only for the nonlinear |cos| fit
from scipy.optimize import curve_fit


# ============================ Operators (JAX, jittable) ============================
@partial(jax.jit, static_argnames=['N'])
def hermite_streaming_tridiag(k: float, N: int) -> jnp.ndarray:
    # Your working normalization: off_n = k * sqrt((n+1)/2) * sqrt(2) = k * sqrt(n+1)
    n = jnp.arange(N - 1, dtype=jnp.float64)
    off = k * jnp.sqrt(n + 1.0)
    H = jnp.zeros((N, N), dtype=jnp.float64)
    H = H.at[jnp.arange(N - 1), jnp.arange(1, N)].set(off)
    H = H.at[jnp.arange(1, N), jnp.arange(N - 1)].set(off)
    return H

@partial(jax.jit, static_argnames=['N'])
def hermite_field_one_sided(k: float, N: int) -> jnp.ndarray:
    # PDE-faithful: only dc1/dt gets -i*(1/k)*c0  => H[1,0]=1/k only
    H = jnp.zeros((N, N), dtype=jnp.float64)
    return jax.lax.cond(jnp.abs(k) > 0.0,
                        lambda HH: HH.at[1, 0].set(1.0 / k),
                        lambda HH: HH, H)

@partial(jax.jit, static_argnames=['N'])
def build_H(k: float, N: int) -> jnp.ndarray:
    return hermite_streaming_tridiag(k, N) + hermite_field_one_sided(k, N)

@partial(jax.jit, static_argnames=['N'])
def build_collision_matrix(N: int, nu0: float, hyper_p: int = 0) -> jnp.ndarray:
    # Zero collisions at n=0,1,2; LB/hyper-LB for n>=3
    n = jnp.arange(N, dtype=jnp.float64)
    mask = (n >= 3).astype(jnp.float64)
    denom = jnp.maximum(N - 1.0, 1.0)
    p = jnp.asarray(hyper_p, dtype=jnp.float64)
    scale = jnp.power(n / denom, p)
    nu_n = nu0 * scale * mask
    return jnp.diag(nu_n * n)

# ============================ Backends ============================
def rhs_real(t, y, args):
    H, C = args
    N = H.shape[0]
    x = y[:N]; z = y[N:]
    dx = H @ z - C @ x
    dz = - H @ x - C @ z
    return jnp.concatenate([dx, dz])

def solve_diffrax(H, C, c0, tmax, nt, rtol, atol, dt0):
    if not HAS_DIFFRAX:
        raise RuntimeError("Diffrax not installed; use --backend eig.")
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    y0 = jnp.concatenate([jnp.real(c0), jnp.imag(c0)])
    controller = PIDController(rtol=rtol, atol=atol, jump_ts=ts)
    sol = diffeqsolve(ODETerm(rhs_real), Tsit5(),
                      t0=0.0, t1=tmax, dt0=dt0,
                      y0=y0, args=(H, C),
                      stepsize_controller=controller,
                      saveat=SaveAt(ts=ts), max_steps=2_000_000)
    Y = sol.ys
    N = H.shape[0]
    C_t = (Y[:, :N] + 1j * Y[:, N:]).T
    return np.asarray(ts), np.asarray(C_t)

@jax.jit
def eig_cache(A):
    w, V = jnp.linalg.eig(A)
    Vinv = jnp.linalg.inv(V)
    return w, V, Vinv

@jax.jit
def evolve_cached(w, V, Vinv, c0, ts):
    alpha = Vinv @ c0
    phases = jnp.exp(w[:, None] * ts[None, :])
    return V @ (phases * alpha[:, None])

def solve_eig(H, C, c0, tmax, nt):
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    A = (-1j) * H - C
    w, V, Vinv = eig_cache(A)
    C_t = evolve_cached(w, V, Vinv, c0, ts)
    return np.asarray(ts), np.asarray(C_t)

# ============================ Single fit: |cos|-model ============================
def _initial_guess_from_complex_linear(ts, Ek, tfit):
    """JAX-aided quick estimates (γ, ω, A, φ) to initialize the nonlinear |cos| fit."""
    ts_j = jnp.asarray(ts); Ek_j = jnp.asarray(Ek)
    m = (ts_j >= tfit[0]) & (ts_j <= tfit[1])
    tm = ts_j[m]; Em = Ek_j[m]
    amp = jnp.abs(Em) + 1e-300
    # unwrap phase (simple, stable implementation)
    d = jnp.diff(jnp.angle(Em))
    two_pi = 2.0 * jnp.pi
    dmod = (d + jnp.pi) % two_pi - jnp.pi
    phi = jnp.concatenate([jnp.angle(Em[:1]), jnp.angle(Em[:1]) + jnp.cumsum(dmod)])

    X = jnp.stack([jnp.ones_like(tm), tm], axis=1)
    # log-amp fit
    XtX = X.T @ X
    beta_re = jnp.linalg.solve(XtX, X.T @ jnp.log(amp))
    logA, gamma = beta_re
    # phase fit
    beta_im = jnp.linalg.solve(XtX, X.T @ phi)
    phi0, omega = beta_im
    A = jnp.exp(logA)
    # adapt φ to |cos| model (phase modulo π is fine); align to first sample
    phi_abs = float(phi0 % jnp.pi)
    return float(gamma), float(omega), float(A), float(phi_abs)

def fit_decaying_cosine_abs(ts, Ek, tfit, maxfev=40000, bootstrap=0, seed=0):
    """
    Fit |E_k(t)| ≈ A * exp(gamma * t) * |cos(omega * t + phi)|
    Returns: (gamma, omega, A, phi, se_gamma, se_omega)
    """
    ts = np.asarray(ts)
    Ek = np.asarray(Ek)
    mask = (ts >= tfit[0]) & (ts <= tfit[1])
    tm = ts[mask]
    ym = np.abs(Ek[mask]) + 1e-300
    if tm.size < 5:
        raise ValueError("Need at least ~5 points in the fit window.")

    def model(t, A, gamma, omega, phi):
        return A * np.exp(gamma * t) * np.abs(np.cos(omega * t + phi))

    # smarter initials (from complex linear fit)
    g0, w0, A0, phi0 = _initial_guess_from_complex_linear(ts, Ek, tfit)
    # optional FFT refinement of ω (abs(cos) has dominant 2ω component)
    y_detr = ym / (np.exp(g0 * (tm - tm[0])) + 1e-300)
    y_detr = y_detr / (y_detr.max() + 1e-300)
    f = np.fft.rfftfreq(len(tm), d=(tm[1]-tm[0]))
    Y = np.abs(np.fft.rfft(y_detr))
    pk = np.argmax(Y[1:]) + 1  # skip DC
    w_fft = np.pi * (2 * np.pi * f[pk]) / (2 * np.pi)  # ≈ π * f_peak
    omega0 = float(0.5 * (abs(w0) + abs(w_fft))) if np.isfinite(w_fft) else abs(w0)
    p0 = [max(A0, 1e-12), g0, max(omega0, 1e-6), float(phi0)]

    popt, pcov = curve_fit(model, tm, ym, p0=p0, maxfev=maxfev)
    A_fit, gamma_fit, omega_fit, phi_fit = popt

    def cov_to_se(pcov_):
        se = np.sqrt(np.maximum(np.diag(pcov_), 0.0))
        return [float(x) if np.isfinite(x) else np.nan for x in se]

    se_A, se_gamma, se_omega, se_phi = cov_to_se(pcov)

    # optional bootstrap for robust SEs
    if bootstrap and bootstrap > 1:
        rng = np.random.default_rng(seed)
        B = int(bootstrap)
        boots = []
        n = len(tm)
        for _ in range(B):
            idx = rng.integers(0, n, n)
            try:
                p, _ = curve_fit(model, tm[idx], ym[idx], p0=popt, maxfev=maxfev)
                boots.append(p)
            except Exception:
                pass
        if len(boots) > 1:
            boots = np.asarray(boots)
            _, gsd, osd, _ = boots.std(axis=0, ddof=1)
            se_gamma, se_omega = float(gsd), float(osd)

    return float(gamma_fit), float(omega_fit), float(A_fit), float(phi_fit), float(se_gamma), float(se_omega)

# ============================ Driver + plots ============================
def run_and_plot(
    backend="diffrax",
    N=256, k=0.5,
    nu0=0.0, hyper_p=0,
    tmax=30.0, nt=1200,
    tfit=(5.0, 15.0),
    seed_c1=True,
    rtol=1e-7, atol=1e-10, dt0=1e-2,
    vmin_log_cn=-14.0,
    bootstrap=0
):
    # (optional) warmup to avoid counting the first-time JIT in timings
    _ = build_H(k, N).block_until_ready()
    _ = build_collision_matrix(N, 0.0, 0).block_until_ready()

    t0 = time.perf_counter()
    H = build_H(k, N)
    C = build_collision_matrix(N, nu0, hyper_p)
    c0 = jnp.zeros((N,), dtype=jnp.complex128).at[0].set(1e-2)
    if seed_c1:
        c0 = c0.at[1].set(1e-3)
    t1 = time.perf_counter()

    if backend == "diffrax":
        ts_np, C_np = solve_diffrax(H, C, c0, tmax, nt, rtol, atol, dt0)
    elif backend == "eig":
        ts_np, C_np = solve_eig(H, C, c0, tmax, nt)
    else:
        raise ValueError("backend must be 'diffrax' or 'eig'")
    t2 = time.perf_counter()

    c0_t = C_np[0, :]
    Ek_t = 1j * c0_t / k
    gamma, omega, A, phi, se_gamma, se_omega = fit_decaying_cosine_abs(
        ts_np, Ek_t, tfit, bootstrap=bootstrap
    )
    t3 = time.perf_counter()

    print(f"[{backend}|abs] build={t1-t0:.3f}s  solve={t2-t1:.3f}s  fit={t3-t2:.3f}s")
    print(f"  γ = {gamma:.6f} ± {se_gamma:.6f}    ω = {omega:.6f} ± {se_omega:.6f}")

    # Overlays
    eps = 1e-300
    fit_abs = A * np.exp(gamma * ts_np) * np.abs(np.cos(omega * ts_np + phi))

    # Plot 1: log|E_k|
    plt.figure()
    plt.plot(ts_np, np.log(np.abs(Ek_t) + eps), 'k', label='log |E_k| (sim)')
    plt.plot(ts_np, np.log(fit_abs + eps), 'r--', lw=2, label='log |A e^{γt} cos| fit')
    plt.axvline(tfit[0], ls="--", c="gray"); plt.axvline(tfit[1], ls="--", c="gray")
    plt.xlabel("t"); plt.ylabel("log |E_k(t)|")
    plt.title(f"log |E_k| fit ({backend}):  γ={gamma:.3f}, ω={omega:.3f}")
    plt.legend(); plt.tight_layout()

    # Plot 2: Hermite imshow (clipped)
    plt.figure()
    plt.imshow(np.log(np.abs(C_np) + eps), aspect="auto", origin="lower",
               extent=[ts_np[0], ts_np[-1], 0, C_np.shape[0]],
               interpolation="nearest", vmin=vmin_log_cn)
    plt.colorbar(label="log |c_n(t)|")
    plt.xlabel("t"); plt.ylabel("Hermite n")
    plt.title(f"Phase mixing: |c_n(t)| ({backend})")
    plt.tight_layout()

    # Plot 3: Energy split
    E_t = 1j * C_np[0, :] / k
    W_field = 0.5 * np.abs(E_t)**2
    W_kin   = 0.5 * (C_np[0, :] + C_np[2, :])#.sum(axis=0)
    print(W_kin.shape, W_kin[0])
    W_total = W_field + W_kin
    plt.figure()
    plt.plot(ts_np, W_field, 'b-',  label="Field")
    plt.plot(ts_np, W_kin,   'r-',  label="Kinetic")
    plt.plot(ts_np, W_total, 'k--', label="Total")
    plt.yscale("log")
    plt.xlabel("t"); plt.ylabel("Energy")
    plt.title(f"Free energy ({backend}; ν0={nu0}, p={hyper_p})")
    plt.legend(); plt.tight_layout()

    plt.show()
    return ts_np, C_np, Ek_t, gamma

# ============================ CLI ============================
def main():
    p = argparse.ArgumentParser(description="FH Landau damping with Diffrax/Eig and |cos|-model fit.")
    p.add_argument("--backend", type=str, default="diffrax", choices=["diffrax", "eig"])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--k", type=float, default=0.5)
    p.add_argument("--nu0", type=float, default=0.0)
    p.add_argument("--hyper_p", type=int, default=0)
    p.add_argument("--tmax", type=float, default=30.0)
    p.add_argument("--nt", type=int, default=1200)
    p.add_argument("--tfit0", type=float, default=5.0)
    p.add_argument("--tfit1", type=float, default=15.0)
    p.add_argument("--seed_c1", action="store_true")
    p.add_argument("--rtol", type=float, default=1e-7)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--dt0", type=float, default=1e-2)
    p.add_argument("--vmin_log_cn", type=float, default=-14.0)
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap rounds for SEs (0=off)")
    args = p.parse_args()

    if args.backend == "diffrax" and not HAS_DIFFRAX:
        raise SystemExit("Diffrax not installed. Try `pip install diffrax` or use --backend eig.")

    run_and_plot(
        backend=args.backend,
        N=args.N, k=args.k,
        nu0=args.nu0, hyper_p=args.hyper_p,
        tmax=args.tmax, nt=args.nt,
        tfit=(args.tfit0, args.tfit1),
        seed_c1=args.seed_c1,
        rtol=args.rtol, atol=args.atol, dt0=args.dt0,
        vmin_log_cn=args.vmin_log_cn,
        bootstrap=args.bootstrap
    )

if __name__ == "__main__":
    main()
