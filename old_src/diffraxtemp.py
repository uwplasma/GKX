# -------------------- Diffrax-based FH Vlasov–Poisson (no eigendecomp) --------------------
# Solves:  dot c = (-i H(k) - C) c   with H = streaming(tridiag) + field(rank-1),  C = LB collisions
# Uses a real state y = [Re(c); Im(c)] to keep Diffrax happy.

# Enable 64-bit BEFORE importing jax.numpy
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from scipy.optimize import curve_fit


def fit_decaying_cosine_abs(ts, Ek, tfit):
    """
    Fit |Ek(t)| ≈ A * exp(gamma * t) * |cos(omega * t + phi)|
    on t in [tfit[0], tfit[1]]. Returns gamma, omega, A, phi, se_gamma, se_omega.
    """
    ts = np.asarray(ts)
    Ek = np.asarray(Ek)
    mask = (ts >= tfit[0]) & (ts <= tfit[1])
    tm = ts[mask]
    amp = np.abs(Ek[mask]) + 1e-300

    def model(t, A, gamma, omega, phi):
        return A * np.exp(gamma * t) * np.abs(np.cos(omega * t + phi))

    # crude initial guesses
    logA_guess = np.log(np.max(amp) + 1e-300)
    gamma_guess = (np.log(amp[-1]) - np.log(amp[0])) / max(tm[-1] - tm[0], 1e-9)
    omega_guess = 2*np.pi / max((tm[-1]-tm[0]) / 5.0, 1e-3)  # ~5 cycles
    phi_guess = 0.0
    p0 = [np.exp(logA_guess), gamma_guess, omega_guess, phi_guess]

    popt, pcov = curve_fit(model, tm, amp, p0=p0, maxfev=20000)
    A_fit, gamma_fit, omega_fit, phi_fit = popt

    # standard errors from covariance diag (guard NaNs/Infs)
    se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
    se_A, se_gamma, se_omega, se_phi = [float(x) if np.isfinite(x) else np.nan for x in se]

    return gamma_fit, omega_fit, A_fit, phi_fit, se_gamma, se_omega

# -------------------- Operators --------------------
def hermite_streaming_tridiag(k: float, N: int) -> jnp.ndarray:
    """[H_stream]_n,n+1 = [H_stream]_n+1,n = k * sqrt((n+1)/2)."""
    n = jnp.arange(N - 1, dtype=jnp.float64)
    off = k * jnp.sqrt((n + 1) / 2.0) * jnp.sqrt(2)
    H = jnp.zeros((N, N), dtype=jnp.float64)
    H = H.at[jnp.arange(N - 1), jnp.arange(1, N)].set(off)
    H = H.at[jnp.arange(1, N), jnp.arange(N - 1)].set(off)
    return H  # real symmetric

def hermite_field_rank1(k: float, N: int) -> jnp.ndarray:
    """Field closure with your normalization: E_k = i c0 / k  ->  [H_field]_{1,0}=[0,1]=1/k."""
    H = jnp.zeros((N, N), dtype=jnp.float64)
    def nz(H):
        H = H.at[1, 0].set(1.0 / k)
        return H
    return jax.lax.cond(jnp.abs(k) > 0.0, nz, lambda H: H, H)

def build_Hk(k: float, N: int) -> jnp.ndarray:
    return hermite_streaming_tridiag(k, N) + hermite_field_rank1(k, N)

def build_collision_matrix(N: int, nu0: float, hyper_p: int = 0) -> jnp.ndarray:
    """
    Lenard–Bernstein with optional 'hyper' factor:
      C = diag( nu(n)*n ),   nu(n) = nu0 * (n/(N-1))^p    (p=0 -> textbook LB)
    """
    n = jnp.arange(N, dtype=jnp.float64)
    if hyper_p > 0:
        scale = (n / jnp.maximum(N - 1, 1)) ** hyper_p
        return jnp.diag(nu0 * scale * n)  # LB factor * n
    else:
        return jnp.diag(nu0 * n)

def build_generator_A(H: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """A = -i H - C (not used directly in Diffrax; here for reference)."""
    return (-1j) * H - C

# -------------------- Real-valued RHS for Diffrax --------------------
# Let c = x + i y.  dot c = (-i H - C) c  =>
#   dx/dt =  H y - C x
#   dy/dt = -H x - C y
def rhs_real(t, y, args):
    H, C = args  # both (N,N) real
    N = H.shape[0]
    x = y[:N]
    z = y[N:]  # imag part
    dx = H @ z - C @ x
    dz = - H @ x - C @ z
    return jnp.concatenate([dx, dz])

# -------------------- Driver + diagnostics --------------------
def run_and_plot_diffrax(
    N=256, k=0.5,                      # physics
    nu0=0.0, hyper_p=0,                # collisions; set nu0>0 for LB/hyper-LB
    tmax=60.0, nt=1200, tfit=(10.,50.),# time grid and fit window
    seed_c1=True,                      # small c1 to seed phase mixing
    rtol=1e-7, atol=1e-10, dt0=1e-2    # solver controls
):

    # Build operators
    H = build_Hk(k, N)                           # real symmetric
    C = build_collision_matrix(N, nu0, hyper_p)  # real diagonal

    # Initial condition in real form
    c0 = jnp.zeros((N,), dtype=jnp.complex128)
    c0 = c0.at[0].set(1e-2)
    if seed_c1:
        c0 = c0.at[1].set(1e-3)
    y0 = jnp.concatenate([jnp.real(c0), jnp.imag(c0)])  # (2N,)

    # Time grid and solver
    ts = jnp.linspace(0.0, tmax, nt, dtype=jnp.float64)
    controller = PIDController(rtol=rtol, atol=atol, jump_ts=ts)
    term = ODETerm(rhs_real)
    sol = diffeqsolve(
        term, Tsit5(),
        t0=0.0, t1=tmax, dt0=dt0,
        y0=y0, args=(H, C),
        stepsize_controller=controller,
        saveat=SaveAt(ts=ts),
        max_steps=2_000_000
    )
    Y = sol.ys  # (nt, 2N)
    Xr = Y[:, :N]      # Re c_n(t)
    Xi = Y[:, N:]      # Im c_n(t)
    C_t = Xr + 1j * Xi # (nt, N) complex, but time-major; transpose for (N, nt)
    C_t = C_t.T        # Now (N, nt), like your previous code

    # Observable and fit
    c0_t = C_t[0, :]
    Ek_t = 1j * c0_t / k
    ts_np = np.asarray(ts)

    gamma_hat, omega_hat, A_hat, phi_hat, se_gamma, se_omega = fit_decaying_cosine_abs(ts_np, Ek_t, tfit)
    print(f"[Diffrax] N={N}, k={k}, ν0={nu0}, p={hyper_p}  ->  "
        f"γ = {gamma_hat:.6f} ± {se_gamma:.6f},  ω = {omega_hat:.6f} ± {se_omega:.6f}")

    # Build fitted |cos|-based model over the whole domain (no mask)
    eps = 1e-300
    Ek_fit_abs = A_hat * np.exp(gamma_hat * ts_np) * np.abs(np.cos(omega_hat * ts_np + phi_hat))

    # Plot on log|E_k|
    plt.figure()
    plt.plot(ts_np, np.log(np.abs(Ek_t) + eps), 'k', label='log |E_k| (sim)')
    plt.plot(ts_np, np.log(Ek_fit_abs + eps), 'r--', lw=2, label='log |A e^{γt} cos| fit')
    plt.axvline(tfit[0], ls="--", color='gray'); plt.axvline(tfit[1], ls="--", color='gray')
    plt.xlabel("t"); plt.ylabel("log |E_k(t)|")
    plt.title(f"log |E_k| with |cos| fit:  γ={gamma_hat:.3f},  ω={omega_hat:.3f}")
    plt.legend(); plt.tight_layout()

    # 2) Imshow of |c_n(t)|
    C_np = np.asarray(C_t)
    plt.figure()
    plt.imshow(np.log(np.abs(C_np)+1e-300), aspect="auto", origin="lower", vmin=-14,
               extent=[ts_np[0], ts_np[-1], 0, N], interpolation="nearest")
    plt.colorbar(label="log |c_n(t)|")
    plt.xlabel("t"); plt.ylabel("Hermite n")
    plt.title("Phase mixing: |c_n(t)| (Diffrax)")
    plt.tight_layout()

    # 3) Free-energy split (collisionless conserved; with LB decays)
    E  = 1j * C_np[0, :] / k
    W_field = 0.5 * np.abs(E)**2                           # = 0.5 * |c0|^2 / k^2
    W_kin   = 0.5 * (np.abs(C_np[1:, :])**2).sum(axis=0)   # 1/2 sum_{n>=1} |c_n|^2
    W_total = W_field + W_kin

    plt.figure()
    plt.plot(ts_np, W_field, 'b-',  label="Field energy")
    plt.plot(ts_np, W_kin,   'r-',  label="Kinetic energy")
    plt.plot(ts_np, W_total, 'k--', label="Total energy")
    plt.xlabel("t"); plt.ylabel("Energy")
    ttl = "Free energy (collisionless ~ flat; LB -> decay)"
    if nu0 > 0: ttl += f"  (ν0={nu0}, p={hyper_p})"
    plt.title(ttl); plt.legend(); plt.tight_layout()

    plt.show()
    return ts, C_np, Ek_t, gamma_hat

# -------------------- Run example --------------------
if __name__ == "__main__":
    ts, C_np, Ek_t, gamma_hat = run_and_plot_diffrax(
        N=256, k=1.1, nu0=1e-5, hyper_p=2, tmax=20.0, nt=1200,
        tfit=(10.0, 20.0), seed_c1=False, rtol=1e-7, atol=1e-10, dt0=1e-2
    )
