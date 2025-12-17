# spectraxgk/_diagnostics.py
import jax
import jax.numpy as jnp
from ._model import solve_phi_from_quasineutrality_boltzmann_e

__all__ = ["diagnostics"]

def diagnostics(output):
    """
    Post-run diagnostics (not part of the time integrator).

    Computes phi_k(t) and simple “free-energy-like” proxies.
    Uses vmap instead of Python loops.
    """
    Gk_t = output["Gk"]  # (T,Nl,Nh,Ny,Nx,Nz)

    phi_t = jax.vmap(lambda Gk: solve_phi_from_quasineutrality_boltzmann_e(Gk, output))(Gk_t)
    # phi_t: (T,Ny,Nx,Nz)

    Wg = 0.5 * jnp.sum(jnp.abs(Gk_t) ** 2, axis=(1, 2, 3, 4, 5))
    tau_e = output["tau_e"]
    Wphi = 0.5 * jnp.sum(jnp.abs(phi_t) ** 2, axis=(1, 2, 3)) * (1.0 + 1.0 / tau_e)
    Wtot = Wg + Wphi

    output.update({
        "Phi_k": phi_t,
        "W_g": Wg,
        "W_phi": Wphi,
        "W_total": Wtot,
    })
