import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, Tsit5

from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._model_multispecies import rhs_gk_multispecies, enforce_conjugate_symmetry_fftshifted
from spectraxgk._simulation_multispecies import _pack_complex_flat, _unpack_complex_flat, vector_field_real


def _project(G, p):
    # same manifold as RHS: dealias + optional reality
    G = jnp.where(p["mask23"][None, None, None, ...], G, 0.0 + 0.0j)
    return enforce_conjugate_symmetry_fftshifted(G, p) if bool(p.get("enforce_reality", False)) else G


def test_gradB_zero_profile_is_noop():
    # If B_eps=0 then gradlnB(z)=0; enabling gradB should change nothing.
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.2, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=0.2, vth=1.0, nu=0.0, Upar=0.0),
    ]
    base = dict(
        species=species,
        enable_streaming=True,
        enable_nonlinear=False,
        enable_collisions=False,
        enforce_reality=True,
        lambda_D=0.2,
        B_eps=0.0,
        B_mode=1,
    )

    p0 = initialize_simulation_parameters_multispecies({**base, "enable_gradB_parallel": False},
                                                       Nx=5, Ny=5, Nz=17, Nl=3, Nh=6, timesteps=10, dt=1e-2)
    p1 = initialize_simulation_parameters_multispecies({**base, "enable_gradB_parallel": True},
                                                       Nx=5, Ny=5, Nz=17, Nl=3, Nh=6, timesteps=10, dt=1e-2)

    key = jax.random.PRNGKey(0)
    G = (jax.random.normal(key, (p0["Ns"], p0["Nl"], p0["Nh"], p0["Ny"], p0["Nx"], p0["Nz"]))
         + 1j * jax.random.normal(key, (p0["Ns"], p0["Nl"], p0["Nh"], p0["Ny"], p0["Nx"], p0["Nz"]))).astype(p0["Gk_0"].dtype)
    G = _project(G, p0)

    dG0, _ = rhs_gk_multispecies(G, p0, Nh=p0["Nh"], Nl=p0["Nl"])
    dG1, _ = rhs_gk_multispecies(G, p1, Nh=p1["Nh"], Nl=p1["Nl"])

    assert jnp.allclose(dG0, dG1, rtol=1e-8, atol=1e-10)


def test_gradB_short_run_no_nans():
    # Nonlinear/collisions off: gradB+streaming should remain finite.
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.2, vth=1.0, nu=0.0, Upar=0.0),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=0.2, vth=1.0, nu=0.0, Upar=0.0),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(
            species=species,
            enable_streaming=True,
            enable_gradB_parallel=True,
            enable_nonlinear=False,
            enable_collisions=False,
            enforce_reality=False,
            lambda_D=0.2,
            B_eps=0.15,
            B_mode=1,
            nx0=0, ny0=0, nz0=1,
            pert_amp=1e-4,
            t_max=0.2,
        ),
        Nx=1, Ny=1, Nz=33, Nl=2, Nh=6, timesteps=10, dt=0.01
    )

    Ns, Nl, Nh, Ny, Nx, Nz = int(p["Ns"]), int(p["Nl"]), int(p["Nh"]), int(p["Ny"]), int(p["Nx"]), int(p["Nz"])
    Ncomplex = Ns * Nl * Nh * Ny * Nx * Nz

    y0 = _pack_complex_flat(p["Gk_0"])
    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex))

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=0.0, t1=float(p["t_max"]),
        dt0=0.01,
        y0=y0,
        args=p,
        saveat=SaveAt(ts=jnp.array([float(p["t_max"])])),
        stepsize_controller=ConstantStepSize(),
        max_steps=100000,
    )

    yT = sol.ys[0]
    GT = _unpack_complex_flat(yT, (Ns, Nl, Nh, Ny, Nx, Nz), Ncomplex)

    assert jnp.all(jnp.isfinite(jnp.real(GT)))
    assert jnp.all(jnp.isfinite(jnp.imag(GT)))


def test_streaming_energy_directional_derivative_small():
    # Physics test: d/dt (0.5||H||^2) ~ 0 along the RHS in the linear collisionless case.
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=0.3, vth=1.0, nu=0.0, Upar=0.2),
        dict(name="e",   q=-1.0, T=1.0, n0=1.0, rho=0.3, vth=1.0, nu=0.0, Upar=-0.2),
    ]
    p = initialize_simulation_parameters_multispecies(
        dict(
            species=species,
            enable_streaming=True,
            enable_nonlinear=False,
            enable_collisions=False,
            enforce_reality=True,
            lambda_D=0.2,  # avoids den_qn degeneracy except at k=0 (gauge-fixed)
        ),
        Nx=5, Ny=5, Nz=17, Nl=3, Nh=8, timesteps=10, dt=1e-2
    )

    key = jax.random.PRNGKey(123)
    G = (jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))
         + 1j * jax.random.normal(key, (p["Ns"], p["Nl"], p["Nh"], p["Ny"], p["Nx"], p["Nz"]))).astype(p["Gk_0"].dtype)
    G = _project(G, p)

    dG, _ = rhs_gk_multispecies(G, p, Nh=p["Nh"], Nl=p["Nl"])

    from spectraxgk._model_multispecies import cheap_diagnostics_multispecies

    eps = jnp.asarray(1e-6 if (G.real.dtype == jnp.float64) else 1e-4, dtype=G.real.dtype)

    Wplus  = cheap_diagnostics_multispecies(_project(G + eps * dG, p), p)["W_free"]
    Wminus = cheap_diagnostics_multispecies(_project(G - eps * dG, p), p)["W_free"]
    dW = (Wplus - Wminus) / (2*eps)

    assert float(jnp.abs(dW)) < 1e-3
