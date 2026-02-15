import pytest
import jax
import jax.numpy as jnp

from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, ConstantStepSize

from spectraxgk._initialization_multispecies import initialize_simulation_parameters_multispecies
from spectraxgk._simulation_multispecies import _pack_complex_flat, _unpack_complex_flat, vector_field_real


def test_parallel_drift_matches_analytic():
    # Minimal sizes; keep a nonzero kz mode
    Nx, Ny, Nz = 1, 1, 17
    Nl, Nh = 1, 1

    U = 1.3
    species = [
        dict(name="ion", q=+1.0, T=1.0, n0=1.0, rho=1.0, vth=0.0, nu=0.0, Upar=U),
    ]

    p = initialize_simulation_parameters_multispecies(
        dict(
            species=species,
            enable_streaming=True,
            enable_nonlinear=False,
            enable_collisions=False,
            enforce_reality=False,
            nx0=0, ny0=0, nz0=1,
            pert_amp=1e-3,
            t_max=0.4,
        ),
        Nx=Nx, Ny=Ny, Nz=Nz, Nl=Nl, Nh=Nh, timesteps=20, dt=0.01
    )

    Ns = int(p["Ns"])
    Ncomplex = int(Ns * Nl * Nh * Ny * Nx * Nz)

    y0 = _pack_complex_flat(p["Gk_0"])
    t0, t1, dt0 = 0.0, float(p["t_max"]), 0.01

    term = ODETerm(lambda t, y, args: vector_field_real(t, y, args, Ns, Nl, Nh, Ny, Nx, Nz, Ncomplex))

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=t0, t1=t1,
        dt0=dt0,
        y0=y0,
        args=p,
        saveat=SaveAt(ts=jnp.array([t1])),
        stepsize_controller=ConstantStepSize(),
        max_steps=100000,
    )

    yT = sol.ys[0]
    G0 = p["Gk_0"]
    GT = _unpack_complex_flat(yT, (Ns, Nl, Nh, Ny, Nx, Nz), Ncomplex)

    # Identify the kz=+1 mode in fftshifted ordering
    kz_grid = p["kz_grid"][0, 0, :]  # (Nz,)
    # For Nz odd, kz=+1*2π/Lz corresponds to index Nz//2 + 1
    iz = Nz // 2 + 1
    kz = float(kz_grid[iz])

    phase = jnp.exp(-1j * kz * U * t1)
    expected = G0.at[0, 0, 0, 0, 0, iz].get() * phase

    got = GT.at[0, 0, 0, 0, 0, iz].get()

    # Tight accuracy check (this should be extremely accurate)
    assert jnp.allclose(got, expected, rtol=5e-5, atol=5e-6)
