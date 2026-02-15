# tests/test_initialization.py
import jax.numpy as jnp

from spectraxgk._initialization import initialize_simulation_parameters


def test_initialize_shapes_and_keys():
    p = initialize_simulation_parameters(
        user_parameters=dict(t_max=1.0, nu=0.0, pert_amp=1e-3),
        Nx=9, Ny=7, Nz=5, Nh=8, Nl=4, timesteps=10, dt=1e-2
    )

    assert p["kx_grid"].shape == (7, 9, 5)
    assert p["Jl_grid"].shape == (4, 7, 9, 5)
    assert p["Gk_0"].shape == (4, 8, 7, 9, 5)

    # center gauge
    Ny, Nx, Nz = 7, 9, 5
    assert p["mask23"].shape == (Ny, Nx, Nz)

    # conjugate index arrays exist
    assert p["conj_y"].shape == (Ny,)
    assert p["conj_x"].shape == (Nx,)
    assert p["conj_z"].shape == (Nz,)


def test_initial_condition_conjugate_pair_real_cosine_like():
    Nx, Ny, Nz = 9, 9, 7
    Nl, Nh = 3, 6
    p = initialize_simulation_parameters(
        user_parameters=dict(nx0=1, ny0=0, nz0=0, pert_amp=1e-3),
        Nx=Nx, Ny=Ny, Nz=Nz, Nh=Nh, Nl=Nl, timesteps=10, dt=1e-2
    )
    G0 = p["Gk_0"]

    ky0 = Ny // 2 + 0
    kx0 = Nx // 2 + 1
    kz0 = Nz // 2 + 0

    # conjugate indices must match helper logic
    ky1 = int(p["conj_y"][ky0])
    kx1 = int(p["conj_x"][kx0])
    kz1 = int(p["conj_z"][kz0])

    a0 = G0[0, 0, ky0, kx0, kz0]
    a1 = G0[0, 0, ky1, kx1, kz1]

    # both set to same real amplitude
    assert jnp.isclose(jnp.imag(a0), 0.0)
    assert jnp.isclose(jnp.imag(a1), 0.0)
    assert jnp.isclose(jnp.real(a0), jnp.real(a1))
