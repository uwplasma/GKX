# tests/test_simulation_save_modes.py
import numpy as np
import jax.numpy as jnp

from spectraxgk._simulation import simulation


def test_save_diagnostics_contains_expected_keys():
    out = simulation(
        input_parameters=dict(t_max=0.2, nu=0.0, pert_amp=1e-3),
        Nx=7, Ny=7, Nz=5, Nl=3, Nh=6,
        timesteps=10,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
        progress=False,
    )
    for k in ["time", "W_g", "W_phi", "W_total", "D_coll", "max_abs_G", "max_abs_phi"]:
        assert k in out
    assert out["time"].shape[0] == out["W_total"].shape[0]


def test_save_final_contains_final_state_only():
    out = simulation(
        input_parameters=dict(t_max=0.1, nu=0.0, pert_amp=1e-3),
        Nx=7, Ny=7, Nz=5, Nl=3, Nh=6,
        timesteps=5,
        dt=1e-2,
        adaptive_time_step=False,
        save="final",
        progress=False,
    )
    assert "Gk_final" in out
    assert out["Gk_final"].dtype == jnp.complex128
    assert out["Gk_final"].shape == (3, 6, 7, 7, 5)


def test_save_full_contains_time_series_state():
    out = simulation(
        input_parameters=dict(t_max=0.1, nu=0.0, pert_amp=1e-3),
        Nx=5, Ny=5, Nz=3, Nl=2, Nh=4,
        timesteps=6,
        dt=1e-2,
        adaptive_time_step=False,
        save="full",
        progress=False,
    )
    assert "Gk" in out
    assert out["Gk"].shape[0] == out["time"].shape[0]
    assert out["Gk"].shape[1:] == (2, 4, 5, 5, 3)


def test_save_diagnostics_with_probe_adds_probe_key():
    out = simulation(
        input_parameters=dict(t_max=0.2, nu=0.0, pert_amp=1e-3),
        Nx=7, Ny=7, Nz=5, Nl=3, Nh=6,
        timesteps=10,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
        probe=dict(ky=7//2, kx=7//2+1, kz=5//2, lmax=2, mmax=3),
        progress=False,
    )
    assert "probe_G_lm" in out
    assert out["probe_G_lm"].shape == (out["time"].shape[0], 2, 3)
