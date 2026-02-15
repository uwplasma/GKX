# tests/conftest.py
import os

# (Best-effort) keep tests deterministic and lightweight
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# Must be set before importing spectraxgk modules that read this env var.
os.environ.setdefault("SPECTRAX_X64", "1")

import pytest
import numpy as np

import matplotlib
matplotlib.use("Agg")  # no GUI in CI

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from spectraxgk._initialization import initialize_simulation_parameters
from spectraxgk._simulation import simulation


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def tiny_sizes():
    # very small sizes to keep runtime low
    return dict(Nx=7, Ny=7, Nz=5, Nl=3, Nh=6)


@pytest.fixture
def tiny_params(tiny_sizes):
    p = initialize_simulation_parameters(
        user_parameters=dict(
            t_max=1.0,
            dt=1e-2,
            nu=0.0,
            pert_amp=1e-3,
            enable_streaming=True,
            enable_nonlinear=True,
            enable_collisions=True,
            enforce_reality=True,
        ),
        Nx=tiny_sizes["Nx"],
        Ny=tiny_sizes["Ny"],
        Nz=tiny_sizes["Nz"],
        Nl=tiny_sizes["Nl"],
        Nh=tiny_sizes["Nh"],
        timesteps=10,
        dt=1e-2,
    )
    return p


@pytest.fixture
def tiny_state(tiny_params, tiny_sizes):
    # deterministic pseudo-random complex state, respecting typical shapes
    Nl, Nh = tiny_sizes["Nl"], tiny_sizes["Nh"]
    Ny, Nx, Nz = tiny_sizes["Ny"], tiny_sizes["Nx"], tiny_sizes["Nz"]

    key = jax.random.PRNGKey(0)
    re = jax.random.normal(key, (Nl, Nh, Ny, Nx, Nz), dtype=jnp.float64)
    im = jax.random.normal(jax.random.fold_in(key, 1), (Nl, Nh, Ny, Nx, Nz), dtype=jnp.float64)
    Gk = (re + 1j * im).astype(jnp.complex128)

    # de-aliasing is done inside RHS; keep as-is here
    return Gk


@pytest.fixture
def run_small_sim():
    """
    Helper to run a small simulation quickly with stable deterministic settings.
    """
    def _run(**kwargs):
        defaults = dict(
            input_parameters=dict(
                t_max=1.0,
                nu=0.0,
                pert_amp=1e-3,
                enable_streaming=True,
                enable_nonlinear=True,
                enable_collisions=True,
                enforce_reality=True,
            ),
            Nx=7, Ny=7, Nz=5, Nl=3, Nh=6,
            timesteps=20,
            dt=1e-2,
            adaptive_time_step=False,
            save="diagnostics",
            save_every=1,
            progress=False,
        )
        defaults.update(kwargs)
        return simulation(**defaults)
    return _run
