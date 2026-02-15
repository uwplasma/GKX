# tests/test_plotting.py
import numpy as np

from spectraxgk._plot import plot, plot_probe_velocity_reconstruction


def test_plot_runs_without_gui(run_small_sim):
    out = run_small_sim(
        input_parameters=dict(
            t_max=0.2, nu=0.0, pert_amp=1e-3,
            enable_streaming=True, enable_nonlinear=False, enable_collisions=False, enforce_reality=True
        ),
        timesteps=10,
        save="diagnostics",
    )
    # Should not raise
    plot(out)


def test_probe_velocity_reconstruction_runs(run_small_sim):
    out = run_small_sim(
        input_parameters=dict(
            t_max=0.2, nu=0.0, pert_amp=1e-3,
            enable_streaming=True, enable_nonlinear=False, enable_collisions=False, enforce_reality=True
        ),
        timesteps=10,
        save="diagnostics",
        probe=dict(ky=7//2, kx=7//2+1, kz=5//2, lmax=2, mmax=6),
    )
    # Should not raise
    plot_probe_velocity_reconstruction(out, l_pick=0)
