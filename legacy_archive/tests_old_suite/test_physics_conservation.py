# tests/test_physics_conservation.py
import numpy as np


def test_collisionless_energy_nearly_conserved(run_small_sim):
    out = run_small_sim(
        input_parameters=dict(
            t_max=0.5,
            nu=0.0,
            pert_amp=1e-3,
            enable_streaming=True,
            enable_nonlinear=True,
            enable_collisions=False,
            enforce_reality=True,
        ),
        timesteps=30,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
    )
    W = np.asarray(out["W_total"])
    rel = abs(W[-1] - W[0]) / max(abs(W[0]), 1e-30)
    assert rel < 5e-3


def test_collisional_energy_decays_and_cumulative_dissipation_increases(run_small_sim):
    out = run_small_sim(
        input_parameters=dict(
            t_max=0.5,
            nu=0.05,
            pert_amp=1e-3,
            enable_streaming=True,
            enable_nonlinear=False,
            enable_collisions=True,
            enforce_reality=True,
        ),
        timesteps=40,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
    )
    W = np.asarray(out["W_total"])
    assert W[-1] <= W[0] * (1.0 + 1e-3)

    cum = np.asarray(out["Cum_D_coll"])
    assert np.all(np.diff(cum) >= -1e-10)
    assert cum[-1] >= -1e-12
