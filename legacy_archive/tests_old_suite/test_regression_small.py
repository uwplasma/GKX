# tests/test_regression_small.py
import numpy as np
from spectraxgk._simulation import simulation


def test_regression_small_run_invariants():
    out = simulation(
        input_parameters=dict(
            t_max=0.3,
            nu=0.01,
            pert_amp=1e-3,
            enable_streaming=True,
            enable_nonlinear=True,
            enable_collisions=True,
            enforce_reality=True,
        ),
        Nx=7, Ny=7, Nz=5,
        Nl=3, Nh=6,
        timesteps=25,
        dt=1e-2,
        adaptive_time_step=False,
        save="diagnostics",
        progress=False,
    )

    t = np.asarray(out["time"])
    W = np.asarray(out["W_total"])
    D = np.asarray(out["D_coll"])
    cum = np.asarray(out["Cum_D_coll"])

    # Basic sanity invariants
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(D))
    assert np.all(np.isfinite(cum))

    # D_coll should be >= 0 (up to tiny numerical noise)
    assert np.min(D) >= -1e-10

    # Cum_D_coll should match trapezoid(D_coll) reasonably
    cum_ref = np.concatenate([[0.0], np.cumsum(0.5 * (D[1:] + D[:-1]) * (t[1:] - t[:-1]))])
    assert np.max(np.abs(cum - cum_ref)) <= 1e-9

    # With collisions on, energy should not blow up
    assert W[-1] <= 2.0 * W[0]
