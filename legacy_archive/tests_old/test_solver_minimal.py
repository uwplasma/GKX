import numpy as np
from spectraxgk_old.io_config import FullConfig, SimConfig, GridConfig, ICConfig, PathsConfig
from spectraxgk_old.solver import run_simulation


def test_run_small_constant_step(tmp_path):
    cfg = FullConfig(
        sim=SimConfig(tmax=0.5, nt=21, precision="x64", solver="tsit5", adaptive=False, dt=0.025),
        grid=GridConfig(Nn=4, Nm=3, kpar=0.0, vth=1.0, nu=0.1),
        ic=ICConfig(kind="n0_mode", amp=1e-3, phase=0.0),
        paths=PathsConfig(outdir=str(tmp_path), outfile="mini.npz"),
    )
    info = run_simulation(cfg)
    data = np.load(info["outfile"], allow_pickle=True)
    # Basic shape checks
    C = data["C"]
    t = data["t"]
    assert C.shape == (cfg.sim.nt, cfg.grid.Nn, cfg.grid.Nm)
    assert t.shape[0] == cfg.sim.nt
