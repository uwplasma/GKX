from spectraxgk.backends import resolve_kgrid, run_fourier
from spectraxgk.io_config import BCCfg, Config, GridCfg, HermiteCfg, PlotCfg, SimCfg, SpeciesCfg


def _tiny_cfg():
    sim = SimCfg(mode="fourier", backend="eig", tmax=1e-6, nt=8, nonlinear=False)
    grid = GridCfg(L=1.0, Nx=16)
    hermite = HermiteCfg(N=8)
    bc = BCCfg(kind="periodic")
    plot = PlotCfg()
    species = [
        SpeciesCfg(
            name="e",
            q=-1.0,
            n0=1e6,
            mass_base="electron",
            mass_multiple=1.0,
            temperature_eV=1.0,
            drift_c=0.0,
            amplitude=1e-3,
            k=1,
        )
    ]
    return Config(sim=sim, grid=grid, hermite=hermite, bc=bc, plot=plot, species=species)


def test_fourier_linear_shapes():
    cfg = _tiny_cfg()
    ts, out = run_fourier(cfg)
    assert "C_kSnt" in out and "Ek_kt" in out and "k" in out
    C = out["C_kSnt"]
    Nk = resolve_kgrid(cfg.grid).shape[0]
    assert C.shape[0] == Nk
    assert C.shape[1] == 1
    assert C.shape[2] == cfg.hermite.N
    assert C.shape[3] == cfg.sim.nt
    assert ts.shape[0] == cfg.sim.nt
