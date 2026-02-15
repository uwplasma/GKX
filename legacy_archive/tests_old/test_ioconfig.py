import io
import os
from spectraxgk_old.io_config import read_toml


TOML_TXT = b"""
[sim]
tmax=1.0
nt=3
precision="x64"
[grid]
Nn=4
Nm=2
kpar=0.0
vth=1.0
nu=0.0
[ic]
kind="n0_mode"
amp=1e-3
phase=0.0
[paths]
outdir="outputs"
outfile="out.npz"
"""


def test_read_toml(tmp_path):
    p = tmp_path / "case.toml"
    p.write_bytes(TOML_TXT)
    cfg = read_toml(str(p))
    assert cfg.sim.nt == 3
    assert cfg.grid.Nn == 4
    assert cfg.paths.outfile == "out.npz"
