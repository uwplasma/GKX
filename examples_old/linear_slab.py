from spectraxgk_old.io_config import read_toml
from spectraxgk_old.solver import run_simulation
from spectraxgk_old.post import load_result, plot_energy

cfg = read_toml("examples/linear_slab.toml")
info = run_simulation(cfg)
res = load_result(info["outfile"])
plot_energy(res)