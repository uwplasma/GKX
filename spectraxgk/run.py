# run.py
# Ensure x64 BEFORE importing jax
import os
os.environ["JAX_ENABLE_X64"] = "true"

import argparse
import time

import jax
jax.config.update("jax_enable_x64", True)

from io_config import read_toml
from backends import run_fourier, run_dg
from plots import render_suite_onefigure


def main():
    parser = argparse.ArgumentParser(description="Run 1D Vlasov–Poisson experiment")
    parser.add_argument("--input", type=str, default="input.toml",
                        help="Path to TOML config (default: input.toml)")
    args = parser.parse_args()

    cfg = read_toml(args.input)

    t0 = time.perf_counter()
    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")
    print(f"[run] solve time: {time.perf_counter() - t0:.2f}s")

    # one call to plot everything (diagnostics + distributions)
    render_suite_onefigure(cfg, ts, out)


if __name__ == "__main__":
    main()
