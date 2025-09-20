# Ensure x64 BEFORE importing jax
import os

os.environ.setdefault("JAX_ENABLE_X64", "true")

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

from spectraxgk.backends import run_dg, run_fourier
from spectraxgk.io_config import read_toml
from spectraxgk.plots import render_suite_onefigure
from spectraxgk.util import (
    _compute_lambda_D,
    _compute_wp,
    _pick_debye_species,
    print_sim_summary,
    print_units_banner,
)


def main():
    parser = argparse.ArgumentParser(description="Run simulation from TOML")
    parser.add_argument(
        "--input",
        type=str,
        default="examples/two_stream.toml",
        help="Path to TOML config (default: examples/two_stream.toml)",
    )
    args = parser.parse_args()

    if args.input is None:
        raise SystemExit("Please provide --input path to a TOML config (see repo examples/).")

    cfg = read_toml(args.input)

    # --- Units normalization ---
    sp_ref, idx_ref = _pick_debye_species(cfg)
    omega_p = _compute_wp(sp_ref)  # rad/s
    tmax_over_wp = float(cfg.sim.tmax)
    cfg.sim.tmax = tmax_over_wp / omega_p  # seconds
    lambda_d = _compute_lambda_D(sp_ref)
    cfg.grid.L = float(cfg.grid.L_lambdaD) * lambda_d

    # Print statements
    print_units_banner(cfg, sp_ref, idx_ref, omega_p, tmax_over_wp)
    print_sim_summary(cfg)

    # --- Solve ---
    t0 = time.perf_counter()
    if cfg.sim.mode == "fourier":
        ts, out = run_fourier(cfg)
    elif cfg.sim.mode == "dg":
        ts, out = run_dg(cfg)
    else:
        raise SystemExit("sim.mode must be 'fourier' or 'dg'")
    print(f"[run] solve time: {time.perf_counter() - t0:.2f}s")

    render_suite_onefigure(cfg, ts, out)


if __name__ == "__main__":
    main()
