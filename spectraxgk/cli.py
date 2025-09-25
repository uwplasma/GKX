from __future__ import annotations
import argparse
from .io_config import read_toml
from .solver import run_simulation


def main():
    p = argparse.ArgumentParser(description="Run SPECTRAX-GK from a TOML config")
    # Support either positional TOML or --input TOML
    p.add_argument("input_path", nargs="?", default=None, help="Path to TOML config (positional)")
    p.add_argument("--input", dest="input_path_flag", type=str, default=None,
    help="Path to TOML config (optional flag)")
    args = p.parse_args()


    path = args.input_path_flag or args.input_path or "examples/linear_slab.toml"
    cfg = read_toml(path)
    info = run_simulation(cfg)
    print(f"[OK] wrote {info['outfile']}")