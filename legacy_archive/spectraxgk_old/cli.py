from __future__ import annotations
import time
from .args import parse_args
from .io_config import read_toml
from .solver import run_simulation
from .pretty import init_pretty, print_preflight, print_summary, info_line


def main():
    args = parse_args()
    init_pretty(prefer_rich=not args.no_rich)

    cfg = read_toml(args.path)

    # Preflight BEFORE solving
    print_preflight(args.path, cfg)
    if args.dry_run:
        return

    info_line("Starting solve…")
    t0 = time.time()
    info = run_simulation(cfg)
    t1 = time.time()

    print_summary(cfg, info, t1 - t0)