# Examples

A numbered gallery, one directory per workflow. Each `run.py` reads top to
bottom: editable UPPER_CASE parameters at the top, then geometry, the run,
a printed summary, a JSON summary, and a figure under
`outputs/<group>/` in the working directory. `case.toml` is a short tutorial
deck that runs in seconds to a minute on a laptop CPU. Where it exists,
`case_full.toml` is the literature- or production-resolution deck. Run it
with `gkx case_full.toml`, or point `CASE` in `run.py` at it. Full decks are
not run in CI.

| Group | What it shows | Tutorial runtime |
|---|---|---|
| [`01_linear_tokamak/`](01_linear_tokamak) | Cyclone ITG growth-rate scan and eigenfunction, s-alpha geometry | ~10 s CPU |
| [`02_linear_stellarator/`](02_linear_stellarator) | ITG scan in an HSX-like quasi-helically symmetric VMEC equilibrium (needs `vmex` once to build the wout) | ~1 min CPU |
| [`03_nonlinear_tokamak/`](03_nonlinear_tokamak) | Nonlinear Cyclone ITG with streamed energy and heat-flux diagnostics | ~10 s CPU |
| [`04_nonlinear_stellarator/`](04_nonlinear_stellarator) | Nonlinear ITG in the same VMEC stellarator flux tube (needs `vmex` once) | ~1 min CPU |
| [`05_kinetic_electrons/`](05_kinetic_electrons) | Cyclone ITG with kinetic electrons against adiabatic electrons | ~10 s CPU |
| [`06_electromagnetic/`](06_electromagnetic) | Kinetic ballooning mode with A_parallel; the tutorial deck exercises the path, `case_full.toml` resolves the mode | ~10 s CPU |
| [`07_collisions/`](07_collisions) | Every shipped collision operator on one ITG case; optional collisionality scan | <1 min CPU |
| [`08_quasilinear/`](08_quasilinear) | Quasilinear heat-flux spectrum (`run.py`) and its implicit eigenpair sensitivities (`implicit_sensitivity.py`) | ~10 s + <1 min CPU |
| [`09_autodiff/`](09_autodiff) | Recover a/L_Ti and a/L_n from two linear modes by autodiff Gauss-Newton; `geometry_bridge.py` differentiates through VMEC/Boozer geometry | ~1 min CPU; bridge: minutes |
| [`10_vmex_optimization/`](10_vmex_optimization) | VMEX QA stellarator optimization with a physical nonlinear GKX heat-flux objective (see its README) | research run, GPU |
| [`11_parallel_scan/`](11_parallel_scan) | Independent `k_y` scan dispatched to parallel workers, checked against the serial scan | ~10 s CPU |
| [`12_restart_and_analysis/`](12_restart_and_analysis) | Restart a nonlinear run from its saved state and average the joined heat-flux trace | ~20 s CPU |

Shared inputs:

- `vmec/`: small VMEC input decks. Generate the `wout_*.nc` files with
  `vmec/generate_wouts.sh`, or let groups 02 and 04 solve the one they need.
- `common_input.toml`: the default deck behind the `gkx wout_XXX.nc`
  shorthand. It is packaged with GKX as `gkx/data/common_input.toml`.

Validation-only decks (GX parity, convergence ladders, restart and device
gates) live in [`benchmarks/cases/`](../benchmarks/cases). Every gallery
script is executed at tutorial resolution by
`tests/integration/examples/test_examples.py`. Scripts that need `vmex` or a
multi-minute campaign are skipped there, with the reason stated.

Examples are run from any directory, for example
`python examples/01_linear_tokamak/run.py`. Outputs land in `./outputs/`.
