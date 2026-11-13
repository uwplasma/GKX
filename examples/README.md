# Examples

One directory per step of the [tutorial](../docs/tutorials.rst); read and run
them in order. Each `run.py` has the same shape: imports, editable parameters
at the top, a run that prints its progress, then printed, saved and plotted
results under `./outputs/NN_*`. Run from any directory:

```bash
python examples/01_linear_tokamak/run.py
```

`case.toml` uses tutorial resolution.
`case_full.toml`, where present, is the literature-resolution deck; run it with
`gkx case_full.toml`. Full decks are not run in CI.

| Directory | What it shows |
|---|---|
| [`01_linear_tokamak/`](01_linear_tokamak) | Cyclone ITG growth-rate scan and eigenfunction |
| [`02_linear_stellarator/`](02_linear_stellarator) | the same in an HSX-like VMEC stellarator |
| [`03_nonlinear_tokamak/`](03_nonlinear_tokamak) | nonlinear Cyclone ITG |
| [`04_nonlinear_stellarator/`](04_nonlinear_stellarator) | nonlinear ITG in the stellarator flux tube |
| [`05_kinetic_electrons/`](05_kinetic_electrons) | kinetic electrons, tokamak and QA stellarator, per-species `Nl`/`Nm`, explicit and `imex-ars3` |
| [`06_electromagnetic/`](06_electromagnetic) | kinetic ballooning mode with A_parallel |
| [`07_collisions/`](07_collisions) | every collision operator on one ITG case |
| [`08_quasilinear/`](08_quasilinear) | quasilinear spectrum; implicit eigenpair sensitivities |
| [`09_autodiff/`](09_autodiff) | recover gradients from two modes by autodiff |
| [`10_vmex_optimization/`](10_vmex_optimization) | VMEX QA optimization with a nonlinear heat-flux objective |
| [`11_parallel_scan/`](11_parallel_scan) | parallel `k_y` scan, checked against serial |
| [`12_restart_and_analysis/`](12_restart_and_analysis) | restart a nonlinear run, average the joined trace |
| [`13_advanced_time_stepping/`](13_advanced_time_stepping) | advanced: pin the nonlinear stepper (`rk4-pi`, `lawson-rk4-pi`, fixed step) against the default |

Steps 02, 04 and 05 (stellarator) need a VMEC `wout` from [`vmec/`](vmec). If
it is missing, the script solves it once with `vmex` (`pip install vmex`) or stops with a message. `vmec/generate_wouts.sh` builds all
of them. `common_input.toml` is the deck behind `gkx wout_XXX.nc`; it lists only the keys that differ from the defaults, and the decks here start from it and comment each departure from its stepper and end-damping choice.
[`reference_inputs/`](reference_inputs) holds a linear, a nonlinear and a
kinetic-electron electromagnetic deck that set every key at its default, one
comment per key; `docs/inputs.rst` lists the keys.

Validation decks (GX parity, convergence ladders) live in
[`benchmarks/cases/`](../benchmarks/cases). Every script is run at tutorial
resolution by `tests/integration/examples/test_examples.py`; scripts that need
`vmex` or a long campaign are skipped there with the reason stated.
