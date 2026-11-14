# Examples

One directory per step of the [tutorial](../docs/tutorials.rst); read and run
them in order. Each `run.py` is a plain top-to-bottom script with the same
layout: a docstring, editable constants in capitals, a fast path (the deck run
through `gkx.solve` or `gkx.scan`; the deck's `[run]` block sets `ky`, `Nl`, `Nm` and `solver` unless the call passes them), a research path where one
exists (convergence, eigensolver, autodiff or stepper checks, labelled in the
script), then `summary.json` and one figure in `./outputs/NN_*`. Run from any
directory:

```bash
python examples/01_linear_tokamak/run.py
```

`case.toml` uses a small resolution; its header comment states the accuracy.
`case_full.toml`, where present, is the literature-resolution deck; run it with
`gkx case_full.toml`. Full decks are not run in CI.

| Directory | What it shows |
|---|---|
| [`01_linear_tokamak/`](01_linear_tokamak) | Cyclone ITG growth-rate scan and eigenfunction; velocity-resolution and eigensolver check; the same mode in Miller geometry (`case_miller.toml`) |
| [`02_linear_stellarator/`](02_linear_stellarator) | the same in an HSX-like VMEC stellarator |
| [`03_nonlinear_tokamak/`](03_nonlinear_tokamak) | nonlinear Cyclone ITG; a run to saturation with the saturation verdict and the window mean and standard error |
| [`04_nonlinear_stellarator/`](04_nonlinear_stellarator) | nonlinear ITG in the stellarator flux tube |
| [`05_kinetic_electrons/`](05_kinetic_electrons) | kinetic electrons, tokamak and QA stellarator, per-species `Nl`/`Nm`, explicit and `imex-ars3` |
| [`06_electromagnetic/`](06_electromagnetic) | kinetic ballooning mode with A_parallel |
| [`07_collisions/`](07_collisions) | every collision operator on one ITG case |
| [`08_quasilinear/`](08_quasilinear) | quasilinear spectrum; implicit eigenpair derivatives of its observables |
| [`09_autodiff/`](09_autodiff) | autodiff Jacobian of the growth rates, and a gradient inverse problem |
| [`10_vmex_optimization/`](10_vmex_optimization) | VMEX QA optimization with a nonlinear heat-flux objective |
| [`11_parallel_scan/`](11_parallel_scan) | parallel `k_y` scan, checked against serial |
| [`12_restart_and_analysis/`](12_restart_and_analysis) | restart a nonlinear run, classify and average the joined trace |
| [`13_advanced_time_stepping/`](13_advanced_time_stepping) | advanced: pin the nonlinear stepper (`rk4-pi`, `lawson-rk4-pi`, fixed step) against the default; `gkx.prepare` to compile once and solve repeatedly |
| [`14_zonal_flows/`](14_zonal_flows) | Rosenbluth-Hinton residual and GAM against the formula; collisional zonal damping |

Steps 02, 04 and 05 (stellarator) need a VMEC `wout` from [`vmec/`](vmec). If
it is missing, the script solves it once with `vmex` (`pip install vmex`) or stops with a message. `vmec/generate_wouts.sh` builds all
of them. `common_input.toml` is the deck behind `gkx wout_XXX.nc`; it lists only the keys that differ from the defaults, and the decks here start from it and comment each departure from its stepper and end-damping choice.
[`reference_inputs/`](reference_inputs) holds a linear, a nonlinear and a
kinetic-electron electromagnetic deck that set every key at its default, one
comment per key; `docs/inputs.rst` lists the keys.

## Command line

Every `case.toml` also runs from the shell, with the same `[run]` defaults:

```bash
gkx examples/06_electromagnetic/case.toml                   # one linear point or a nonlinear run
gkx examples/06_electromagnetic/case.toml --set physics.beta=0.01 --set run.ky=0.4   # any key
gkx run --config examples/01_linear_tokamak/case.toml --ky 0.2 --Nl 8 --Nm 24 --solver krylov
gkx scan --config examples/01_linear_tokamak/case.toml --ky-values 0.2,0.3,0.4 --out outputs/scan
gkx run --config examples/03_nonlinear_tokamak/case.toml --until-saturated --out outputs/nl
gkx validate examples/03_nonlinear_tokamak/case.toml        # check it is runnable
```

`--out PREFIX` names the outputs (`PREFIX.summary.json`, `.timeseries.csv` or `.diagnostics.csv`, and PNG figures). `--set section.key=value` overrides any deck key for that call. A flag overrides the deck's `[run]` value; `--plot` also opens the figures in
a window, `--no-plots` skips the PNGs. `gkx run --help` lists the flags. The Python scripts
do the same through `gkx.solve` and `gkx.scan`.

Validation decks (GX parity, convergence ladders) live in
[`benchmarks/cases/`](../benchmarks/cases). Every script is run at tutorial
resolution by `tests/integration/examples/test_examples.py`; scripts that need
`vmex` or a long campaign are skipped there with the reason stated.
