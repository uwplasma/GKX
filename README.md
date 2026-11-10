# GKX

[![Release](https://img.shields.io/github/v/release/uwplasma/GKX?display_name=tag)](https://github.com/uwplasma/GKX/releases)
[![PyPI](https://img.shields.io/pypi/v/gkx.svg)](https://pypi.org/project/gkx/)
[![CI](https://github.com/uwplasma/GKX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/GKX/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/GKX/graph/badge.svg)](https://codecov.io/gh/uwplasma/GKX)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![Docs](https://readthedocs.org/projects/gkx/badge/?version=latest)](https://gkx.readthedocs.io)

GKX is a JAX gyrokinetic solver for tokamak and stellarator flux tubes. It
reads a VMEC or VMEX equilibrium, or builds an analytic Miller or s-alpha
geometry, and computes linear stability and nonlinear turbulence in a
Hermite-Laguerre velocity basis. The whole path, equilibrium included, is
differentiable on CPUs and GPUs.

- **Run what you have:** a VMEC/VMEX `wout`, a Miller or s-alpha tokamak, or
  one TOML deck. The executable sizes the grid and says why.
- **Trust the eigenvalue:** every returned eigenpair is checked against the
  full operator; an under-resolved run warns instead of reporting a number.
- **Differentiate it:** implicit eigenvalue derivatives, and a checkpointed
  adjoint of the nonlinear heat flux over a saturated window.
- **Optimize with it:** the heat flux, growth rate or quasilinear flux is an
  objective in VMEX stellarator shape optimization.

<img src="docs/_static/turbulence_loop.webp" width="720" alt="Saturated ITG turbulence on a Cyclone flux tube, shown as a perpendicular cut and as the field-aligned tube">

Illustration of ITG turbulence on a Cyclone flux tube
([full-rate movie](https://github.com/uwplasma/GKX/releases/download/v1.7.0/gkx-cyclone-itg-turbulence.mp4)).

## Install and first run

```bash
pip install gkx
gkx
```

Python 3.10+. The wheel installs CPU JAX; for GPUs add an accelerator JAX wheel
([JAX install guide](https://docs.jax.dev/en/latest/installation.html)).
`gkx` with no arguments runs a linear Cyclone case and prints its growth
rate `gamma` and real frequency `omega`. The [tutorial](docs/tutorials.rst) walks
through twelve runs, from this one to a stellarator optimization; each is a
directory in [examples/](examples/).

## Run a case

```bash
gkx wout_circular_tokamak.nc --estimate   # size the grid, explain it, exit
gkx wout_circular_tokamak.nc              # nonlinear ITG run to saturation
gkx plot wout_circular_tokamak/gkx.out.nc # replot a saved bundle
gkx examples/01_linear_tokamak/case.toml  # any TOML deck
```

`--estimate` derives each grid entry from the geometry and prints the reason. A
run stops when heat flux, field energy and free energy are stationary, or
reports that it did not. It writes figures, a restartable NetCDF bundle and the
resolved deck that reproduces it ([inputs](docs/inputs.rst),
[outputs](docs/outputs.rst)).

From Python:

```python
import gkx

case = gkx.load("examples/01_linear_tokamak/case.toml")
result = gkx.solve(
    case, ky_target=0.3, Nl=case.run.Nl, Nm=case.run.Nm, show_progress=True
)
print(result.gamma, result.omega)
```

Independent scans and ensembles support parallel workers; nonlinear solves can
opt into a species–Hermite device mesh, with serial execution as the default.
See [parallel execution](docs/parallelization.rst) and
[features and limitations](docs/release_scope.rst) for supported configurations,
identity checks, and eigenmode and finite-window gradient limitations.

## Models and defaults

| Choice | Default | Notes |
|---|---|---|
| Geometry | s-alpha, Miller, VMEC/VMEX, imported | `[geometry] model`; `gkx wout_*.nc` for any VMEC |
| Velocity space | Hermite-Laguerre, `(Nl, Nm)`, per species with `[[species]] Nl`/`Nm` | hypercollisions damp the last Hermite moments; kinetic-electron decks that name no resolution get ions (12, 32) and electrons (6, 48) linear, (4, 16) nonlinear ([guidance](docs/guidance.rst)) |
| Hermite closure (opt-in) | `[[species]] hermite_closure = "reflectionless"` | absorbing last-moment closure per species; check resolution and closure sensitivity ([numerics](docs/numerics.rst)) |
| Perpendicular velocity (opt-in) | nodal `mu` grid: `[run] mu_vperp_max = 4.0`, `Nl` = nodes | linear runs; check node count, velocity extent and agreement with the Laguerre representation ([numerics](docs/numerics.rst)) |
| Collisions | linearized Landau | conserving, H-theorem; Dougherty, Sugama alternatives ([operators](docs/operators.rst)) |
| End damping | one rate, `[time] damp_ends_rate` | same on every route; no time step enters ([numerics](docs/numerics.rst)) |
| Time stepping | no `dt`: nonlinear `auto` (`lawson-rk4-pi` where the magnetic drift is at least 56% of the linear CFL bound, else `rk4-pi`), or adaptive `imex-ark3` for kinetic electrons at beta <= 3e-4; linear `rk4` + CFL | a `dt` gives `rk2` at that step; `fixed_dt = false` keeps the controller's stepper ([method guidance](docs/guidance.rst)) |
| Eigensolver | `auto`: banded direct shift-invert where it fits, else Arnoldi on the time-advance map; every pair certified | matrix-free shift-invert opt-in ([algorithms](docs/algorithms.rst)) |
| SOLVAX | `solvax>=0.35.0` | provides the sparse eigensolver and the `rk4-pi` stepper; installed with `pip install gkx` |
| `ky` layout | `half`: the `1 + Ny//2` non-negative rows | `[grid] ky_layout = "full"` stores the two-sided axis; check equivalent states and diagnostics when comparing layouts |
| Drift kernel | fused Pallas kernel on NVIDIA GPUs when `Nx*Nz >= 64` and `Nl*Nm <= 256` | default there; CPU and other GPUs keep the `jnp` terms, which derivatives use ([performance](docs/performance.rst)) |
| Devices | one device | `[parallel] strategy = "batch"`, `axis = "ky"`, `num_devices` spreads independent `k_y` scans and ensembles; results match the serial run ([parallelization](docs/parallelization.rst)) |
| Precision | `float32`; `float64` with `JAX_ENABLE_X64=true` | eigensolver, Krylov residual and conditioning checks need `float64` |
| CPU threads | XLA's own; Eigen pool off on fewer than four CPUs | choose thread and worker counts by measuring the intended workload ([performance](docs/performance.rst)) |

[Resolution, step and solver guidance](docs/guidance.rst) says what to set for
kinetic electrons and electromagnetic runs, and how to check it.

## Validation and comparisons

| Problem | Quantities to compare |
| --- | --- |
| Slab waves and Landau damping | growth or damping rate, real frequency, dispersion and recurrence |
| Cyclone, Miller, ETG and KBM | eigenpair residual, branch identity, growth rate, frequency and eigenfunction |
| W7-X and HSX flux tubes | geometry profiles, fields, linear modes and transport diagnostics |
| Nonlinear turbulence | stationary heat-flux windows, energies, resolved spectra and replicated-seed uncertainty |

Run the input decks in [benchmarks/cases](benchmarks/cases) and the workflows
in [examples](examples), then compare the same model, geometry, normalization,
resolution, timestep policy, collisions and diagnostic definitions. The
[validation guide](docs/validation.rst) explains convergence and uncertainty
checks; [conventions](docs/conventions.rst) gives cross-code conversions.
Agreement at one finite resolution supports that configuration and observable.

`benchmarks/cases/cyclone_nonlinear_t400.toml` (64x64x24, (Nl, Nm) = (4, 8)) is a
speed benchmark, not a converged calculation: the saturated heat flux is
Q = 8.6 at Nx = 64 and 4.8 at Nx = 192, and (4, 8) overestimates the linear
growth rate by 9 to 86% at ky = 0.3 to 0.5.

References include the [GX code paper](https://doi.org/10.1017/S0022377822000617)
and the [W7-X stella–GENE benchmark](https://doi.org/10.1017/S0022377822000393).
Use their declared models and analysis protocols when reconstructing a
comparison. Keep generated results outside version control and record the
input deck, command, source revisions and acceptance criteria with each result.

## Derivatives and optimization

Eigenvalue derivatives use `dλ/dp = wᴴ(dA/dp)v / (wᴴv)` with residual,
overlap and conditioning checks. Nonlinear heat-flux derivatives use a
checkpointed discrete adjoint over a finite post-saturation window. Compare
these derivatives with finite differences across perturbation sizes and
window lengths; assess a design change with independent long simulations.
See [nonlinear autodiff](docs/nonlinear_autodiff.rst).

[VMEX](https://github.com/uwplasma/vmex) composes its equilibrium derivative
with GKX solver objectives. Run
[the optimization example](examples/10_vmex_optimization/run.py) to combine
heat flux with equilibrium constraints. Quasilinear flux requires calibration
and held-out nonlinear validation before it supports an absolute transport
prediction ([quasilinear](docs/quasilinear.rst),
[stellarator optimization](docs/stellarator_optimization.rst)).

## Performance

Reuse prepared cases when topology is fixed. Warm the exact callable before
timing, wait for device completion and report compilation separately from
execution. Check numerical agreement before comparing kernels or device
layouts, and include diagnostics and I/O in end-to-end measurements.

Each species can use its own `Nl` and `Nm`; validate the observable at higher
moment counts before reducing them. Named species–Hermite parallel execution
has narrower operator support than whole-state placement, so inspect the
resolved routing message when assessing scaling.
See [performance](docs/performance.rst) and
[parallel execution](docs/parallelization.rst).

## Documentation and development

Documentation: **[gkx.readthedocs.io](https://gkx.readthedocs.io)**. Start with
the [tutorial](docs/tutorials.rst) and the [guidance page](docs/guidance.rst);
equations and algorithms are in [theory](docs/theory.rst),
[operators](docs/operators.rst) and [numerics](docs/numerics.rst); modules in
[design](docs/design.rst).

```bash
git clone https://github.com/uwplasma/GKX
cd GKX
pip install -e ".[dev]"
pytest -n 4
```

CI requires 95% line coverage plus the physics, convergence, comparison,
differentiability and performance gates ([CONTRIBUTING.md](CONTRIBUTING.md)).
Keep generated test and benchmark outputs outside version control.
Cite GKX with [CITATION.cff](CITATION.cff).

## License

MIT; see [LICENSE](LICENSE).
