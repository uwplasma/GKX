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

## Comparison with GX, GS2 and stella

<img src="docs/_static/four_code_linear.png" width="860" alt="Growth rate, real frequency and growth-rate deviation from GKX for GKX, GX, GS2 and stella: Cyclone with adiabatic and kinetic electrons, a finite-beta scan, and W7-X">

<img src="docs/_static/four_code_speed.png" width="860" alt="Growth-rate error against time to solution, nonlinear Cyclone heat-flux traces and wall times for the four codes">

GKX, GX, GS2 and stella solve the same flux-tube problems in GX units: circular Miller Cyclone (`a/L_T = 2.49`, `a/L_n = 0.8`, `q = 1.4`, `s_hat = 0.8`) with adiabatic or kinetic electrons (`T_e = T_i`, `m_i/m_e = 3670`), and the W7-X standard VMEC configuration (torflux 0.64, adiabatic electrons). GS2 and stella use `sqrt(2T/m)` as the reference speed, so their `gamma`, `omega` and `k_y` are converted with `gamma_GX = sqrt(2) gamma_GS2` and `k_y,GX = k_y,GS2/sqrt(2)`. Each code is shown at its own converged velocity-space and parallel resolution. With adiabatic electrons the converged growth rates at `k_y rho_i = 0.3` are 0.1255 (GKX), 0.1255 (GX), 0.1254 (GS2) and 0.1252 (stella); GKX and GX share the Hermite-Laguerre model and agree to 0.1% for `k_y rho_i >= 0.2`, while GS2 and stella agree with each other and sit below them by 1% at `k_y rho_i = 0.4` and by 3.5% at 0.6. With kinetic electrons GKX and GX agree to 0.2%, GS2 and stella lie within 6% of them in `gamma`, and GS2 reports a real frequency 7% (`k_y rho_i = 0.3`) to 14% (0.2) below the others. In the `A_parallel` scan at `k_y rho_i = 0.3` all three codes that include it show ITG stabilization up to `beta_ref = 0.01` and the kinetic ballooning branch above it; GS2 is within 1% of GKX and GX up to `beta_ref = 0.01` and 5-6% lower on the ballooning branch. On W7-X, GKX and GX agree to 0.1% and stella lies 2-4% below for `k_y rho_i >= 1`; at `k_y rho_i <= 0.6` several modes are nearly degenerate, the fitted branch depends on the code, and the three do not agree. At matched accuracy the GKX initial-value route on one RTX A4000 takes 4.2 s per `k_y` (10 s with compilation) against 47 s for GX on the same GPU and 24 s for GS2 and 29 s for stella on 4 cores; the nonlinear Cyclone heat flux of the GX-paper deck is `Q_i = 6.9 +- 0.2` (GKX) and `6.6 +- 0.3` (GX) in gyro-Bohm units, against 7.2 +- 0.8 published for GX, in 523 s and 1610 s. GS2 and stella nonlinear runs and a stella electromagnetic scan are not part of this comparison; the GPU was shared with other jobs throughout.

| Cyclone, adiabatic electrons, `k_y rho_i = 0.3` | Resolution | DOF per `k_y` | `gamma` (own converged) | Time per `k_y` |
| --- | --- | --- | --- | --- |
| GKX, RTX A4000 | `(Nl, Nm) = (12, 32)`, `Nz = 144` | 55,296 | 0.1255 | 4.2 s (10 s with compilation) |
| GX, RTX A4000 | `(12, 32)`, `Nz = 144` | 55,296 | 0.1255 | 47 s |
| GS2, 4 cores | 48 `theta` per `2 pi` x 3, 16 energies, 41 pitch angles | 190,240 | 0.1255 | 24 s |
| stella, 4 cores | `nzed = 48` x 3, `nvgrid = 24`, `nmu = 12` | 83,520 | 0.1252 | 29 s |

| Nonlinear Cyclone, adiabatic electrons | Resolution | DOF | `Q_i / Q_gB`, `t = 80 .. 400` | Wall time to `t = 400` |
| --- | --- | --- | --- | --- |
| GKX, RTX A4000 | `(Nx, Ny, Nz) = (192, 64, 24)`, `(Nl, Nm) = (4, 8)` | 2,145,792 | 6.9 +- 0.2 | 523 s |
| GX, RTX A4000 | same | 2,145,792 | 6.6 +- 0.3 | 1610 s (from the step rate) |

The linear rows list the cheapest resolution whose growth rate stays within 1% of the code's own highest-resolution value over `k_y rho_i = 0.2` to `0.5`; the time is one `k_y` to `t v_ti/a = 60` (GS2 and stella: the same physical time), and the GKX and GX values come from the initial-value route (the certified eigensolver takes 43 s at `(12, 32)` on the same GPU; see [guidance](docs/guidance.rst)). DOF counts parallel points times velocity points of one kinetic species. GKX and GX run in single precision. `docs/figures/four_code_comparison.json` holds every plotted value and `python docs/figures/four_code_comparison.py` redraws both figures from it.

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
| Time stepping | no `dt`: nonlinear `rk4-pi`, or adaptive `imex-ark3` for kinetic electrons at beta <= 3e-4; linear `rk4` + CFL | a `dt` gives `rk2` at that step; `fixed_dt = false` keeps the controller's stepper ([method guidance](docs/guidance.rst)) |
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
