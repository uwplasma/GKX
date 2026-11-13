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

<img src="docs/_static/four_code_linear.png" width="860" alt="Growth rate, real frequency and growth-rate deviation from GKX for GKX, GX, GS2 and stella: Cyclone with adiabatic and kinetic electrons, a finite-beta scan, and W7-X">

<img src="docs/_static/four_code_speed.png" width="860" alt="Wall time of one linear k_y at 1% growth-rate accuracy and of the nonlinear Cyclone run to t = 400 for GKX, GX, GS2 and stella, with hardware and heat flux">

<img src="docs/_static/four_code_traces.png" width="860" alt="Nonlinear Cyclone heat flux against time for GKX, GX, GS2 and stella">

GKX, GX, GS2 and stella solve the same flux-tube problems in GX units: circular Miller Cyclone (`a/L_T = 2.49`, `a/L_n = 0.8`, `q = 1.4`, `s_hat = 0.8`) with adiabatic or kinetic electrons (`T_e = T_i`, `m_i/m_e = 3670`), and the W7-X standard VMEC configuration (torflux 0.64, adiabatic electrons). GS2 and stella use `sqrt(2T/m)` as the reference speed; their `gamma`, `omega`, `k_y`, time and heat flux are converted with `gamma_GX = sqrt(2) gamma_GS2`, `k_y,GX = k_y,GS2/sqrt(2)`, `t_GX = t_GS2/sqrt(2)` and `Q_GX = 2 sqrt(2) Q_GS2`. Each code is at its own converged resolution in the linear panels. The adiabatic Cyclone growth rate at `k_y rho_i = 0.3` is 0.1255 (GKX, GX), 0.1254 (GS2) and 0.1252 (stella); GKX and GX share the Hermite-Laguerre model and agree to 0.1% for `k_y rho_i >= 0.2`, GS2 and stella lie 1% below them at `k_y rho_i = 0.4` and 3.5% at 0.6, and with kinetic electrons they stay within 6%. In the `A_parallel` scan GS2 is within 1% up to `beta_ref = 0.01` and 5-6% lower on the ballooning branch above it; on W7-X stella is 2-4% below for `k_y rho_i >= 1`, and for `k_y rho_i <= 0.6` near-degenerate modes make the fitted branch depend on the code. One linear `k_y` takes 4.2 s on an RTX A4000 with GKX at 1% accuracy, against 47 s for GX on the same GPU and 24 s for GS2 and 29 s for stella on 4 cores. The nonlinear Cyclone deck of the GX paper reaches `t v_ti/a = 400` in 523 s with GKX (`Q_i = 6.9 +- 0.2`) and 1610 s with GX (`6.6 +- 0.3`; paper 7.2 +- 0.8). GS2 and stella run the same box on 8 MPI ranks of an 18-core Xeon W-2295 at 2.3 and 2.0 s per step: in 5.9 h (47 core-hours each) they reach `t = 100` and `140`, and at that step rate and time step they need about 33 h and 16 h (264 and 128 core-hours) for `t = 400`. Both reproduce the first peak and the relaxation that follows; the stella mean over `t = 104 .. 140` is 6.1 +- 0.8 (GKX 7.3, GX 6.7 over the same interval) but spans five autocorrelation times, and GS2 has not left the relaxation. The GPU and the CPU were shared with other jobs throughout.

| Code | Hardware | DOF per `k_y` (linear) | Linear time per `k_y` | Nonlinear time to `t = 400` | Nonlinear `Q_i` | `gamma`, `k_y rho_i = 0.3` |
| --- | --- | --- | --- | --- | --- | --- |
| GKX | RTX A4000 | 55,296 | 4.2 s (10 s with compilation) | 523 s | 6.9 +- 0.2 | 0.1255 |
| GX | RTX A4000 | 55,296 | 47 s | 1610 s (from the step rate) | 6.6 +- 0.3 | 0.1255 |
| GS2 | 4 cores linear, 8 cores nonlinear | 190,240 | 24 s | about 33 h (264 core-h); run to `t = 100` | not reached | 0.1254 |
| stella | 4 cores linear, 8 cores nonlinear | 83,520 | 29 s | about 16 h (128 core-h); run to `t = 140` | 6.1 +- 0.8 over `t = 104 .. 140`, not converged | 0.1252 |

The linear rows use the cheapest resolution whose growth rate stays within 1% of the code's own highest-resolution value over `k_y rho_i = 0.2` to `0.5`: GKX and GX `(Nl, Nm) = (12, 32)`, `Nz = 144`; GS2 48 `theta` per `2 pi` x 3, 16 energies, 41 pitch angles; stella `nzed = 48` x 3, `nvgrid = 24`, `nmu = 12`. The time is one `k_y` to `t v_ti/a = 60` (GS2 and stella: the same physical time), taken from the initial-value route for GKX and GX; the certified eigensolver takes 43 s at `(12, 32)` on the same GPU (see [guidance](docs/guidance.rst)). The nonlinear deck is `(Nx, Ny, Nz) = (192, 64, 24)`, `y0 = 28.2`, adiabatic electrons, with 127 x 22 Fourier modes: GKX and GX use `(Nl, Nm) = (4, 8)` (2.1 M degrees of freedom), GS2 21 pitch angles x 8 energies x 25 `theta` (23.5 M, growth rate 1.2% below its converged value at `k_y rho_i = 0.3`) and stella 24 `v_par` x 6 `mu` x 25 `z` (10.1 M, 3% below). GS2 and stella have no end damping or hypercollisions; all four use hyperdiffusion with `D = 0.05`. GS2 halves its time step to 0.01 (code units) at the first peak under its default CFL control, stella keeps 0.02. The projected times assume the last step rate and time step; the heat flux uses the same estimator for every code. GKX and GX run in single precision. `docs/figures/four_code_comparison.json` holds every plotted value and `python docs/figures/four_code_comparison.py` redraws the three figures from it.

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
| Velocity space | Hermite-Laguerre, `(Nl, Nm)`, per species with `[[species]] Nl`/`Nm` | hypercollisions damp the last Hermite moments; kinetic-electron decks that name no resolution get ions (12, 32) and electrons (6, 48) linear, ions (4, 8) and electrons (4, 48) nonlinear ([guidance](docs/guidance.rst)) |
| Hermite closure (opt-in) | `[[species]] hermite_closure = "reflectionless"` | absorbing last-moment closure per species; check resolution and closure sensitivity ([numerics](docs/numerics.rst)) |
| Perpendicular velocity (opt-in) | nodal `mu` grid: `[run] mu_vperp_max = 4.0`, `Nl` = nodes | linear runs; check node count, velocity extent and agreement with the Laguerre representation ([numerics](docs/numerics.rst)) |
| Collisions | linearized Landau | conserving, H-theorem; Dougherty, Sugama alternatives ([operators](docs/operators.rst)) |
| End damping | one rate, `[time] damp_ends_rate` | same on every route; no time step enters ([numerics](docs/numerics.rst)) |
| Time stepping | no `dt`: nonlinear `auto` (`lawson-rk4-pi` where the magnetic drift is at least 56% of the linear CFL bound and the step cap at least 2.6 times the `rk4-pi` bound step, else `rk4-pi`), or adaptive `imex-ark3` for kinetic electrons at beta <= 3e-4; linear `rk4` + CFL | a `dt` gives `rk2` at that step; `fixed_dt = false` keeps the controller's stepper ([method guidance](docs/guidance.rst)) |
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
