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
Hermite-Laguerre velocity basis. Linear growth rates and fixed-step nonlinear
heat-flux windows are differentiable, equilibrium included, and the tests
compare those gradients with finite differences on CPU. The adaptive-step
nonlinear run and its saturation stop are not differentiated.

## Comparison across gyrokinetic codes

<img src="docs/_static/four_code_linear.png" width="860" alt="Growth rate, real frequency and growth-rate deviation from GKX for GKX, GX, GS2 and stella: Cyclone with adiabatic and kinetic electrons, a finite-beta scan, and W7-X">

<img src="docs/_static/four_code_speed.png" width="860" alt="Wall time of one linear k_y at 1% growth-rate accuracy and of the nonlinear Cyclone run to t = 400 for GKX, GX, GS2 and stella, with hardware and heat flux">

<img src="docs/_static/four_code_traces.png" width="860" alt="Nonlinear Cyclone heat flux against time for GKX, GX, GS2 and stella">

GKX, GX, GS2 and stella solve the same Cyclone and W7-X flux-tube problems in
GX units: linear growth rate and frequency, wall time, and the nonlinear
Cyclone heat flux. Redraw the figures with
`python docs/figures/four_code_comparison.py`, which reads
`docs/figures/four_code_comparison.json`. The GKX runs use the decks in
`benchmarks/cases`; the GX, GS2 and stella runs need those codes installed.
Resolutions, hardware and caveats are in the
[validation guide](docs/validation.rst).

## Install and first run

```bash
pip install gkx
gkx
```

Python 3.10+. The wheel installs CPU JAX; for GPUs add an accelerator JAX wheel
([JAX install guide](https://docs.jax.dev/en/latest/installation.html)).
`gkx` with no arguments runs a linear Cyclone case and prints its growth rate
`gamma` and real frequency `omega`. The [tutorial](docs/tutorials.rst) walks
through the runs in [examples/](examples), from this one to a stellarator
optimization.

## Run a case

```bash
gkx wout_circular_tokamak.nc --estimate   # size the grid, explain it, exit
gkx wout_circular_tokamak.nc              # nonlinear ITG run to saturation
gkx plot wout_circular_tokamak/gkx.out.nc # replot a saved bundle
gkx examples/01_linear_tokamak/case.toml --plot  # any TOML deck; --plot opens the figures
```

A run stops when heat flux, field energy and free energy are stationary, or
reports that it did not. It writes figures, a restartable NetCDF bundle and the
resolved deck that reproduces it.

```python
import gkx

case = gkx.load("examples/01_linear_tokamak/case.toml")
result = gkx.solve(case, ky_target=0.3, Nl=case.run.Nl, Nm=case.run.Nm)
print(result.gamma, result.omega)
```

Every deck key, with type and default, is listed in [inputs](docs/inputs.rst);
[examples/reference_inputs](examples/reference_inputs) holds decks that set
every key at its default.

## Defaults

| Choice | Default |
|---|---|
| Velocity space | Hermite-Laguerre `(Nl, Nm)`; kinetic electrons get per-species counts |
| Collisions | linearized Landau |
| Nonlinear stepper | `auto`: `lawson-rk4-pi` or `rk4-pi`; `imex-ark3` for kinetic electrons at beta <= 3e-4 |
| Linear solver | `auto`: time integration, then the certified eigensolver (banded shift-invert, else Arnoldi); `krylov` runs the eigensolver directly |
| End damping | one rate on every route, 30 thermal transits across the end layer |
| `ky` layout | `half`, the non-negative rows |
| Precision | `float32`; `JAX_ENABLE_X64=true` for `float64` |

[Guidance](docs/guidance.rst) says what to set for kinetic electrons and
electromagnetic runs, and how to check it.

## Collision operators

`collision_operator` in `[time]` selects the model; `nu` sets the collision
frequency for all of them ([operators](docs/operators.rst)).

- `landau` (default): the linearized Landau operator, exact in the drift-kinetic limit. Use it unless a comparison calls for another model.
- `sugama`: Landau test part with momentum- and energy-restoring field part, self-adjoint between unlike species and temperatures. Use it to compare with codes that use this model.
- `improved_sugama`: `sugama` with the exact Landau heat-flux and friction terms; it tracks `landau` at high collisionality.
- `lenard_bernstein`: conserving Lenard-Bernstein/Dougherty relaxation, the cheapest model. It captures collisional damping qualitatively, not the Landau operator.
- `coulomb_finite_kperp`: tabulated gyrokinetic Coulomb operator with finite k_perp, for one species.

<img src="docs/_static/collision_operators.png" width="860" alt="Collisional trapped-electron mode growth rate against collisionality for the GKX landau, sugama, improved_sugama and lenard_bernstein operators, GS2 and stella, and its reduction relative to the collisionless rate">

Collisional trapped-electron mode at k_y rho_i = 0.3 on the
`benchmarks/cases/tem_collisional.toml` deck. At (Nl, Nm) = (4, 8) the four
operators lie within 4% of one another; GKX is not converged there. Redraw with
`python docs/figures/collision_operators.py`.

<img src="docs/_static/collision_validation.png" width="860" alt="Left: GKX landau collisional trapped-electron mode growth rate against velocity-space size, with GS2 and stella. Right: zonal-flow damping of the four GKX operators against the Xiao-Catto-Molvig formula">

The collisional growth rate of `landau` converges to 0.2587 +- 0.0006, 3% below
GS2 (0.2666) and 0.5% above stella's Fokker-Planck test part (0.2575 at its
second grid); the ratio to the collisionless rate is 0.73 +- 0.02 against GS2's
0.736. The zonal flow relaxes to 0.006 of its initial value for every
operator, against 0.0051 from the formula, which the decay rate matches to
about 10% only below a neoclassical collisionality of 0.3, at the finest resolution
([validation](docs/validation.rst)).

## Flow shear and imposed zonal flows

Two `[physics]` keys add an equilibrium ExB flow to a nonlinear run. Both default
to off.

```toml
[physics]
exb_shear = 0.1          # constant shearing rate gamma_E in v_t/a; 0.0 is off
zonal_flow = [0.0, 0.0, 0.0, 0.4]   # cosine amplitudes of phi_ext, n = 1, 2, ...; [] is off
# zonal_flow = "profile.txt"        # or two columns, x in rho_ref and phi_ext
```

`exb_shear` is a constant shearing rate, typically 0.05 to 0.3, with v_t =
sqrt(T/m). It is sqrt(2) times a GS2 or stella value in sqrt(2T/m)/a and equals
GX `g_exb`. A wave follows kx*(t) = kx - ky gamma_E t. Use it for a slowly varying
equilibrium E_r, such as a neoclassical one, where one shearing rate describes the
box. `zonal_flow` imposes a static potential phi_ext(x) with V_y = dphi_ext/dx
and local shearing rate d2(phi_ext)/dx2, in the units of phi. Use it when the flow
reverses within the box. A file must span the built Lx. Details are in
[numerics](docs/numerics.rst) and [inputs](docs/inputs.rst).

<img src="docs/_static/flow_shear.png" width="760" alt="Cyclone ion heat flux against constant shearing rate for GKX and GX at nx = 192 and 64, and against the peak shear of a four-period zonal flow for two seeds">

Cyclone with adiabatic electrons, each point restarted from a saturated unsheared
state and averaged over the last half of 250 time units. Left: at nx = 192 the
flux falls to 7% of the unsheared value at gamma_E = 0.2 in GKX and to 6% in GX.
Right: a sinusoidal flow of four periods across the box reduces the flux from
7.4-7.7 to 3.8 at peak shear 0.2, the two seeds agreeing to 5%. The numbers and the plot are in
`docs/figures/flow_shear.py`.

Restart a sheared run from a saturated unsheared one (`[output] restart`): from
noise it may not ignite. It also needs a wide kx range, Nx >= 128 for Cyclone.
Both keys need `nonlinear = true` and refuse `[parallel]` sharding. There is no
parallel-velocity-gradient drive, and an imposed flow is static.

## Derivatives and optimization

Eigenvalue derivatives use `dλ/dp = wᴴ(dA/dp)v / (wᴴv)` with residual, overlap
and conditioning checks. Nonlinear heat-flux derivatives use a checkpointed
discrete adjoint over a finite post-saturation window. Compare these
derivatives with finite differences across perturbation sizes and window
lengths; assess a design change with independent long simulations
([nonlinear autodiff](docs/nonlinear_autodiff.rst)).

[VMEX](https://github.com/uwplasma/vmex) composes its equilibrium derivative
with GKX objectives in stellarator shape optimization; the
[optimization example](examples/10_vmex_optimization/run.py) combines heat flux
with equilibrium constraints. Quasilinear flux requires calibration and
held-out nonlinear validation before it supports an absolute transport
prediction ([quasilinear](docs/quasilinear.rst),
[stellarator optimization](docs/stellarator_optimization.rst)).

<img src="docs/_static/turbulence_loop.webp" width="720" alt="Saturated ITG turbulence on a Cyclone flux tube, shown as a perpendicular cut and as the field-aligned tube">

Saturated ITG turbulence on a Cyclone flux tube
([full-rate movie](https://github.com/uwplasma/GKX/releases/download/v1.7.0/gkx-cyclone-itg-turbulence.mp4)).

## Documentation and development

Documentation: [gkx.readthedocs.io](https://gkx.readthedocs.io). Start with
the [tutorial](docs/tutorials.rst) and the [guidance](docs/guidance.rst);
validation and cross-code protocols are in [validation](docs/validation.rst),
performance and parallel execution in [performance](docs/performance.rst) and
[parallelization](docs/parallelization.rst).

```bash
git clone https://github.com/uwplasma/GKX
cd GKX
pip install -e ".[dev]"
pytest -n 4
```

CI requires 95% combined line and branch coverage plus the physics, convergence, comparison,
differentiability and performance gates ([CONTRIBUTING.md](CONTRIBUTING.md)).
Cite GKX with [CITATION.cff](CITATION.cff). MIT license; see [LICENSE](LICENSE).
