# GKX

[![Release](https://img.shields.io/github/v/release/uwplasma/GKX?display_name=tag)](https://github.com/uwplasma/GKX/releases)
[![PyPI](https://img.shields.io/pypi/v/gkx.svg)](https://pypi.org/project/gkx/)
[![CI](https://github.com/uwplasma/GKX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/GKX/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/GKX/graph/badge.svg)](https://codecov.io/gh/uwplasma/GKX)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/gkx)](pyproject.toml)
[![Docs](https://readthedocs.org/projects/gkx/badge/?version=latest)](https://gkx.readthedocs.io)

JAX-native gyrokinetics for tokamak and stellarator flux tubes: linear
stability, nonlinear turbulence, and differentiable objectives on CPUs and GPUs.
GKX uses Hermite–Laguerre velocity moments and accepts analytic or VMEC/VMEX
geometry. Research claims are bounded by the [validated scope](#claim-scope).

<img src="docs/_static/turbulence_loop.webp" width="720" alt="Cyclone ITG turbulence: perpendicular cut and field-aligned tube">

A recorded Cyclone ITG simulation, shown in perpendicular and real-space views.
[Movie](https://github.com/uwplasma/GKX/releases/download/v1.7.0/gkx-cyclone-itg-turbulence.mp4)
· [reproduction](tools/artifacts/build_turbulence_movie.py).

## Start here

```bash
pip install gkx
gkx                 # self-contained linear Cyclone demonstration
gkx --help
```

Python 3.11+. For GPUs, install the appropriate
[JAX accelerator wheel](https://docs.jax.dev/en/latest/installation.html).
The no-input demo uses a deliberately coarse velocity grid: it checks the
workflow and agrees with its discrete eigenvalue within 1%, not a converged
gyrokinetic reference. It writes the resolved input, diagnostic tables and plot.

For checked-in examples and development:

```bash
git clone https://github.com/uwplasma/GKX
cd GKX
pip install -e ".[dev]"
gkx examples/linear/axisymmetric/cyclone.toml
gkx run-runtime-nonlinear \
  --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
  --steps 200 --out cyclone.out.nc
gkx plot cyclone.out.nc
```

The 200-step example is a short execution check, not a saturation guarantee.
Linear results report eigenvalue residuals; resolution convergence is a separate
test. More cases: [linear](examples/linear), [nonlinear](examples/nonlinear),
[optimization](examples/optimization), [teaching](examples/theory_and_demos).

## Run your equilibrium

```bash
gkx wout_circular_tokamak.nc --estimate   # explain the proposed grid
gkx wout_circular_tokamak.nc              # nonlinear ITG run
gkx plot wout_circular_tokamak/gkx.out.nc  # regenerate plots
```

[VMEX](https://github.com/uwplasma/vmex) and VMEC `wout` files use the same
entry point. The output directory contains `gkx.toml`, diagnostic and restart
NetCDF bundles, a summary, and flux/spectrum/field-line figures.
`--estimate=cautious` chooses a more conservative starting grid; neither
estimate replaces convergence in box size, spatial/velocity resolution and time.

Use [common_input.toml](examples/common_input.toml) as the configuration
template. [Inputs](docs/inputs.rst) explains all keys; [outputs](docs/outputs.rst)
explains units and artifacts. Important controls:

- `tprim`, `fprim`: normalized inverse temperature/density gradient lengths.
- `Nl`, `Nm`: Laguerre/perpendicular and Hermite/parallel velocity resolution.
- `gamma`, `omega`: linear growth rate and oscillation frequency.
- `Wphi`, `Wg`: field-energy and distribution/free-energy diagnostics,
  **not heat flux**; their precise definitions are in [theory](docs/theory.rst).
- `run_to="saturation"`: apply statistical stopping gates;
  `run_to="t_max"`: use a prescribed horizon.

A quiet field-energy trace does not establish stationary transport.
Adaptive-time averaging and sequential stopping still require qualification.
[Statistics and resolution](docs/numerics.rst) · [active plan](plan.md).

## Capabilities

| Capability | Implementation and boundary |
| --- | --- |
| Velocity moments | Hermite–Laguerre recurrences; closure/dissipation and truncation must converge for the chosen observable |
| Linear stability | Certified eigenpairs, matrix-free routes and linked twist-and-shift chains; [solvers](docs/solvers.rst) |
| Electrostatic and electromagnetic fields | Kinetic/adiabatic species and three-field equations; broad full-EM transport qualification remains open |
| Geometry | Analytic, Miller, VMEC/Boozer and VMEX paths; derivatives on supported smooth, fixed-topology controls |
| Collisions | LB/Dougherty, Sugama variants and bounded Coulomb projections; [species/moment/wavelength limits](docs/operators.rst) |
| Derivatives | Implicit eigenmode rules and checkpointed finite-window nonlinear adjoints |
| Parallel work | Independent scans and ensembles; production nonlinear domain decomposition is not established |
| Reproducibility | Resolved decks, restart bundles, [evidence ledger](docs/verification_matrix.rst) and explicit failure status |

The distribution state is `G[species, laguerre, hermite, ky, kx, z]`.
Perpendicular dynamics use Fourier transforms; the parallel grid follows a field
line. Streaming couples neighboring Hermite orders; mirror/drift terms couple
nearby moments. This structure does not make the full nonlinear system a
block-tridiagonal solve. [Equations](docs/theory.rst) · [methods](docs/algorithms.rst).

## From Python

```python
import jax.numpy as jnp
from gkx import CycloneBaseCase, LinearParams, integrate_linear_from_config
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry

cfg = CycloneBaseCase()
grid = build_spectral_grid(cfg.grid)
geometry = SAlphaGeometry.from_config(cfg.geometry)
state = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size),
                  dtype=jnp.complex64)
state = state.at[0, 0, 0, 0, :].set(1.0e-3)
trajectory, potential = integrate_linear_from_config(
    state, grid, geometry, LinearParams(), cfg.time
)
```

For repeated fixed-length runs, `gkx.prepare(case, steps=N)` prepares a case;
`warmup()` separates compilation from timed solves. This differs from adaptive
saturation stopping. See [API](docs/api.rst) and [examples](docs/examples.rst).

## Differentiate a nonlinear window

Given a detached saturated state, compatible grid, parameters, and a smooth
fixed-topology `geometry(shape)` function, the objective is:

```python
import gkx
import jax

def loss(shape):
    return gkx.nonlinear_heat_flux_window(
        saturated, grid, geometry(shape), params, dt, steps, terms=terms
    )

heat_flux, gradient = jax.value_and_grad(loss)(shape0)
```

This is the derivative of the prescribed discrete window at fixed initial state,
not a certified derivative of infinite-time mean transport. The complete
[nonlinear AD guide](docs/nonlinear_autodiff.rst) covers state preparation,
checkpointing and finite-difference checks.
[Eigenmode derivatives](docs/differentiable_eigensolver.rst) instead use implicit
eigenvalue rules and bordered solves for eigenvector observables.

![Checkpointed adjoint memory and derivative checks](docs/_static/nonlinear_autodiff_validation.png)

The recorded 1024-step Cyclone experiment reduces temporary state from
7.82 GB to 187 MB on CPU and 7.80 GB to 148 MB on GPU, at 1.92×/1.77× runtime.
This is a scoped checkpointing measurement, not a universal speed or memory bound.
[Recipe and horizon limits](docs/nonlinear_autodiff.rst).

### QA optimization

[QA_optimization.py](examples/optimization/QA_optimization.py) couples VMEX's
vacuum QA objectives to a prescribed GKX heat-flux window. Finite profile
gradients drive turbulence even when the equilibrium is vacuum.

![Initial and candidate QA equilibria](docs/_static/qa_transport_equilibria.png)
![Historical matched QA heat-flux traces and convergence](docs/_static/qa_transport_reduction.svg)

These initial/candidate comparisons are retained as historical, negative
qualification evidence. They predate the periodic hypercollision correction;
4 of 48 nominal traces fail final drift. **Statistically resolved QA transport
reduction is not established.** Promotion requires stationary individual traces,
correlation-aware uncertainty, resolved spectra, independent seeds and
grid/timestep convergence on the current operator.
[Conditions, scripts and results](docs/stellarator_optimization.rst).

## Validation

The table reports historical tracked linear-scan differences:
`100 * max|GKX − reference| / max|reference|`. It is not fresh certification of
the current operator; [damping-reference migration](https://github.com/uwplasma/GKX/issues/194)
and affected benchmark regeneration remain open.

| Case | `gamma` | `omega` |
| --- | ---: | ---: |
| KAW | 0.0004% | 0.051% |
| ETG | 0.040% | 0.074% |
| W7-X | 0.265% | 0.296% |
| HSX | 0.577% | 0.273% |
| Cyclone Miller | 5.51% | 1.25% |
| Cyclone ITG | 6.83% | 1.59% |
| **KBM** | **20.0%** | **11.1%** |

KBM remains an explicit discrepancy. CI recomputes these percentages from their
tracked sources; that is artifact consistency, not proof of matched physical
models. [Benchmarks](docs/benchmarks.rst) · [verification matrix](docs/verification_matrix.rst)
· [related codes](docs/codes.rst).

## Performance and parallelism

![Recorded runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

Compare at matched accuracy and uncertainty, including compilation, spin-up,
forward/gradient evaluation and peak memory. Smaller state storage does not
guarantee faster execution. [Measurements and reproduction](docs/performance.rst).

Independent `ky` scans and ensembles preserve serial ordering.
Sensitivity sweeps can use the same deterministic independent-work
reconstruction, but they need a dedicated matched scaling artifact before
speedup promotion. Whole-state nonlinear sharding remains a correctness/profiling
route, not a production scaling claim. [Parallelization](docs/parallelization.rst).

## Claim scope

- Quasilinear outputs are screening/ranking diagnostics,
  **not a runtime/TOML absolute-flux predictor**. The declared Solovev and
  shaped-pressure stress outliers remain outside the scoped model claim.
- Coulomb support is bounded by species, moments and wavelength; finite-`k`
  tables do not establish arbitrary-order or unlike-species Landau physics.
- W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron
  extensions are deferred.
- Full three-field EM transport, equilibrium ExB flow shear and production
  nonlinear domain decomposition remain open qualification goals.

[Release scope](docs/release_scope.rst) is the claim authority;
[plan.md](plan.md) is the execution authority. No new release is scheduled.

## Documentation and development

[Quickstart](docs/quickstart.rst) · [physics](docs/theory.rst) ·
[geometry](docs/geometry.rst) · [numerics](docs/numerics.rst) ·
[figure recipes](docs/manuscript_figures.rst) · [testing](docs/testing.rst) ·
[full documentation](https://gkx.readthedocs.io)

```bash
pytest
python tools/release/run_test_gates.py fast
ruff check .
python -m sphinx -W -b html docs docs/_build/html
```

Coverage is only one gate: physics, mathematical identities, convergence,
derivatives and performance need separate evidence.
Bug reports and reproducible problem decks are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md) for supported dependencies and review rules.
GKX is distributed under the [MIT License](LICENSE).
