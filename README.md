# GKX

[![Release](https://img.shields.io/github/v/release/uwplasma/GKX?display_name=tag)](https://github.com/uwplasma/GKX/releases)
[![PyPI](https://img.shields.io/pypi/v/gkx.svg)](https://pypi.org/project/gkx/)
[![CI](https://github.com/uwplasma/GKX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/GKX/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/GKX/graph/badge.svg)](https://codecov.io/gh/uwplasma/GKX)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![Docs](https://readthedocs.org/projects/gkx/badge/?version=latest)](https://gkx.readthedocs.io)

GKX is a JAX-native gyrokinetic solver for linear stability,
nonlinear turbulence, differentiable analysis, and stellarator design. It uses
Fourier perpendicular coordinates, a Hermite-Laguerre velocity basis, and
field-aligned analytic, Miller, or VMEC geometry. The package runs on CPUs and
GPUs, exposes a Python API for autodiff and optimization, and provides a simple
executable for routine simulations.

## Installation

```bash
pip install gkx
```

For development:

```bash
git clone https://github.com/uwplasma/GKX
cd GKX
pip install -e .
```

## Quickstart

Run the built-in linear initial-value example:

```bash
gkx
```

The equivalent `gkx` entry point is also installed. The default run
prints setup, progress, elapsed time, and ETA, then writes its input, summary,
time series, eigenfunction, and a two-panel plot in the current directory.

Run a checked-in case or plot an existing result:

```bash
gkx examples/linear/axisymmetric/cyclone.toml
gkx run-runtime-nonlinear \
  --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
  --steps 200 --out cyclone.out.nc
gkx --plot cyclone.out.nc
```

Generate the small VMEC equilibria used by the self-contained examples:

```bash
pip install vmex
cd examples/vmec
./generate_wouts.sh
```

Full documentation is hosted at **[gkx.readthedocs.io](https://gkx.readthedocs.io)**.
Start with the [quickstart](https://gkx.readthedocs.io/en/latest/quickstart.html) and
[input reference](https://gkx.readthedocs.io/en/latest/inputs.html) for linear,
nonlinear, Miller, VMEC, restart, quasilinear, and plotting workflows.

## Highlights

- Electrostatic and electromagnetic gyrokinetics with kinetic or Boltzmann species.
- Linear initial-value, dominant-eigenmode, and nonlinear turbulence solvers.
- Analytic s-alpha, Miller, imported VMEC, and differentiable VMEC/Boozer geometry.
- JAX JIT, forward/reverse autodiff, implicit eigenvalue derivatives, and UQ tools.
- Quasilinear transport diagnostics with explicit saturation-rule metadata.
- CPU/GPU execution and production parallelization for independent scans and ensembles.
- Restartable NetCDF output and `gkx --plot` publication-style figures.
- Five selectable collision operators, from a conserving Lenard-Bernstein model
  to the full linearized Coulomb (Landau) operator with finite-Larmor-radius
  effects.

## What GKX Solves

The gyrokinetic equation for the perturbed distribution of each species,
expanded in a **Hermite-Laguerre** velocity basis. Writing the gyrocenter
distribution as

```
delta f_s = F_Maxwellian * sum_{m,l}  G_s^{m,l}  psi_m(v_par / v_th)  L_l(mu B / T)
```

with `psi_m = H_m / sqrt(2^m m! sqrt(pi))` the normalized Hermite functions and
`L_l` the Laguerre polynomials. This turns velocity space into two spectral
indices: `m` resolves parallel dynamics (Landau damping, parallel heat flux),
`l` resolves perpendicular dynamics (FLR effects, trapping). The evolved state
is a single array

```
G[species, laguerre l, hermite m, ky, kx, z]
```

and each physical effect becomes a specific coupling on it:

| Term | What it does to `G` | Set by |
| --- | --- | --- |
| Parallel streaming | couples `m` to `m±1` (a ladder in Hermite index) | geometry `gradpar` |
| Magnetic mirror | couples `m` and `l` together | `bgrad` |
| Curvature / grad-B drift | multiplies by `i(k · v_d)` | geometry curvature |
| Diamagnetic drive | injects free energy from the gradients | `[[species]] tprim`, `fprim` |
| Collisions | couples moments within a species | `collision_operator` |
| Nonlinearity | `E × B` convolution in `(kx, ky)`, pseudo-spectral | nonlinear solver |
| Field solve | quasineutrality + parallel Ampere for `phi`, `A_par`, `B_par` | `beta`, species list |

Perpendicular directions are Fourier (`kx`, `ky`); the parallel direction `z`
follows a field line (flux tube). Electrons are kinetic or Boltzmann.

**Why a moment basis:** `m` and `l` are the same kind of index as `kx` and
`ky`, so the whole problem is dense linear algebra on one array — which is what
a GPU is good at. The cost is that the parallel ladder must be terminated
somewhere, which is what the closure section below is about.

## Geometry

| `[geometry] model` | Gives you | Needs |
| --- | --- | --- |
| `"s-alpha"` | circular tokamak, `B = B0/(1 + eps cos theta)` | `q`, `s_hat`, `epsilon`, `R0` |
| `"slab"` | uniform field, sharpest numerics tests | grid only |
| `"imported-eik"` / `"vmec-eik"` | Miller or full 3D stellarator from a file | `geometry_file` |

Miller equilibria and VMEC/Boozer flux tubes are also built in-process through
the Python API, where the metric coefficients stay differentiable — that is the
path stellarator shape optimization uses.

## Velocity Resolution, Closures and Recurrence

Truncating the Hermite ladder at `m = M` is not free. The free-energy pulse
streams up in `m`, reaches the end and comes *back*: **recurrence**, at

```
t_rec ~ 2 sqrt(M) / (k_par v_th)
```

Nothing after `t_rec` is physics. Because `t_rec` grows only as `sqrt(M)`,
adding moments is a weak fix; the end of the ladder has to absorb instead.

| Closure | What happens at `m = M` | Revived `|g_0|` (initial = 1) | Error on resolved window |
| --- | --- | --- | --- |
| Truncation (default) | `G_{M+1} = 0` — a reflecting wall | **0.999** | **1.0** |
| Hypercollisions | scale-selective damping over `m > 2` | 0.0002 | 2e-4 |
| Reflectionless | outgoing condition, `G_{M+1} = -i sgn(k_par) R G_M` | 0.019 | 2e-2 |

Measured on the free-streaming hierarchy at `N_m = 64`, against a converged
`N_m = 1024` reference. Truncation reflects the pulse back **essentially
intact** — it provides no dissipation whatsoever, and everything after `t_rec`
is an artifact.

The reflectionless coefficient `R = M/sqrt(2(M+1)) * Gamma(M/2)/Gamma((M+1)/2)`
(Kanekar et al., JPP **81**, 305810104 (2015), Eq. 4.36) tends to 1 as `M` grows,
so absorption becomes exact with resolution instead of needing retuning. At
`M = 3` it reproduces the Hammett-Perkins three-pole coefficient exactly, which
is an independent check on the family.

On these measurements a **hypercollision at the GX Appendix-B normalization
outperforms the reflectionless closure on both metrics**. The closure's
advantages are that it has no free parameter and touches only `m = M`, so it
cannot bias resolved moments; it is not a drop-in improvement on a well-tuned
hypercollision. Hypercollisions remain the shipped default
(`[physics] hypercollisions = true`); the reflectionless closure is reached
through the Python API
(`linked_streaming_contribution(..., hermite_closure="reflectionless")`).

![Recurrence and Hermite closure](docs/_static/recurrence_hermite_closure.png)

## Configuration

One TOML file. Every key has a default, so a working input is short — the
sections you actually touch most days are the first five.

| Section | Controls | Common keys |
| --- | --- | --- |
| `[[species]]` | one block per species | `charge`, `mass`, `temperature`, `tprim`, `fprim`, `nu`, `kinetic` |
| `[grid]` | resolution and box | `Nx`, `Ny`, `Nz`, `Lx`, `Ly`, `boundary` |
| `[geometry]` | equilibrium | `model`, `q`, `s_hat`, `epsilon`, `R0`, `geometry_file` |
| `[time]` | integration | `t_max`, `dt`, `method`, `collision_operator` |
| `[physics]` | what to include | `linear`, `nonlinear`, `electrostatic`, `adiabatic_electrons`, `tau_e` |
| `[init]` | initial condition | `init_field`, `init_amp`, `gaussian_width` |
| `[collisions]` | collision and hypercollision rates | `nu_hermite`, `nu_laguerre`, `nu_hyper`, `p_hyper` |
| `[terms]` | switch individual terms on/off (0/1) | `streaming`, `mirror`, `curvature`, `diamagnetic`, `nonlinear` |
| `[run]`, `[scan]` | single run / `k_y` scan resolution | `ky`, `Nl`, `Nm`, `solver` |
| `[normalization]` | benchmark normalization contract | `contract`, `diagnostic_norm` |
| `[fit]` | growth-rate fit window | `auto_window`, `window_method` |

`[terms]` is the debugging lever: setting one coefficient to `0.0` removes
exactly that term, which is how most of the physics gates isolate what they
test.

Full key-by-key reference:
[inputs](https://gkx.readthedocs.io/en/latest/inputs.html).

## Main Validation Results

The release atlas compares growth rates, frequencies, eigenfunctions, and
nonlinear transport windows with established gyrokinetic reference results.
Promoted cases include Cyclone ITG, Cyclone Miller, KBM, W7-X, and HSX, with
ETG and kinetic-electron stress cases kept at their documented claim level.

![Linear and nonlinear benchmark summary](docs/_static/benchmark_readme_panel.png)

The exact equations, normalization, grids, boundary conditions, diagnostic
windows, tolerances, and artifact provenance are in the
[benchmark documentation](docs/benchmarks.rst) and
[verification matrix](docs/verification_matrix.rst). A visual overlay alone is
not treated as parity evidence.

## Collision Operators

GKX ships five collision models spanning the full hierarchy, selected with one
TOML key or one argument to the Python factory:

| `collision_operator` | Model | Reference |
| --- | --- | --- |
| `none` / `lenard_bernstein` | Conserving diagonal Lenard-Bernstein/Dougherty relaxation | built in |
| `sugama` | Drift-kinetic Sugama, conservative by construction | Frei, Ernst & Ricci (2022), Eqs. (C6a)-(C6f) |
| `improved_sugama` | Improved Sugama, corrected Pfirsch-Schlüter friction | Sugama et al. (2019); Frei, Ernst & Ricci (2022) |
| `coulomb` | Drift-kinetic linearized Coulomb (Landau) | Frei, Ernst & Ricci (2022), Eqs. (C9a)-(C9f) |
| `coulomb_finite_kperp` | Gyrokinetic Coulomb retaining finite `k_perp` | Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021), Eqs. (3.47)-(3.50) |

```toml
[time]
collision_operator = "coulomb_finite_kperp"
```

The models agree in the collisionless limit and separate as collisionality
rises, with Lenard-Bernstein over-damping and finite-Larmor gyroaveraging
weakening the collisional damping relative to the drift-kinetic operator:

![Collision operator comparison](docs/_static/collision_operator_comparison.png)

Reproduce with
[`examples/theory_and_demos/collision_operator_comparison.py`](examples/theory_and_demos/collision_operator_comparison.py)
(`--nu-scan` draws the figure), or run
[`examples/linear/axisymmetric/cyclone_coulomb_collisions.toml`](examples/linear/axisymmetric/cyclone_coulomb_collisions.toml)
through the executable.

### How GKX compares

GKX shares its Hermite-Laguerre gyro-moment velocity representation with GX,
which is what makes GX the closest parity reference. The differences that
matter for verification are the collision hierarchy and the differentiable
geometry path:

| | GKX | GX | GENE |
| --- | --- | --- | --- |
| Velocity space | Hermite-Laguerre moments | Hermite-Laguerre moments | grid in `(v_par, mu)` |
| Collision models | 5, through gyrokinetic Coulomb | Dougherty + hypercollisions | Landau and model operators |
| Differentiable | JAX autodiff end to end | not a design goal | not a design goal |

This records scope, not quality — both codes are mature and each is stronger
than GKX in areas GKX does not attempt. A well-converged Dougherty operator can
also be a better physical answer than an unconverged Coulomb one; see
[related codes](docs/codes.rst) for the qualifications.

### How the operators are verified

Every shipped matrix is checked against the published closed forms rather than
only against itself. All twelve Appendix-C coefficients reproduce exactly, and
the structural properties a linearized collision operator must satisfy are
gated numerically:

| Property | Result |
| --- | --- |
| Density, parallel-momentum, energy conservation | machine precision (≤ 2.2e-16) |
| H-theorem (negative semidefinite) | holds for every model |
| Onsager self-adjointness | exact (≤ 3.4e-17) |
| Published Appendix-C coefficients | reproduce to 1e-12 |
| Finite-Larmor `b -> 0` limit | reduces to the drift-kinetic operator exactly |
| Finite-Larmor conservation defect | scales as `B^1.96`-`B^1.99`, i.e. first order in `b` |

The finite-Larmor operator acts on gyrocenter moments, whose conservation is
modified by gyroaveraging, so the *ordering* is the test: the defect must vanish
at `b = 0` and enter at first order in `b`. Tables ship at 8 and 18
Hermite-Laguerre moments, generated in 60-digit arithmetic and stored as
checksummed float64.

The Coulomb tables are generated for like-species collisions; a multispecies
request is refused rather than silently extrapolated. Equations, thresholds,
machine-readable results, literature links, and the reproduction recipe are in
the [collision-operator documentation](docs/operators.rst).

![Coulomb collision operator verification](docs/_static/collision_operator_verification.png)

![Paper-resolution collisional zonal response](docs/_static/collision_finite_wavelength_zonal_response.png)

At ``(P,J)=(24,10)``, the drift-kinetic traces approach the Xiao residual and
the finite-wavelength tails reproduce the published original < improved <
Coulomb ordering at both ``kx rho_i=0.1`` and ``0.2``. The improved model is
also closer to Coulomb over ``t nu_ii <= 10`` at both wavenumbers. Equations,
velocity-space convergence, compact replay data, and the Figure 12--14 gate
are documented in [Operators and Terms](docs/operators.rst).

## Runtime and Memory

![Runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

Measured cold wall time and peak memory, CPU and GPU.

| Row type | What it includes |
| --- | --- |
| Cold JAX | startup + compilation |
| Prepared | compiled once, geometry and numerics fixed |

Prepared throughput depends on the software stack and GPU state. Details in
[performance](docs/performance.rst).

## Differentiable Python API

```python
import jax.numpy as jnp

from gkx import CycloneBaseCase, LinearParams, integrate_linear_from_config
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry

cfg = CycloneBaseCase()
grid = build_spectral_grid(cfg.grid)
geometry = SAlphaGeometry.from_config(cfg.geometry)
parameters = LinearParams()
state = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
state = state.at[0, 0, 0, 0, :].set(1.0e-3)
trajectory, potential = integrate_linear_from_config(
    state, grid, geometry, parameters, cfg.time
)
```

For repeated nonlinear calls with fixed geometry and numerical policy, prepare
the compiled simulation once:

```python
from gkx.solvers.nonlinear.diagnostic_integration import prepare_nonlinear_explicit_diagnostics

simulation = prepare_nonlinear_explicit_diagnostics(
    initial_state,
    grid,
    geometry,
    parameters,
    dt=0.02,
    steps=400,
    resolved_diagnostics=False,
)
time, diagnostics, final_state, fields = simulation.run()
```

The prepared object accepts another same-shape initial state without rebuilding
the scan. A matched rebuilt cache/parameter PyTree can also remain dynamic for
autodiff; geometry layout is fixed, and dynamic-geometry compile reuse remains
an active differentiability lane.

The planted two-mode inverse problem below recovers two gradient parameters and
checks the autodiff Jacobian against finite differences. The single-mode demo in
the docs intentionally demonstrates non-identifiability rather than exact
parameter recovery.

![Two-mode autodiff inverse validation](docs/_static/autodiff_inverse_twomode.png)

See [differentiable geometry](docs/geometry.rst),
[algorithms](docs/algorithms.rst), and [stellarator optimization](docs/stellarator_optimization.rst)
for JVP, VJP, implicit differentiation, conditioning, covariance, and finite-difference gates.

## Quasilinear Modeling

![Stellarator quasilinear usefulness](docs/_static/quasilinear_stellarator_usefulness.png)

**Use it for:** ranking and correlation studies, optimization screening.
**Not for:** absolute heat flux — it is not a runtime/TOML absolute-flux
predictor. Absolute-flux promotion stays rejected while the declared Solovev and
shaped-pressure stress outliers are retained.

Derivations, calibration splits, uncertainty and holdout gates:
[quasilinear docs](docs/quasilinear.rst).

## QA ITG Optimization

Objective = aspect ratio + mean iota + quasisymmetry, plus one GKX residual
(growth rate, quasilinear, or nonlinear window). Baseline is the max-mode-5 QA
workflow; all transport comparisons use solved VMEC equilibria.

![VMEX QA max-mode-5 optimizer sweep](docs/_static/vmex_qa_full_sweep_panel.png)

**These rows are not promoted turbulent-flux designs.** Their matched long
post-transient nonlinear audits use converged post-transient heat-flux windows
and show no statistically significant transport reduction against the strict QA
baseline — useful negative evidence for objective conditioning and optimizer
choice.

The RBC(1,1) scan below is a landscape and noise diagnostic, not a source of
admitted candidates.

![QA RBC(1,1) transport landscape](docs/_static/vmec_boundary_transport_landscape_rbc11_full.png)

Scripts: [examples/optimization](examples/optimization). Objective equations,
optimizer policies and long-window audits:
[optimization docs](docs/stellarator_optimization.rst).

## Parallelization

| Status | Covers |
| --- | --- |
| Production | independent `k_y` scans, quasilinear/UQ ensembles, file-backed tasks — deterministic ordering, serial-identity gated |
| Needs a scaling artifact | sensitivity sweeps |
| Diagnostic only | nonlinear whole-state and domain decomposition |

Sensitivity sweeps can use the same deterministic independent-work
reconstruction, but they need a dedicated matched scaling artifact before any
speedup claim is promoted. Nonlinear speedup is not claimed until species-first
and Hermite-second decomposition, Hermite halo exchange, field-moment
collectives, and transport-window identity all pass.

Details: [parallelization](docs/parallelization.rst).

## Current Claim Scope

Validated release claims are bounded by the [release scope](docs/release_scope.rst):

- Standard electrostatic/electromagnetic full gyrokinetics is validated only on
  the promoted cases and observables in the verification matrix.
- Quasilinear outputs are diagnostics and screening models, not universal
  absolute nonlinear heat-flux predictions.
- Nonlinear optimization evidence requires matched, replicated, long
  post-transient windows; startup or reduced envelopes are not production evidence.
- W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron extensions are deferred.
- The Sugama, improved-Sugama, and Coulomb operators are verified against their
  published closed forms and structural invariants, and are validated for
  like-species collisions; species-coupled Coulomb coefficients remain open.
- Collision operators run on the fixed-step cached integrator. The diffrax,
  sharded, and Krylov eigenvalue paths reject them rather than silently
  substituting the built-in diagonal term.
- Production nonlinear domain decomposition and equilibrium ExB flow shear
  remain open.

## Examples and Documentation

The repository keeps small runnable examples under:

- [`examples/linear`](examples/linear): axisymmetric and stellarator linear runs.
- [`examples/nonlinear`](examples/nonlinear): nonlinear turbulence and restarts.
- [`examples/optimization`](examples/optimization): differentiable QA workflows.
- [`examples/theory_and_demos`](examples/theory_and_demos): numerical and autodiff demonstrations.
- [`benchmarks`](benchmarks): comparison inputs, drivers, and compact result indexes.

Detailed user and developer documentation:

- [Physics and equations](docs/theory.rst)
- [Operators and models](docs/operators.rst)
- [Numerics and solvers](docs/numerics.rst)
- [Geometry](docs/geometry.rst)
- [Outputs and plotting](docs/outputs.rst)
- [Testing and validation](docs/testing.rst)
- [Code structure](docs/code_structure.rst)
- [Release and research scope](docs/release_scope.rst)

## Testing

```bash
pytest
python tools/release/run_test_gates.py fast
python tools/release/run_test_gates.py wide-coverage \
  --shards 48 --timeout 300 --fail-under 95 \
  --pytest-arg=-o --pytest-arg=addopts= --pytest-arg=-m --pytest-arg="not slow"
python -m sphinx -W -b html docs docs/_build/html
```

The package-wide CI coverage gate is at least 95%. Physics, convergence,
comparison, differentiability, and performance gates are required in addition
to line coverage.

## License

GKX is distributed under the [MIT License](LICENSE).
