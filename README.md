# GKX

[![Release](https://img.shields.io/github/v/release/uwplasma/GKX?display_name=tag)](https://github.com/uwplasma/GKX/releases)
[![PyPI](https://img.shields.io/pypi/v/gkx.svg)](https://pypi.org/project/gkx/)
[![CI](https://github.com/uwplasma/GKX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/GKX/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/GKX/graph/badge.svg)](https://codecov.io/gh/uwplasma/GKX)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![Docs](https://readthedocs.org/projects/gkx/badge/?version=latest)](https://gkx.readthedocs.io)

GKX is a **JAX-native gyrokinetic solver** for tokamaks and stellarators: linear
stability, nonlinear turbulence, and differentiable analysis that plugs straight
into stellarator optimization. It runs on CPUs and GPUs, from a one-line command
or from Python.

```bash
pip install gkx && gkx
```

<img src="docs/_static/turbulence_loop.webp" width="900" alt="Saturated ITG turbulence on a Cyclone flux tube, shown as a perpendicular cut and as the field-aligned tube">

**Saturated ITG turbulence on a Cyclone flux tube** (32×32×24, `t ≈ 300–376`),
shown as the perpendicular cut a gyrokineticist reads and as the field-aligned
tube in real space — the same data twice, because a flux-tube movie that only
shows the perpendicular plane hides the parallel elongation that defines the
turbulence. Amplitude is steady across the whole loop, not growing.

The loop is an animated image rather than a `<video>`, which GitHub strips from
Markdown: 60 frames at 10 fps, 828 kB of animated WebP, chosen over GIF because
no GIF fit the repository's 1 MB per-file limit at a resolution worth showing.
The
**[full-rate movie](https://github.com/uwplasma/GKX/releases/download/v1.7.0/gkx-cyclone-itg-turbulence.mp4)**
(1.8 MB, 120 frames at 20 fps) is a release asset. Regenerate the physics with
[`build_turbulence_movie.py`](tools/artifacts/build_turbulence_movie.py). For a
physics run, continue its saved saturated state with the same method and CFL
policy; only the two displayed cuts are retained:

```bash
python tools/artifacts/build_turbulence_movie.py CASE.toml \
  --initial-state saturated_state.npz --snapshots movie_cuts.npz
python tools/artifacts/build_turbulence_movie.py --render-from movie_cuts.npz \
  --output turbulence.mp4
```

Re-encode the compact README loop from that MP4 with:

```bash
mkdir -p frames
ffmpeg -i gkx-cyclone-itg-turbulence.mp4 \
  -vf "fps=10,scale=1100:-1:flags=lanczos" frames/f_%04d.png
img2webp -loop 0 -lossy -q 60 -m 6 -d 100 frames/f_*.png \
  -o docs/_static/turbulence_loop.webp
```

## Why GKX

| | |
| --- | --- |
| **Collisions other codes don't have** | Five operators, from Lenard-Bernstein up to the **full gyrokinetic Coulomb** with finite-Larmor-radius effects — selected with one TOML key |
| **Differentiable end to end** | JAX autodiff through geometry, solver and diagnostics, including implicit eigenvalue derivatives — real gradients for stellarator shape optimization |
| **Resolutions a dense solver can't hold** | Matrix-free eigenmodes at `n = 494,592`, where the dense operator alone would be **3.6 TiB** — with the eigenpair still differentiable |
| **Verified against exact physics** | Landau roots to 0.004%, conservation to machine precision, published Appendix-C coefficients to 1e-12 |
| **Fast where it matters** | GPU execution, restartable NetCDF output, publication figures from `gkx --plot` |

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

## Differentiable matrix-free eigenmodes

Design studies need a few physical modes and their derivatives, not a dense
spectrum. GKX applies the full gyrokinetic RHS inside a restarted eigensolver, so
storage is `O(n m)` rather than `O(n²)`.

![Matrix-free reach](docs/_static/eigensolver_reach.png)

The dense path is bounded by **memory, not speed** — at the largest tested
truncation (`n = 494,592`) its operator alone would be 3.6 TiB, so it cannot
represent the problem at any speed.

```python
settings = gkx.AdaptiveLinearEigensolverConfig(tolerance=1e-9, candidate_count=2)

def objective(boundary):
    values = gkx.solver_objective_vector_from_geometry(
        build_solver_geometry(boundary),
        n_laguerre=16, n_hermite=24,
        eigensolver="adaptive-propagator", adaptive_config=settings,
    )
    return values[-1]          # quasilinear transport objective

value, gradient = jax.value_and_grad(objective)(initial_shape)
```

Reverse mode uses `dλ/dp = wᴴ(dA/dp)v / (wᴴv)` plus a bordered solve for
eigenvector observables — no differentiation through the iteration. Residual,
overlap, spectral-gap and conditioning gates reject ambiguous modes.

| | Memory | Branch handling | Derivative |
| --- | ---: | --- | --- |
| Dense eigensolve | `O(n²)` | all modes at small `n` | dense eigenvector AD |
| Initial-value fit | `O(n)` | can switch at crossings | long-trajectory AD |
| **GKX** | **`O(n m)`** | **certified candidates + continuation** | **implicit JAX VJP** |

Opt-in: the default stays dense so established results are unchanged. Measured
boundaries, the sparse fallback, the physics-aware shift inverse and what is
*not* claimed are in the [eigensolver documentation](docs/differentiable_eigensolver.rst);
the inner-solver choice behind it is in [numerical defaults](docs/solvax_defaults.rst).



## Validation

Every figure below is anchored to something external — an exact root, a
published coefficient, or a tracked reference run — and regenerates from a
script in [`tools/artifacts`](tools/artifacts).

### Landau damping against the exact kinetic roots

GKX's own linear operator, extrapolated to zero collisionality, against the
roots of `1 + T_i/T_e + zeta Z(zeta) = 0` solved to double precision:

| | exact | GKX | error |
| --- | --- | --- | --- |
| `T_e/T_i = 1`, `omega` | 2.045904866 | 2.047220793 | 0.064% |
| `T_e/T_i = 1`, `gamma` | -0.851330459 | -0.849234188 | 0.246% |
| `T_e/T_i = 10`, `omega` | 3.728834801 | 3.728993838 | 0.004% |
| `T_e/T_i = 10`, `gamma` | -0.058337421 | -0.058339802 | 0.004% |

![Landau damping validation](docs/_static/landau_damping_validation.png)

The measurement is harder than it looks, and the figure shows why: a
collisionless truncated Hermite system has a **purely real spectrum** (verified
to 3e-14), so it cannot Landau damp at all — what looks like damping is a
transient ending at recurrence. The root is not an eigenvalue either; it is a
pole reached by `nu -> 0` extrapolation.

### Linear benchmark parity

![Linear benchmark parity](docs/_static/benchmark_linear_parity.png)

Max relative difference against tracked reference results: ETG and KAW below
0.1%, W7-X and HSX below 0.6%, Cyclone ITG 6.8%, **KBM 20%** — the KBM case is
the known outlier and is documented at its claim level rather than smoothed
over. Per-case detail: [benchmarks](docs/benchmarks.rst) and the
[verification matrix](docs/verification_matrix.rst).

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

Every shipped matrix is checked against the published closed forms, not only
against itself:

| Property | Result |
| --- | --- |
| Density, parallel-momentum, energy conservation | machine precision (≤ 2.2e-16) |
| H-theorem (negative semidefinite) | holds for every model |
| Onsager self-adjointness | exact (≤ 3.4e-17) |
| Published Appendix-C coefficients | reproduce to 1e-12 |
| Finite-Larmor `b -> 0` limit | reduces to the drift-kinetic operator exactly |
| Finite-Larmor conservation defect | scales as `B^1.96`-`B^1.99`, first order in `b` |

![Coulomb collision operator verification](docs/_static/collision_operator_verification.png)

The finite-Larmor operator acts on gyrocenter moments, whose conservation is
modified by gyroaveraging, so the *ordering* is the test: the defect must vanish
at `b = 0` and enter at first order. Coulomb tables are generated for
like-species collisions; a multispecies request is refused rather than silently
extrapolated. Equations, thresholds, convergence panels and the reproduction
recipe: [operators](docs/operators.rst).


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

## Runtime and Memory

![Runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

Cold wall time and peak memory across the tracked cases. Cold times include JAX
startup and compilation, which is the right number for "how long does a run
take" and the wrong one for kernel speed.

GKX CPU→GPU speedup spans **0.7× to 13.1×** depending on the case — the small
linear cases are dominated by startup, so the GPU can be slower. Warm timings
and profiler artifacts: [performance](docs/performance.rst).

### Run to saturation

The cheapest way to make a nonlinear run faster is to not integrate past the
point where the answer has stopped changing. Diagnosed nonlinear runs therefore
stop at saturation by default (`[time] run_to = "saturation"`). GKX integrates
in chunks and, after each one, measures the heat-flux trace with the spin-up
phase excluded: it stops when the autocorrelation-corrected relative SEM of the
windowed mean falls to `saturation_rel_sem` (default 5%), the window is long
enough (`saturation_min_window`, ten autocorrelation times when unset), and the
two halves of the window agree within twice their combined SEM. `t_max` stays
the hard cap, so nothing is lost if the run never saturates, and the summary
reports the window it averaged over together with `mean ± SEM` — the number you
would have computed by hand afterwards. Set `run_to = "t_max"` or pass
`--no-until-saturated` for a fixed horizon.

GX has no equivalent: it runs a fixed `nstep`/`t_max` and can only be halted
early by dropping a `.stop` file next to the run, which is a manual
intervention rather than a convergence criterion (`src/run_gx.cu:128`,
`src/diagnostics.cu:319-324`).

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

## Velocity resolution and recurrence

Truncating the Hermite ladder at `m = M` makes the end of it a **reflecting
wall**: free energy streams up in `m`, hits the wall, and returns as
*recurrence* at

```
t_rec ~ 2 sqrt(M) / (k_par v_th)
```

Nothing after `t_rec` is physics, and because `t_rec` grows only as `sqrt(M)`,
adding moments is a weak fix — the ladder has to absorb instead. Measured on the
free-streaming hierarchy, a hard truncation returns **99.9% of the initial
amplitude** (it dissipates nothing), while hypercollisions cut that to 0.0002.

Hypercollisions are the default and are what you want. GKX also ships an opt-in
**reflectionless closure** (Kanekar et al., JPP **81**, 305810104 (2015)) whose
coefficient tends to 1 with resolution, so it needs no tuning and touches only
`m = M`; on measurement it does not beat a well-tuned hypercollision.

Full derivation, both metrics, resolution scans and the closure coefficient:
[numerics](docs/numerics.rst).

## Quasilinear Modeling

![Stellarator quasilinear usefulness](docs/_static/quasilinear_stellarator_usefulness.png)

**Use it for:** ranking and correlation studies, optimization screening.
**Not for:** absolute heat flux — it is not a runtime/TOML absolute-flux
predictor. Absolute-flux promotion stays rejected while the declared Solovev and
shaped-pressure stress outliers are retained.

Derivations, calibration splits, uncertainty and holdout gates:
[quasilinear docs](docs/quasilinear.rst).

## Nonlinear autodiff and QA optimization

GKX differentiates one production nonlinear objective: the physical heat flux
averaged over a post-saturation RK window. A block-checkpointed discrete adjoint
stores `O(sqrt(N))` distribution states and works on CPU and GPU.

```python
def loss(shape):
    return gkx.nonlinear_heat_flux_window(
        saturated, grid, geometry(shape), params, dt, steps, terms=terms
    )

heat_flux, gradient = jax.value_and_grad(loss)(shape0)
```

![Nonlinear adjoint memory and derivative validation](docs/_static/nonlinear_autodiff_validation.png)

**One derivative, bounded memory.** On one host, one 16x16x16 Cyclone case and
one 1024-step window, block checkpointing cuts the measured temporary state from
7.82 GB to 187 MB on 36 CPU cores and from 7.80 GB to 148 MB on an RTX A4000,
for 1.9x and 1.8x more runtime. The discrete adjoint and centered finite
differences agree to 1e-11 through 1024 steps and part at 2048, where chaotic
trajectory separation sets the useful window length; longer windows warn.
The same gradient agrees between CPU and an A4000 to 1.5e-15.
[Regenerate every number](docs/nonlinear_autodiff.rst).

The single [`QA_optimization.py`](examples/optimization/QA_optimization.py)
follows VMEX's vacuum QA mode ladder and adds this heat flux as a fourth tuple.
Finite `a/L_T=3` and `a/L_n=1` drive ITG turbulence. The analytic Jacobian
composes VMEX's implicit equilibrium derivative with the exact GKX window
derivative.

![Initial and optimized QA equilibria](docs/_static/qa_transport_equilibria.png)

**A small shape step with a resolved transport effect.** Eight low-order
boundary coefficients move; aspect ratio and mean iota change by less than
0.05%, while the 3-D LCFS and LCFS Boozer `|B|` show where the equilibrium
changes. The QA residual remains `O(10^-3)`; all panels use the same
`|B|/<|B|>` color scale.

The saturation state is detached and refreshed after accepted stages. The
window is a local design derivative. Independent matched runs validate the
accepted direction with replicated saturated trajectories.

![Matched QA heat-flux traces and convergence](docs/_static/qa_transport_reduction.svg)

**The startup spike is excluded; the shaded window is measured.** Across 24
nominal matched seeds, the result is a **12.26% reduction** (95% CI
10.64--13.88%).
The stationary 24x24 and `(Nl,Nm)=(6,12)` refinements give 8.50%
(6.34--10.66%) and 12.32% (9.62--15.03%); all 16 pairs improve in both. The
orange cross is the short 24x24 pilot rejected by its stationarity test.

This is one vacuum QA surface and field line, not a universal transport claim;
broader claims still require converged post-transient heat-flux windows across
surfaces and field lines. See the concise [autodiff
mathematics](docs/nonlinear_autodiff.rst) and the [equations, scripts, matched
statistics, and resolution study](docs/stellarator_optimization.rst).


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

## Full feature list

- Electrostatic and electromagnetic gyrokinetics with kinetic or Boltzmann species.
- Linear initial-value, dominant-eigenmode, and nonlinear turbulence solvers.
- Matrix-free eigenmodes with certified residuals, branch continuation, and
  implicit derivatives — `O(n m)` storage instead of `O(n²)`.
- Analytic s-alpha, Miller, imported VMEC, and differentiable VMEC/Boozer geometry.
- JAX JIT, forward/reverse autodiff, implicit eigenvalue derivatives, and UQ tools.
- Quasilinear transport diagnostics with explicit saturation-rule metadata.
- CPU/GPU execution and production parallelization for independent scans and ensembles.
- Restartable NetCDF output and `gkx --plot` publication-style figures.
- Five selectable collision operators, from a conserving Lenard-Bernstein model
  to the full linearized Coulomb (Landau) operator with finite-Larmor-radius
  effects.

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
- [Matrix-free eigenmodes](docs/differentiable_eigensolver.rst) and the
  [numerical defaults](docs/solvax_defaults.rst) behind them
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
