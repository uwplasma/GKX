# GKX

**Gyrokinetic turbulence and sensitivities in JAX.**
GKX uses Hermite–Laguerre velocity moments and field-aligned flux tubes
for linear stability, nonlinear transport, and stellarator-design studies
on CPUs and GPUs.

<p align="center">
  <img src="docs/_static/turbulence_loop.webp" width="720" alt="Gyrokinetic turbulence evolving in a flux tube">
</p>

[Documentation](https://gkx.readthedocs.io) ·
[Examples](docs/examples.rst) · [Validation](docs/verification_matrix.rst) ·
[Development plan](plan.md)

- **Physics:** electrostatic and electromagnetic equations; adiabatic or kinetic
  species; analytic, Miller, and VMEC geometry. Validation is case-specific.
- **Derivatives:** implicit eigenmode sensitivities and checkpointed,
  finite-window nonlinear heat-flux gradients.
- **Scale:** matrix-free eigenmodes, reusable nonlinear execution, and parallel
  scans/ensembles. Distributed nonlinear execution remains restricted.
- **Research outputs:** resolved inputs, restartable NetCDF, heat-flux histories,
  spectra, and geometry plots.

## Run your first case

```bash
pip install gkx
gkx
```

The no-argument command runs a short linear Cyclone demo and writes its input,
time trace, mode structure, and plot. The first call includes JAX compilation.
This demonstrates the workflow, not a converged eigenmode benchmark.

To use the repository examples:

```bash
git clone https://github.com/uwplasma/GKX
cd GKX
pip install -e .
gkx examples/linear/axisymmetric/cyclone.toml
```

The last command prints the converged eigenvalue from the certified Krylov
path, `gamma ≈ 0.093` and `omega ≈ 0.282` at `ky = 0.3`; the tracked reference
for the full scan is in the parity table below.

| What you can do | One command |
| --- | --- |
| Linear growth rate and frequency | `gkx examples/linear/axisymmetric/cyclone.toml` |
| `ky` scan in parallel | `gkx scan examples/linear/axisymmetric/cyclone.toml` |
| Nonlinear turbulence with restart | `gkx run-runtime-nonlinear --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml --out run.nc` |
| Stellarator from a VMEC/VMEX `wout` | `gkx wout_w7x.nc --estimate` then `gkx wout_w7x.nc` |
| Collision model comparison | `python examples/theory_and_demos/collision_operator_comparison.py` |
| Eigenvalue and heat-flux gradients | `python examples/optimization/QA_optimization.py` |
| Replot any saved run | `gkx plot run.nc` |

### Nonlinear turbulence

```bash
gkx run-runtime-nonlinear \
  --config examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml \
  --steps 200 --out cyclone.out.nc
gkx --plot cyclone.out.nc
```

This is a short workflow demonstration, **not a saturated transport result**.
For a physics calculation, edit the deck's resolution, profiles, and duration;
use the [saturation and restart controls](docs/quickstart.rst).

### Your equilibrium

```bash
gkx wout_your_equilibrium.nc
# Copy/edit the resolved gkx.toml, then:
gkx my_input.toml wout_your_equilibrium.nc
```

The WOUT shortcut starts a nonlinear calculation and can be expensive.
Choose an explicit deck for controlled research comparisons. Vacuum equilibrium
does not mean zero gyrokinetic drive: use finite density and temperature gradients.

| Input/output | Meaning |
| --- | --- |
| `tprim`, `fprim` | `a/L_T = -a d(ln T)/dr`, `a/L_n = -a d(ln n)/dr`; not `R/L` |
| adiabatic electrons | prescribed Boltzmann response, not an evolved electron distribution |
| kinetic electrons | evolved electron distribution; usually more costly |
| `gamma`, `omega` | fitted mode growth rate and oscillation frequency; not transport convergence tests |
| `Wphi`, `Wg` | field-amplitude and distribution-energy diagnostics; not heat flux |
| `Q` | turbulent heat flux; its stationary, correlation-corrected mean is the transport observable |

Full definitions and units: [normalization](docs/normalization.rst),
[inputs](docs/inputs.rst), [outputs](docs/outputs.rst).

## Use Python and derivatives

Load a case through the public API:

```python
import gkx

case = gkx.load("examples/linear/axisymmetric/cyclone.toml")
simulation = gkx.prepare(case, Nl=4, Nm=8)
simulation.print_summary()
result = simulation.solve()
```

Preparing a case fixes its topology; it is not a general differentiable
parameter-update API. Specify preparation options explicitly: this route does
not yet preserve all CLI `[run]` settings. The small moment count above is a
workflow demonstration. Repeated nonlinear calls can accept an in-memory
initial state. See [solver contracts](docs/solvers.rst).

### Linear stability and quasilinear screening

Matrix-free eigenmodes avoid storing a dense operator.
The tracked large case has 494,592 unknowns; a dense complex128 matrix
would require 3.6 TiB. Eigenvalue derivatives use

$$
\frac{d\lambda}{dp}
=\frac{w^\dagger (\partial A/\partial p)v}{w^\dagger v},
$$

with residual and mode-selection checks. Quasilinear objectives additionally
need eigenfunction sensitivities.
[Equations and examples](docs/differentiable_eigensolver.rst).

Quasilinear output is **not a runtime/TOML absolute-flux predictor**.
The declared Solovev and shaped-pressure stress outliers remain failures;
use it for screening only where held-out nonlinear results support the ranking.
[Calibration and limits](docs/quasilinear.rst).

### Nonlinear heat flux

After a spin-up, differentiate a fixed post-saturation window from its turbulent state.
The following is the objective pattern; the linked example supplies the
geometry, grid, parameters, and state:

```python
import jax

def loss(shape):
    return gkx.nonlinear_heat_flux_window(
        saturated, grid, geometry(shape), params, dt, steps, terms=terms
    )

heat_flux, gradient = jax.value_and_grad(loss)(shape0)
```

The initial state is detached. This is the exact derivative of the specified
discrete window, **not** an exact long-time statistical transport derivative.
Use a measured gradient horizon and validate proposed designs independently.

![Checkpointed nonlinear derivatives](docs/_static/nonlinear_autodiff_validation.png)

In the tracked 1,024-step, 16³ Cyclone example, checkpointing reduced measured
temporary memory from 7.82 GB to 187 MB on CPU and 7.80 GB to 148 MB on an
RTX A4000, at 1.92× and 1.77× runtime. These are workload-specific historical
measurements, not current-head speed guarantees.
[Method, finite differences, and reproduction](docs/nonlinear_autodiff.rst).

### QA optimization with VMEX

[QA_optimization.py](examples/optimization/QA_optimization.py) adds physical
GKX heat flux to VMEX's quasisymmetry, aspect-ratio, and iota objective tuples:

```bash
pip install vmex
python examples/optimization/QA_optimization.py
```

This is an advanced, expensive research example. Warm spin-up is wired but
disabled by default; its re-equilibration bias still needs validation.
[Workflow and controls](examples/optimization/README.md).

![Initial and candidate QA boundaries and Boozer contours](docs/_static/qa_transport_equilibria.png)
![Matched QA heat-flux histories and refinements](docs/_static/qa_transport_reduction.svg)

The candidate shows a preliminary 12.26% reduction across nominal pairs, but it is
**not statistically resolved**: 4 of 48 nominal traces fail the final-drift
test. Promotion requires stationary individual traces, autocorrelation-aware
uncertainty, replicated independent runs, and resolution convergence.
[Initial/final conditions, scripts, and all results](docs/stellarator_optimization.rst).

## Validation and present limits

Linear agreement with the tracked scans, measured as
`100 * max|GKX - ref| / max|ref|`:

| Case | `gamma` | `omega` |
| --- | ---: | ---: |
| KAW | 0.0004% | 0.051% |
| ETG | 0.040% | 0.074% |
| W7-X | 0.265% | 0.296% |
| HSX | 0.577% | 0.273% |
| Cyclone Miller | 5.51% | 1.25% |
| Cyclone ITG | 6.83% | 1.59% |
| **KBM** | **20.0%** | **11.1%** |

These are reproducible artifact comparisons, not proof of matched physics
or resolution convergence. KBM remains discrepant; HSX reference provenance
needs completion. [Benchmark sources](tools/benchmark_atlas_manifest.toml).

Advanced collision operators have different validity ranges: drift-kinetic
low-moment checks do not validate finite-wavelength Coulomb tables.
The latter remain under coefficient/derivative audit, with restricted
like-species bases—not a general multispecies Landau implementation.
[Models](docs/operators.rst) · [Shipping criteria](docs/research_grade_program.rst).

W7-X zonal long-window recurrence/damping and W7-X TEM / kinetic-electron
extensions are deferred. General island-containing coil-field turbulence,
broad electromagnetic transport validation, and general equilibrium-flow
physics are not established.
[Current scope](docs/research_grade_plan.rst).

## Performance and parallel runs

![Runtime and memory measurements](docs/_static/runtime_memory_benchmark.png)

Compare **time to a trustworthy observable**, including compilation, spin-up,
sampling error, and derivative cost—not just milliseconds per step.
[Measured workloads](docs/performance.rst).

Independent scans and ensembles are the supported parallel path.
Species–Hermite decomposition exists, but its collision, identity, and
end-to-end speed gates are incomplete.
Sensitivity sweeps can use the same deterministic independent-work
reconstruction, but they need a dedicated accuracy/timing comparison.
[CPU/GPU examples and limits](docs/parallelization.rst).

## Documentation and development

Start with [quickstart](docs/quickstart.rst); then
[physics](docs/theory.rst), [numerics](docs/numerics.rst),
[geometry](docs/geometry.rst), and [testing](docs/testing.rst).
Figure recipes live with the corresponding result pages and
[manuscript figures](docs/manuscript_figures.rst).

```bash
pip install -e '.[dev]'
JAX_ENABLE_X64=true pytest
python tools/release/run_test_gates.py fast
python -m sphinx -W -b html docs docs/_build/html
```

Coverage is necessary, not physics validation. The [plan](plan.md) names the
remaining evidence gates; no new release is scheduled yet.

## Cite

Cite the software by version until the methods paper is out; GitHub renders
[CITATION.cff](CITATION.cff) as "Cite this repository". A Zenodo DOI is minted
at the next release.

```bibtex
@software{gkx,
  author  = {Jorge, Rogerio},
  title   = {GKX: gyrokinetic turbulence and sensitivities in JAX},
  version = {2.0.0},
  year    = {2026},
  url     = {https://github.com/uwplasma/GKX}
}
```

## License

[MIT](LICENSE); see [provenance](PROVENANCE.md).
