# SPECTRAX-GK

SPECTRAX-GK is a clean-room, JAX-native gyrokinetic solver designed for
performance, differentiability, and rapid experimentation. The code uses a
Hermite-Laguerre velocity-space representation with Fourier perpendicular
coordinates in a field-aligned flux-tube geometry. The initial validation target
is the **Cyclone base case** with adiabatic electrons, alongside kinetic-electron
ITG/ETG, KBM beta scans, and TEM linear checks.

![Linear validation summary](docs/_static/linear_summary.png)

## Highlights

- **JAX-first design**: fully differentiable kernels and JIT compilation.
- **Hermite-Laguerre velocity space**: compact spectral representation.
- **Field-aligned flux-tube geometry**: s-alpha analytic model (VMEC/DESC next).
- **Full drift/mirror physics**: curvature/grad-B/mirror couplings + diamagnetic drive.
- **Electromagnetic fields**: coupled :math:`(\\phi, A_\\parallel, B_\\parallel)` solve.
- **Term toggles**: switch linear-operator components via ``LinearTerms``.
- **Field-aligned grid controls**: ``y0``, ``ntheta``, and ``nperiod`` inputs.
- **Stable integrators**: explicit, IMEX, and implicit time stepping options.
- **Cached operators**: precomputed geometry arrays for faster time stepping.
- **Benchmark harness**: reference data + growth-rate extraction tools + comparisons.
- **ETG trend checks**: reduced electron-scale grids for linear trend validation.
- **Auto window fitting**: robust growth-rate extraction from transient signals.
- **Publication-ready plots**: consistent styling and reusable plotting utilities.
- **100% test coverage**: unit, regression, and physics-based checks.

## Installation

```bash
pip install -e .
```

## Quickstart (CLI)

```bash
spectrax-gk cyclone-info
spectrax-gk cyclone-kperp --kx0 0.0 --ky 0.3
```

## Quickstart (Python)

```python
from spectraxgk import load_cyclone_reference, run_cyclone_linear

ref = load_cyclone_reference()
result = run_cyclone_linear(ky_target=0.3, method="rk4")

print(result.gamma, result.omega)
```

## Config-driven integration

```python
import jax.numpy as jnp
from spectraxgk import CycloneBaseCase, LinearParams, integrate_linear_from_config
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid

cfg = CycloneBaseCase()
grid = build_spectral_grid(cfg.grid)
geom = SAlphaGeometry.from_config(cfg.geometry)
params = LinearParams()

G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

G_t, phi_t = integrate_linear_from_config(G0, grid, geom, params, cfg.time)
```

## Examples

```bash
python examples/basis_orthonormality.py
python examples/cyclone_geometry.py
python examples/diffrax_linear_demo.py
python examples/linear_rhs_demo.py
python examples/example.py
python examples/cyclone_linear_benchmark.py
python examples/etg_linear_benchmark.py
python examples/kinetic_linear_benchmark.py
python examples/gradB_coupling_hl_1d.py
python examples/kbm_beta_scan.py
python examples/tem_linear_benchmark.py
python examples/two_stream_hermite_1d.py
```

Diffrax can be enabled for the ETG/TEM/KBM examples with:

```bash
python examples/etg_linear_benchmark.py --diffrax --solver Tsit5 --adaptive
python examples/tem_linear_benchmark.py --diffrax --solver Tsit5 --adaptive
python examples/kbm_beta_scan.py --diffrax --solver Tsit5 --adaptive
```

## Validation status

- **Cyclone base case (adiabatic electrons)**: the benchmark harness reproduces
  published growth rates and real frequencies across the reduced ky scan using
  the full drift/mirror operator.
- **ETG linear trend**: growth rates remain positive across reduced electron-scale
  gradients; real frequencies follow the electron diamagnetic direction.
- **KBM beta scan**: electromagnetic transition between ITG and KBM branches.
- **TEM benchmark**: low-:math:`k_y` trapped-electron branch on the published
  s-alpha parameter set.

## Figures

```bash
python tools/make_figures.py
```

## Integrator benchmark

```bash
python tools/benchmark_integrators.py
```

## Figure parameters

### Cyclone base case (adiabatic electrons, GX Fig. 1)

| Parameter | Value |
| --- | --- |
| Geometry | q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778 |
| Gradients | R/LTi=2.49, R/LTe=0.0, R/Ln=0.8 |
| Species | ions (Z=1, m=1), adiabatic electrons (tau_e=1) |
| Electromagnetic | beta=0, A_parallel=off, B_parallel=off |
| Collisions | nu_i=1.0e-2, hypercollisions=off |
| Operator toggles | streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off |
| Grid | Nx=1, Ny=24, Nz=96, y0=20, ntheta=32, nperiod=2 |
| Velocity resolution | Nl=6, Nm=16 |
| Reference | GX paper Fig. 1 |

### ETG (electron scale, GX Fig. 2b)

| Parameter | Value |
| --- | --- |
| Geometry | q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778 |
| Gradients | R/LTi=2.49, R/LTe=2.49, R/Ln=0.8 |
| Species | ions + electrons, Te/Ti=1, mi/me=3670 |
| Electromagnetic | beta=1.0e-5, A_parallel=on, B_parallel=on |
| Collisions | nu_i=1.0e-2, nu_e=1.65e-4, hypercollisions=off |
| Operator toggles | streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off |
| Grid | Nx=1, Ny=24, Nz=96, y0=0.2, ntheta=32, nperiod=2 |
| Velocity resolution | Nl=6, Nm=16 |
| Reference | GX paper Fig. 2b |

### KBM beta scan (GX Fig. 3)

| Parameter | Value |
| --- | --- |
| Geometry | q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778 |
| Gradients | R/LTi=2.49, R/LTe=2.49, R/Ln=0.8 |
| Species | ions + electrons, Te/Ti=1, mi/me=3670 |
| Electromagnetic | beta_ref scan (values from `kbm_reference.csv`), A_parallel=on, B_parallel=off |
| Collisions | nu_i=0, nu_e=0, hypercollisions=off |
| Operator toggles | streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off |
| Grid | Nx=1, Ny=12, Nz=96, y0=10, ntheta=32, nperiod=2 |
| Velocity resolution | Nl=6, Nm=16 |
| Reference | GX paper Fig. 3 |

### TEM (steep gradients, Frei et al. 2022 Fig. 4)

| Parameter | Value |
| --- | --- |
| Geometry | q=2.7, s_hat=0.5, epsilon=0.18, R0=2.77778, alpha=0 |
| Gradients | R/LTi=20, R/LTe=20, R/Ln=20 |
| Species | ions + electrons, Te/Ti=1, mi/me=370 |
| Electromagnetic | beta=1.0e-4, A_parallel=on, B_parallel=off |
| Collisions | nu_i=0, nu_e=0, hypercollisions=off |
| Operator toggles | streaming/mirror/curvature/grad-B/diamagnetic on; nonlinear off |
| Grid | Nx=1, Ny=24, Nz=160, y0=20, ntheta=32, nperiod=3 |
| Velocity resolution | Nl=6, Nm=16 |
| Reference | Frei et al. 2022 Fig. 4 |

## Documentation

The ReadTheDocs site provides:

- theory and equations
- numerical discretization and algorithms
- geometry and flux-tube model
- benchmark methodology and reference data
- API reference and examples

## Roadmap (high level)

1. Linear electrostatic GK operator (Hermite-Laguerre)
2. Cyclone base case linear benchmarks
3. Nonlinear E x B term and turbulence tests
4. Electromagnetic extensions, multispecies, advanced collisions
5. VMEC/DESC geometry adapters and stellarator benchmarks
6. Performance tuning and profiling

## References

- Laguerre-Hermite pseudo-spectral GK: [arXiv:1708.04029](https://arxiv.org/abs/1708.04029)
- Gyrokinetic equations (Frieman & Chen, 1982): [OSTI record](https://www.osti.gov/biblio/5235502)
- Low-frequency kinetic equations (Antonsen & Lane, 1980): [OSTI record](https://www.osti.gov/biblio/5115944)
- GENE code: [J. Comput. Phys. 230, 6979 (2011)](https://www.sciencedirect.com/science/article/pii/S0021999111002609)
- stella code: [arXiv:1806.02162](https://arxiv.org/abs/1806.02162)

## Development notes

- The previous codebase is preserved on the `legacy` branch and archived in
  `legacy_archive/`.
- This branch uses a `src/` layout and enforces 100% test coverage.

## License

See `LICENSE`.
