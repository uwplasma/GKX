# SPECTRAX-GK

SPECTRAX-GK is a JAX-native gyrokinetic solver built for differentiability,
accelerator execution, and reproducible benchmark workflows. The code uses
Hermite-Laguerre velocity space, Fourier perpendicular coordinates, and
field-aligned flux-tube geometry for linear and nonlinear electrostatic and
electromagnetic runs.

![SPECTRAX-GK benchmark and convergence atlas](docs/_static/benchmark_readme_panel.png)

The figure above is the publication-facing benchmark summary used by the README.
It combines representative convergence checks, core linear and nonlinear
parity panels, imported-geometry and exact-diagnostic audits, and the extended
linear stress matrix. The underlying figure builders are reproducible and live
in `tools/`.

## Highlights

- **JAX-first solver kernels** with differentiable primitives and JIT execution.
- **Hermite-Laguerre velocity representation** for compact kinetic closures.
- **Field-aligned flux-tube geometry** with analytic and imported VMEC workflows.
- **Electromagnetic field solve** for `(\phi, A_\parallel, B_\parallel)`.
- **Explicit, IMEX, and implicit time integrators** with CFL-controlled
  stepping and streaming diagnostics.
- **Operator caching and batched scans** for repeated linear and nonlinear runs.
- **Benchmark tooling** for growth-rate/frequency scans, nonlinear transport
  traces, restart checks, exact-state audits, atlas generation, and runtime
  comparison plots.

## Installation

```bash
pip install -e .
```

## Quickstart (CLI)

```bash
spectrax-gk cyclone-info
spectrax-gk cyclone-kperp --kx0 0.0 --ky 0.3
spectrax-gk run-linear --config examples/configs/cyclone.toml --plot --outdir docs/_static
spectrax-gk scan-linear --config examples/configs/etg.toml --plot --outdir docs/_static
```

## Quickstart (Python)

```python
from spectraxgk import CycloneBaseCase, LinearParams, integrate_linear_from_config
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
import jax.numpy as jnp

cfg = CycloneBaseCase()
grid = build_spectral_grid(cfg.grid)
geom = SAlphaGeometry.from_config(cfg.geometry)
params = LinearParams()

G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

G_t, phi_t = integrate_linear_from_config(G0, grid, geom, params, cfg.time)
```

## Benchmarks

The benchmark layout follows the usual gyrokinetic-paper pattern: linear
overlays of growth rate and real frequency versus `k_y`, and nonlinear time
traces of heat flux, free energy, electrostatic field energy, and magnetic
field energy where relevant.

Representative convergence gate:

- Cyclone resolution and `rho_star` sensitivity are shown explicitly in the
  README panel and used as the tracked grid-convergence check for the benchmark
  suite.

Primary full-GK publication set:

- Cyclone ITG linear and nonlinear
- KBM linear and nonlinear
- W7-X linear and nonlinear
- HSX linear and nonlinear
- Cyclone Miller geometry linear and nonlinear

Core linear benchmark set:

- Cyclone ITG against the tracked benchmark reference
- ETG against the tracked internal benchmark reference
- KBM against the tracked internal benchmark reference and exact-diagnostic audits
- Imported-geometry and exact-diagnostic linear checks for W7-X, HSX, Miller,
  and KAW

Extended linear stress matrix:

- Cyclone kinetic electrons
- TEM
- KBM Miller exact late growth window

Core linear atlas:

![Core linear benchmark atlas](docs/_static/benchmark_core_linear_atlas.png)

Core nonlinear atlas:

![Core nonlinear benchmark atlas](docs/_static/benchmark_core_nonlinear_atlas.png)

Representative convergence panel:

![Representative convergence panel](docs/_static/benchmark_convergence_panel.png)

Extended linear stress matrix:

![Extended linear stress matrix](docs/_static/benchmark_extended_linear_panel.png)

Regenerate the benchmark figures with:

```bash
python tools/make_benchmark_atlas.py
```

The atlas inputs are declared in `tools/benchmark_atlas_manifest.toml`, and
each regeneration writes a summary to
`tools_out/benchmark_atlas_summary.json`.

The full refresh matrix lives in `tools/benchmark_refresh_manifest.toml` and
can be inspected with:

```bash
python tools/run_benchmark_refresh.py --list
```

## Examples

Config-backed case drivers:

```bash
python examples/cyclone_runtime_linear.py
python examples/cyclone_runtime_nonlinear.py --steps 200
python examples/etg_runtime_linear.py
python examples/kbm_runtime_linear.py
python examples/kbm_runtime_nonlinear.py --steps 200
python examples/miller_nonlinear_runtime.py --steps 200
python examples/w7x_nonlinear_vmec_geometry.py --steps 200
python examples/hsx_nonlinear_vmec_geometry.py --steps 200
```

Benchmark and theory demos:

```bash
python examples/basis_orthonormality.py
python examples/cyclone_geometry.py
python examples/diffrax_linear_demo.py
python examples/linear_rhs_demo.py
python examples/example.py
python examples/cyclone_linear_benchmark.py
python examples/etg_linear_benchmark.py
python examples/gradB_coupling_hl_1d.py
python examples/kbm_beta_scan.py
python examples/w7x_linear_imported_geometry.py --geometry-file /path/to/itg_w7x_adiabatic_electrons.eik.nc
python -m spectraxgk.cli run-runtime-linear --config examples/configs/runtime_w7x_linear_imported_geometry.toml
python examples/two_stream_hermite_1d.py
```

The `examples/configs` directory contains the runtime TOMLs used by the
config-backed examples, including Cyclone, ETG, KBM, Miller, W7-X, HSX, and
secondary-slab workflows.

## Documentation

Build the docs locally with:

```bash
cd docs
pip install -r requirements.txt
make html
```

The benchmark discussion, algorithms, inputs, and reference list live in
`docs/`.

## Development

- Run tests with `pytest`.
- Rebuild the benchmark figures with `python tools/make_benchmark_atlas.py`.
- Keep the atlas inputs explicit in `tools/benchmark_atlas_manifest.toml`.
- Keep large benchmark reruns and office-specific audits documented in
  `plan.md`.

## License

BSD 3-Clause License.
