# SPECTRAX-GK

SPECTRAX-GK is a JAX-native gyrokinetic solver built for differentiability,
accelerator execution, and reproducible benchmark workflows. The code uses
Hermite-Laguerre velocity space, Fourier perpendicular coordinates, and
field-aligned flux-tube geometry for linear and nonlinear electrostatic and
electromagnetic runs.

![SPECTRAX-GK benchmark and convergence atlas](docs/_static/benchmark_readme_panel.png)

The figure above is the publication-facing benchmark summary used by the
README. It combines the representative convergence gate, the linear benchmark
master panel, and the nonlinear benchmark master panel. The underlying figure
builders are reproducible and live in `tools/`.

What is covered in that atlas:

- Linear growth-rate and real-frequency overlays for Cyclone ITG, ETG, KBM,
  W7-X, HSX, Cyclone Miller, KAW, and the KBM Miller late-growth window
- Extended linear stress tiles for Cyclone kinetic electrons and TEM
- Nonlinear heat-flux, free-energy, electrostatic-energy, and magnetic-energy
  traces for Cyclone ITG, KBM, W7-X, HSX, Cyclone Miller, and reduced cETG
- A compact convergence gate showing Cyclone resolution and `rho_star`
  sensitivity

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

Tracked benchmark coverage:

- Representative convergence: Cyclone resolution and `rho_star` sensitivity
- Linear overlays: Cyclone ITG, ETG, KBM, W7-X, HSX, Cyclone Miller, KAW, and
  the KBM Miller late-growth window
- Extended linear stress lanes: Cyclone kinetic electrons and TEM
- Nonlinear overlays: Cyclone ITG, KBM, W7-X, HSX, Cyclone Miller, and the
  reduced cETG comparison

Strict exact-window parity gates are kept separate from the broader scan
coverage. The current exact-window closures are the KAW diagnostic window and
the KBM Miller late-growth replay. The broader linear and nonlinear benchmark
panels remain benchmark overlays rather than universal `rtol <= 3e-2` parity
claims for every tile.

Regenerate the benchmark figures with:

```bash
python tools/make_benchmark_atlas.py
```

The atlas inputs are declared in `tools/benchmark_atlas_manifest.toml`, and
each regeneration writes a summary to
`tools_out/benchmark_atlas_summary.json`.

## Runtime and Memory

![Runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

The runtime panel reports measured wall time on a log scale and peak RSS on a
linear scale for the shipped benchmark cases on the tracked office benchmark
host. It compares SPECTRAX-GK CPU, SPECTRAX-GK GPU, and GX where the matching
GX benchmark is stable on the measured build. The reduced-model cETG runtime
row uses a matched short GX input with the same `dt` and total integration
window as the SPECTRAX-GK runtime case.

Current runtime/memory case set:

- Cyclone ITG linear and nonlinear
- ETG linear
- ETG nonlinear reduced model (cETG)
- KBM linear and nonlinear
- KAW linear
- W7-X linear and nonlinear
- HSX linear and nonlinear
- Cyclone Miller nonlinear

The runtime panel is intentionally separate from the benchmark atlas: the
atlas carries growth/frequency and transport/energy parity figures, while the
runtime panel carries wall-time and peak-memory measurements for the same
shipped case families where backend measurements are available.

Regenerate the runtime figure from collected per-case summaries with:

```bash
python tools/benchmark_runtime_memory.py \
  --summary-glob tools_out/runtime_memory_*linear.json \
  --summary-glob tools_out/runtime_memory_*nonlinear.json
```

## Examples

Config-backed case drivers:

```bash
python examples/cyclone_runtime_linear.py
python examples/cyclone_runtime_nonlinear.py --steps 200
python examples/cetg_runtime_nonlinear.py --steps 1000
python examples/etg_runtime_linear.py
python examples/kaw_runtime_linear.py
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
python examples/hsx_linear_imported_geometry.py --geometry-file /path/to/hsx_linear.eik.nc
python -m spectraxgk.cli run-runtime-linear --config examples/configs/runtime_w7x_linear_imported_geometry.toml
python examples/two_stream_hermite_1d.py
```

The `examples/configs` directory contains the runtime TOMLs used by the
config-backed examples, including Cyclone, cETG, ETG, KAW, KBM, Miller, W7-X,
HSX, and secondary-slab workflows.

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
