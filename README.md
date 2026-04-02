# SPECTRAX-GK

SPECTRAX-GK is a JAX-native gyrokinetic solver designed for differentiability, 
high-performance accelerator execution, and advanced stellarator optimization. 
The code employs a Hermite-Laguerre velocity space, Fourier perpendicular 
coordinates, and field-aligned flux-tube geometry to simulate linear and 
nonlinear electrostatic and electromagnetic turbulence in magnetized plasmas.

![SPECTRAX-GK convergence panel](docs/_static/benchmark_convergence_panel.png)

![SPECTRAX-GK linear benchmark panel](docs/_static/benchmark_core_linear_atlas.png)

![SPECTRAX-GK nonlinear benchmark panel](docs/_static/benchmark_core_nonlinear_atlas.png)

The figures above represent the validated benchmark suite, covering convergence, 
linear microinstabilities, and nonlinear transport across diverse magnetic 
configurations.

## Highlights

- **Differentiable JAX-native kernels** for gradient-based optimization and sensitivity analysis.
- **Hermite-Laguerre spectral velocity basis** providing efficient kinetic closures and multi-fidelity modeling.
- **Accelerator-ready execution** on CPUs and GPUs with JIT compilation.
- **Flexible geometry interface** supporting analytic s-alpha, Miller, and direct VMEC equilibrium imports.
- **Electromagnetic turbulence** including $(\phi, A_\parallel, B_\parallel)$ fluctuations.
- **Multi-species support** with kinetic electrons and advanced collision operators.
- **Automated benchmark workflows** for reproducible validation and regression tracking.

## Installation

```bash
pip install -e .
```

## Quickstart (CLI)

```bash
# Get information about the Cyclone base case
spectrax-gk cyclone-info

# Run a linear simulation with a specific k_y
spectrax-gk run-linear --config examples/linear/axisymmetric/runtime_cyclone.toml --plot

# Run directly from a runtime TOML
spectrax-gk examples/nonlinear/axisymmetric/runtime_cetg_reference.toml --steps 200

# Run a linear scan across k_y
spectrax-gk scan-linear --config examples/linear/axisymmetric/cyclone.toml --plot
```

Single-point runtime TOMLs can also carry their own artifact prefix:

```toml
[output]
path = "tools_out/runtime_case"
```

CLI `--out` overrides the TOML value when both are present.

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

SPECTRAX-GK is rigorously validated against standard gyrokinetic benchmarks, including:
- **Linear growth rates and frequencies:** Cyclone ITG, ETG, KBM, W7-X, HSX, Miller, and KAW.
- **Nonlinear transport:** Heat flux and energy traces for ITG, KBM, and stellarator configurations.

The benchmark tooling in `tools/` ensures reproducibility and performance tracking.

## Runtime and Memory

![Runtime and memory comparison](docs/_static/runtime_memory_benchmark.png)

SPECTRAX-GK is optimized for performance across CPU and GPU backends. The runtime panel above compares wall-time and peak memory usage for core linear and nonlinear benchmarks. Performance tracking covers:

- **Cyclone ITG** (linear/nonlinear)
- **KBM** and **ETG** configurations
- **W7-X** and **HSX** stellarator geometries
- **Miller** geometry models

Regenerate the runtime figure from collected per-case summaries with:

```bash
python tools/benchmark_runtime_memory.py \
  --summary-glob tools_out/runtime_memory_*linear.json \
  --summary-glob tools_out/runtime_memory_*nonlinear.json
```

## Examples

The `examples/` directory is organized by physics and configuration:

- **`linear/`**: Linear microinstability drivers for axisymmetric (Tokamak) and non-axisymmetric (Stellarator) geometries.
- **`nonlinear/`**: Nonlinear turbulence simulations and transport analysis.
- **`benchmarks/`**: Scripts for replicating published benchmark results and parameter scans.
- **`theory_and_demos/`**: Pedagogical examples and demonstrations of the underlying numerical methods.

## Documentation

Comprehensive documentation, including theory, algorithms, and API references, is available in `docs/`.

## Contributing

SPECTRAX-GK is an open-source project welcoming contributions. Whether it's improving runtimes, reducing memory usage, or expanding the physics models, your help is appreciated.

## License

BSD 3-Clause License.
