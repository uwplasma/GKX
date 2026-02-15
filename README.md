# SPECTRAX-GK

SPECTRAX-GK is a clean-room rebuild of a gyrokinetic solver in JAX with a
Hermite-Laguerre velocity-space representation and Fourier real-space
representation. The initial target is the Cyclone base case in a simple analytic
s-alpha flux-tube geometry, followed by nonlinear turbulence, electromagnetic
extensions, and VMEC/DESC geometry.

## Status

Stage 0 is in progress. This repo currently provides:
- Hermite and Laguerre basis utilities
- Analytic s-alpha geometry helpers
- Flux-tube spectral grids
- Examples and tests to validate the basis and geometry

## Install

```bash
pip install -e .
```

## Quickstart

```bash
spectrax-gk cyclone-info
spectrax-gk cyclone-kperp --kx0 0.0 --ky 0.3
```

Examples:

```bash
python examples/basis_orthonormality.py
python examples/cyclone_geometry.py
python examples/linear_rhs_demo.py
python examples/cyclone_linear_benchmark.py
```

Example input:

```bash
cat examples/cyclone_base_case.toml
```

## Roadmap (high level)

1. Linear electrostatic GK operator in Hermite-Laguerre basis
2. Cyclone base case linear benchmarks
3. Nonlinear E x B term and turbulence tests
4. Electromagnetic extensions, multispecies, advanced collisions
5. VMEC/DESC geometry adapters and stellarator benchmarks
6. Performance tuning and profiling

## Development notes

- The previous codebase is preserved on the `legacy` branch.
- This branch is a clean rebuild with a `src/` layout.

## License

See `LICENSE`.
