# Benchmarks

This root-level directory contains the lightweight benchmark drivers, runtime
TOML files, and result-index pointers used to reproduce the tracked validation
panels in the documentation.

The drivers intentionally keep generated outputs out of git:

- small scripts, TOML inputs, and result pointers live here,
- generated plots, NetCDF files, restart files, and logs should be written to
  `tools_out/` or an explicit scratch directory,
- publication-facing figures promoted to the docs are curated under
  `docs/_static/` after review and compression.

The current promoted benchmark outputs are indexed in
`benchmarks/results/manifest.toml`. That manifest is deliberately small: it
points to the compressed docs figures and machine-readable CSV/JSON summaries
without copying large artifacts into this directory.

Run from the repository root, for example:

```bash
python benchmarks/cyclone_linear_benchmark.py --outdir tools_out/cyclone_benchmark
python benchmarks/kbm_linear_comparison.py \
  --output tools_out/kbm_linear_comparison.png
python -m gkx.cli run-runtime-linear --config benchmarks/runtime_secondary_slab.toml
python benchmarks/secondary_slab_workflow.py
```

The machine-readable feature and comparison contract is
`benchmarks/capability_matrix.toml`. It separates required-core capabilities,
GKX differentiable extensions, optional research extensions, and
explicitly unsupported features. A comparison result is not considered matched
unless geometry, normalization, grid, initialization, precision, timestepping,
collisions, diagnostics, and analysis windows are all recorded.

The full atlas is built from tracked CSV/JSON assets rather than large transient
simulation directories:

```bash
python tools/artifacts/make_benchmark_atlas.py
```
