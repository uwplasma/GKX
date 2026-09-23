# Benchmarks

This directory holds benchmark data: the validation input decks in `cases/`,
the release reference contracts in `references/`, the result index in
`results/`, and the feature contract `capability_matrix.toml`. The drivers that
run the benchmarks are the modules of `python scripts/benchmark.py` (source in
`scripts/benchmarks/`); `python scripts/benchmark.py --list` prints them.

The drivers intentionally keep generated outputs out of git:

- TOML inputs and result pointers live here, and the drivers in `scripts/`,
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
python scripts/benchmark.py linear_benchmark cyclone --outdir tools_out/cyclone_benchmark
python scripts/benchmark.py linear_benchmark kbm --outdir tools_out
python -m gkx.cli run-runtime-linear --config benchmarks/cases/secondary_slab.toml
python scripts/benchmark.py secondary_slab_workflow
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
python scripts/validate.py make_benchmark_atlas
```
