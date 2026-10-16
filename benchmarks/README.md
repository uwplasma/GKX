# Benchmarks

This directory holds benchmark data: the validation input decks in `cases/`,
the release reference contracts in `references/`, the result index in
`results/`, and the feature contract `capability_matrix.toml`. The decks run with
the `gkx` CLI. The former Python drivers were removed from `scripts/` on
2026-10-15; recover one with `git show df2953026:<path>` (list in `plan/log.md`).

The drivers intentionally keep generated outputs out of git:

- TOML inputs and result pointers live here,
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
python -m gkx.cli run-runtime-linear --config benchmarks/cases/secondary_slab.toml
```

The machine-readable feature and comparison contract is
`benchmarks/capability_matrix.toml`. It separates required-core capabilities,
GKX differentiable extensions, optional research extensions, and
explicitly unsupported features. A comparison result is not considered matched
unless geometry, normalization, grid, initialization, precision, timestepping,
collisions, diagnostics, and analysis windows are all recorded.

The full atlas is built from tracked CSV/JSON assets rather than large transient
simulation directories, by `scripts/artifacts/make_benchmark_atlas.py` (removed; `git show df2953026:<path>`).
