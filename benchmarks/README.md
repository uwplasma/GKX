# Benchmarks

Benchmark input decks live in `cases/`; `capability_matrix.toml` describes the
feature and comparison contract. Run a deck from the repository root:

```bash
gkx benchmarks/cases/secondary_slab.toml
```

Keep generated simulation data, restart files, plots, profiling traces and
benchmark summaries outside version control. Retain the input deck, command,
software versions and analysis settings needed to reproduce a result.

A comparison requires matched geometry, normalization, grid, initialization,
precision, timestepping, collisions, diagnostics and analysis windows.
