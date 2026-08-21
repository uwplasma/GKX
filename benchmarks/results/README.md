# Benchmark Results Index

This directory is the root-level index for benchmark outputs promoted into the
GKX documentation. It intentionally contains only small pointer files.

Tracked benchmark result artifacts remain under `docs/_static/` because Sphinx
serves them directly. Raw simulation directories, NetCDF outputs, restart files,
profiling traces, and logs must stay out of git and should be written to
`tools_out/` or another scratch directory.

Use the manifest here to find the current publication-facing benchmark outputs:

```bash
cat benchmarks/results/manifest.toml
```

Regenerate the compact atlas and runtime-memory panel from the repository root:

```bash
python tools/artifacts/make_benchmark_atlas.py
python benchmarks/performance/benchmark_runtime_memory.py \
  --summary-glob docs/_static/runtime_memory_summary_ship_refresh.json \
  --csv-out docs/_static/runtime_memory_results_ship_refresh.csv \
  --summary-out docs/_static/runtime_memory_summary_ship_refresh.json \
  --plot-out docs/_static/runtime_memory_benchmark.png
```
