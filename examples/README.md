# Examples

These examples are for users learning how to run GKX and inspect its
outputs. They should stay runnable, documented, and distinct from benchmark or
release machinery.

## Recommended Path

1. `quickstart/` or the default executable demo: run `gkx` with no TOML
   to generate a small linear run and plot.
2. `linear/`: axisymmetric and non-axisymmetric linear ITG/ETG/KBM examples.
3. `nonlinear/`: bounded nonlinear transport examples. Long production windows
   should clearly state expected runtime and hardware.
4. `geometry/` or `vmec/`: VMEC/Miller/imported-geometry preparation examples.
5. `quasilinear/`: linear-run postprocessing and quasilinear diagnostic
   examples.
6. `optimization/`: VMEC-JAX/Boozer/GKX optimization workflows for
   growth-rate, quasilinear-flux, and nonlinear-window objectives.
7. `plotting/` or `utilities/`: plotting saved outputs, including
   `gkx --plot <output_file>`.
8. `parallelization/`: independent-work parallelization examples such as ky
   scans or UQ ensembles.

## Example Requirements

Each maintained example should document:

- what physics or workflow it demonstrates,
- expected laptop/runtime class,
- required optional dependencies,
- input files and how to generate them,
- output files and plots,
- whether it is pedagogical, benchmark-level, or long-run research.

Examples should not depend on private local paths or untracked artifacts. If an
example needs VMEC input, it should either include the small input file or point
to a reproducible generation step.

## What Belongs Elsewhere

- Benchmark comparisons and validation matrices belong in `benchmarks/` and
  `docs/`, not examples.
- Release/artifact/profiling scripts belong in `tools/`.
- Test-only workflows belong in `tests/`.
- Legacy or non-promoted physics paths should leave `main` or be clearly moved
  to a draft PR/experiment branch.
