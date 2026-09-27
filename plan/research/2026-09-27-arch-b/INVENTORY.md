# ARCH-B inventory and second contraction (2026-09-27)

Lane: ARCH-B (plan.md G.6 P4; archived plan §8, §8.1, §17.3). Base: `origin/main`
at `cf1d40828` (2.4.0). Branch: `arch/contract-2`. Refreshes the ARCH-A
inventory in `plan/research/2026-09-23-arch-a/INVENTORY.md`.

Reproduce: `python plan/research/2026-09-27-arch-b/inventory.py .` (standard
library; import graph with function-local imports, strongly connected
components, single-consumer modules) and
`python plan/research/2026-09-27-arch-b/fingerprint.py .` (JAX, float64).

## 1. Before and after

| Measure | Base `cf1d40828` | This branch |
| --- | ---: | ---: |
| `src/gkx` Python files / lines | 160 / 79,446 | 136 / 74,748 |
| `tests` Python files / lines | 74 / 84,139 | 29 / 81,598 |
| Import cycles (SCCs, function-local imports counted) | 7 | 1 |
| Single-consumer split modules (cohesion gate, >= 6 names) | 2 | 1 |
| Low-cohesion modules (cohesion gate) | 8 (baseline) | 5 |
| Upward layer imports | 5 (baseline) | 4 |
| Lazy registry rows (`gkx.<name>`) | 260 | 217 |
| Tests collected (`-o addopts=`) | 3,028 | 2,953 |

The remaining cycle is `runtime` <-> `workflows.{nonlinear,runtime.*}` <->
`artifacts.nonlinear_netcdf`: the orchestration layers call back into the
facade. It dissolves with rank 1 below, not with a local edit.

## 2. What was done

- **Deleted** the deprecated reduced stellarator ITG model
  (`objectives/stellarator.py`, `sampling.py`, `vmec_boozer.py`, 2,094 lines;
  its gallery example was dropped in #281); the portfolio sensitivity report and its ten private gate helpers;
  the decomposition contracts (`parallel/decomposition.py`); and 48 test-only
  definitions no run, command, script, example or doc reached (figure builders,
  benchmark scan-policy helpers, W7-X trace loaders, branch/late-time metrics,
  eigenfunction bundles, `run_scan_and_mode`, the shape-aware power-law
  objective, CLI pass-through wrappers), with their tests.
- **Merged** 18 single-consumer helpers into their consumer (geometry Boozer
  constants/drifts/sensitivity/bridge, Krylov propagator, integrator and
  explicit diagnostics, collision factory, velocity streaming, parallel
  identity/batch, demo, solver status, initial phi, GX output, snapshots,
  runtime policies, objective portfolio).
- **Removed pass-throughs**: the `geometry.differentiable` facade hooks
  (module-attribute patching and wrappers; the registry now points at the
  owners), the velocity streaming sync hooks, the duplicate explicit linear
  step, the CLI TOML wrappers.
- **Broke six cycles** by moving one definition each to the lower module
  (Boozer field-line dataclasses, the `pr3` operator apply, the collision
  refusal helper, CLI shorthand routing, shared figure helpers, and the plot
  dispatcher onto the result types).
- **Tests** merged by domain into 25 test files plus four support files; no
  test body changed. Every deleted test exercised a deleted definition; no
  E1-E3 physics or mathematics test was deleted (the eigenvalue-derivative,
  shift-invert fallback and CFL-term tests were kept by keeping their
  functions).

## 3. Fingerprints (office host, XLA:CPU, JAX 0.10.2, float64)

`fingerprint.py` on base and on the branch; compared bitwise (`float.hex`,
SHA-256 prefixes). Linear Cyclone eigenpair, 100-step nonlinear heat-flux
trace, window gradient d<Q>/d(tprim), quasilinear flux at two k_y: identical.
Values are in the PR body.

## 4. Ranked next contractions

| Rank | Item | Files | Lines | Precondition |
| ---: | --- | ---: | ---: | --- |
| 1 | `workflows` + `runtime.py` + dependency-injection records (`*Deps`) -> `solve/`: callers pass callables, tests monkeypatch owners; dissolves the last cycle | -10 | ~3,000 | test_cli/test_runtime_runner monkeypatch audit (~270 patches) |
| 2 | Diagnostics timesteppers (`solvers_nonlinear_diagnostics`, `_diagnostic_integration`, `_imex_diagnostics`) folded into their integrators as sampled scans | -3 | ~1,500 | bitwise trajectory fingerprints (this harness) |
| 3 | Report modules (`diagnostics.validation_gates`, `zonal_validation`, `transport_windows`, `quasilinear_calibration`) -> `scripts/checks` or deleted with their gates | -4 | ~3,200 src | SLIM-SCRIPTS and DOCS lanes agree where the gates live |
| 4 | `artifacts` -> `io/` with one `_artifact_base` owner (two differ today) | -5 | ~1,000 | rank 1 |
| 5 | `terms` + `operators` -> `physics/` + `numerics/operators.py` | -6 | ~800 | rank 2 |
| 6 | Test line reduction: parametrize the per-helper coverage tests in `test_cli.py`, `test_runtime_runner.py`, `test_runtime_artifacts.py` (16k lines, most asserting forwarding) | 0 | ~8,000 tests | rank 1 |
