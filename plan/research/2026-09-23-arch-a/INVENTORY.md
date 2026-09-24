# ARCH-A inventory and first contraction (2026-09-23)

Lane: ARCH-A (plan.md G.3, G.6 P4; archived plan §2.4, §8, §8.1, §8.2).
Base: `origin/chain/p0` at f005418bf. Branch: `arch/contract-1`.

Targets (archived §2.4, §8): at most 45 installable files and 45,000 source
lines; one owner per equation and algorithm; no `runtime`, `workflows`,
`artifacts` or `terms` package in the final topology; no source module whose
main purpose is a manuscript, release or campaign report; no pass-through
wrappers; no duplicated linear/nonlinear field equations; no second
"diagnostics" timestepper.

## 1. Method

Everything below is reproducible from the tree with the standard library plus
the pinned venv (JAX 0.10.2):

- **Import graph.** Every `import`/`from ... import` in `src/gkx`, including
  function-local imports and `"gkx.x.y"` string constants, resolved to the
  owning module. Lazy package exports (`gkx`, `gkx.api`, `gkx.parallel`,
  `gkx.operators*`, `gkx.terms`, `gkx.diagnostics`, `gkx.geometry`, ...) were
  resolved by importing the package and reading each exported object's
  `__module__`, so `from gkx.parallel import independent_map` counts as an edge
  to `gkx.parallel.independent`.
- **Product reachability.** Roots: `gkx.cli`, `gkx.api.prepared`, the fifteen
  names in `gkx.api.__all__`, and every module an `examples/` script imports.
  The lazy compatibility registry (`gkx.api._EXPORT_TARGETS`, 346 rows) is
  *not* a root: a module reachable only through it is not reached by any run,
  CLI command or documented entry point.
- **Symbol usage.** For every top-level definition, identifier references in
  `src`, `tests`, `scripts`, `examples`, `docs`, `benchmarks`, `tools` and CI,
  excluding `__all__` lists and lazy-export tables. A definition with no
  reference outside its own `def` is dead. Name-based, so conservative: a
  shared name keeps a definition alive.
- **Duplicates.** Functions with at least 25 AST nodes, docstring stripped.
  *Exact*: identical `ast.dump` of signature and body. *Near*: identical after
  renaming every identifier and string to a placeholder.
- **Tests.** Each test function classified by the modules its body (and the
  module-level helpers it calls) references: DELETE when every `gkx`/`scripts`
  reference is to a deleted module, MIXED when some are, KEEP otherwise.

## 2. Inventory at the base

| Measure | Base (f005418bf) |
| --- | ---: |
| `src/gkx` files / lines | 187 / 93,953 |
| `tests` files / lines | 81 / 94,628 |
| `scripts` files / lines | 106 / 78,277 |
| Modules unreachable from product roots | 27 modules, 13,738 lines |
| Import cycles (strongly connected components) | 7 |
| `gkx.__all__` / lazy registry rows | 15 / 346 |
| Exact / near-duplicate function groups | 11 / 25 (351 near-duplicate lines) |
| Median module | 427 lines |
| Wheel / sdist | 868,489 / 769,614 bytes |

Import cycles (unchanged by this PR; each is a candidate for the merges in §5):

1. `geometry` ↔ `geometry.flux_tube_contract` ↔ `geometry.vmec_tensor_mapping`
2. `solvers_linear_krylov_algorithms` ↔ `solvers_linear_precond_pr3`
3. `geometry.vmec_boozer_core` ↔ `vmec_state_controls` ↔ `vmec_boozer_derivatives`
4. `artifacts.plotting` ↔ `zonal_plots` ↔ `transport_figures` ↔ `gx_output` ↔ `foreign_output` ↔ `workflows.runtime.results`
5. `workflows.runtime.toml` ↔ `wout` ↔ `resolution`
6. `runtime` ↔ `workflows.nonlinear` ↔ `workflows.runtime.{commands,artifacts,orchestration_scan}` ↔ `artifacts.nonlinear_netcdf`
7. `gkx` ↔ `gkx.api` ↔ `gkx.api.prepared` (lazy facade; benign)

### 2.1 Modules unreachable from any product root

All 27 were reachable only from their own tests, from campaign/report scripts,
or through the compatibility registry.

| Group | Modules | Lines | Purpose |
| --- | --- | ---: | --- |
| Nonlinear spectral-identity / device-z prototypes | `operators.nonlinear.{device_z, spectral_identity_integrator, spectral_identity_reports, spectral_identity_rhs, domain_decomposition, parallel_contracts_domain, parallel_contracts_strategy, parallel}` (+ `spectral_core`, `parallel_contracts_spectral`, reachable only for two tolerance helpers) | 4,683 | identity gates for a reduced bracket operator the production run never calls |
| Diagnostics reports and gates | `diagnostics.{nonlinear_gradient_statistics, nonlinear_transport_optimization, quasilinear_model_selection, transport}` | 3,029 | promotion/claim-boundary reports |
| Diagnostics calibration | `diagnostics.quasilinear_calibration` | 885 | calibration report schema (see §4: kept) |
| VMEC/Boozer objective gates | `objectives.{vmec_boozer_line_search, vmec_boozer_fd, vmec_boozer_gradients, vmec_boozer_context, solver_vmec, geometry, vmec_transport_optimization, vmec_transport_branch, zonal}`, `geometry.vmec_boundary_chain` | 5,873 | FD/line-search/holdout gate reports and default-injecting wrappers |
| Pass-through facades | `solvers_linear`, `solvers_nonlinear`, `solvers_time` | 171 | re-export only |
| Parallel batch map | `parallel.batch` | 159 | kept: CI's wheel smoke test asserts its import path |

### 2.2 Dead definitions after the deletions

37 definitions (755 lines) lost their last reference when the modules above
went, found by iterating the symbol scan to a fixed point: 29 in
`diagnostics/metadata.py` (the nonlinear-gradient evidence contracts), 4 in
`objectives/vmec_boozer.py` (the objective-table entry points `solver_vmec`
wrapped), and one each in `benchmarking_shared.py` and
`geometry/vmec_field_line_sampling.py`.

### 2.3 Exact duplicates at the base

| Group | Disposition here |
| --- | --- |
| `operators.linear.dissipation._species_vector` = `operators.nonlinear.collisions._species_vector` | merged; nonlinear imports the linear owner |
| `callbacks._format_duration` = `workflows.runtime.chunks.format_duration` | merged; `chunks` re-binds the `callbacks` owner |
| `workflows.linear._fit_signal_key` = `workflows.runtime.orchestration_scan._fit_signal_key` | merged into `workflows.runtime.diagnostics` |
| `solvers_linear_integrator_diagnostics._validate_linear_sampling` = `solvers_linear_implicit._validate_implicit_sample_policy` | merged into the implicit owner |
| `diagnostics.analysis.fit_growth_rate_auto_with_stats` ≈ `diagnostics.growth_rates.fit_growth_rate_auto_with_stats` (64/58 lines) | queued: the facade copy exists so tests can monkeypatch it; merge with those tests |
| `artifacts.run_figures._is_netcdf_source` = `artifacts.transport_figures._is_netcdf_bundle_path` | queued with the `artifacts` → `io` merge (lazy-import boundary) |
| two `operators.linear.collisions.*.tree_flatten`, two `operators.moments._get_m`, two `parallel_contracts_*.to_dict` | intra-module; the last pair was deleted |

Near duplicates that are one algorithm written twice (queued, §5): heat and
particle flux in `operators/fluxes.py` (`*_total`, `*_channel_species`) and in
`operators/moments.py` (`*_resolved_species`, `*_channel_resolved_species`) —
the same quadrature with a different velocity weight; `cache_arrays._shift_axis_for_cache`
and `streaming.shift_axis`.

### 2.4 Deprecated aliases

| Alias | Status |
| --- | --- |
| CLI `run-runtime-linear`, `scan-runtime-linear`, `run-runtime-nonlinear` | "removed in the next release" since 2.x; ~20 CLI tests still call them. Queued (rank 3). |
| `LinearParams.R_over_LTi/R_over_Ln/R_over_LTe` | public-API aliases without a removal date; left |
| `objectives/stellarator.py` (reduced analytic model) | module docstring: "slated for removal"; queued (rank 1) because `examples/theory_and_demos/reduced_stellarator_itg/` (EXAMPLES-GALLERY lane) depends on it |
| `KrylovConfig` method labels "batched/incremental/flexible" | compatibility labels validated as aliases; left |

## 3. Executed in this PR (contraction 1)

| Item | Files | Source lines | Risk | Tests affected |
| --- | ---: | ---: | --- | --- |
| Nonlinear spectral-identity / device-z stack (10 modules); `workflows.runtime.parallel_nonlinear` now owns its allclose check (net 0 lines there) | −10 | −4,683 | low: unreachable from runs; the routing identity check keeps its tolerance convention | `test_parallel_nonlinear.py` deleted (55 tests, all identity gates of the deleted stack); 6 device-z profiler contract tests |
| Diagnostics reports/gates (4 modules) | −4 | −3,029 | low: report builders only | `test_nonlinear_gradient_followup.py`, `test_nonlinear_transport_optimization.py`, `test_quasilinear_model_selection.py`, `test_nonlinear_gradient_evidence.py`, `test_check_overdetermined_nonlinear_gradient_campaign.py` deleted |
| VMEC/Boozer objective gates and wrappers (10 modules) | −10 | −5,873 | low: gates and default-injecting wrappers; `objectives.vmec_boozer` keeps the implementations | 72 tests in `test_autodiff_solver_objectives.py`, `test_vmec_transport_objectives.py`, `test_runners_and_orchestration.py`; the two-parameter analytic geometry fixture moved into the test file |
| Pass-through facades `solvers_linear`, `solvers_nonlinear`, `solvers_time` | −3 | −171 | none: imports repointed to owners | one facade-identity test |
| Dead definitions (fixed-point sweep) and the imports only they used | 0 | −864 | none: no reference anywhere | none |
| Exact duplicates merged (4 groups) | 0 | −25 | none: identical bodies | none |
| Lazy registry rows pointing at deleted modules | 0 | −86 | public-surface change: 86 compatibility names removed from `gkx.<name>` | none outside deleted tests |
| Campaign/report scripts that only drove deleted modules (10) and the `objective-gate` subcommand of `build_zonal_flow_artifacts.py` | scripts −10 | scripts −8,030 | low: none runs in CI; `check.py nonlinear-optimization` removed | as above |

No E1–E3 test was deleted. The deleted tests are report-schema, promotion-gate,
facade-identity and profiler-contract tests (E0) and identity gates of a reduced
operator that is not the production RHS.

## 4. Deliberately kept

- `diagnostics.quasilinear_calibration` (885 lines): `docs/quasilinear.rst`
  documents it as the user workflow for train/holdout calibration reports.
  Its removal needs a docs decision, not only an architecture one.
- `parallel.batch`: the CI wheel smoke test asserts
  `gkx.parallel.batch_map.__module__`.
- Frozen evidence JSON under `docs/_static/` for the deleted gates: the docs
  now say which builder was retired. Pruning those files is SLIM/DOCS work.

## 5. Ranked contraction queue

Ranked by lines saved per unit risk. "Tests" is the number of test files that
import the group today.

| Rank | Item | Files | Lines | Risk | Tests | Precondition |
| ---: | --- | ---: | ---: | --- | ---: | --- |
| 1 | Delete the deprecated reduced stellarator model (`objectives.stellarator`, `objectives.vmec_boozer`, `objectives.sampling`); move `smooth_positive` and the portfolio shape into `objectives.vmec_transport` | 3 | ~2,000 | low | 2 | EXAMPLES-GALLERY drops `examples/theory_and_demos/reduced_stellarator_itg/` |
| 2 | Retire the report diagnostics (`diagnostics.validation_gates`, `zonal_validation`, `transport_windows`), keeping the physics functions (Rosenbluth–Hinton residual, window statistics) in `solve/diagnostics` | 3 | ~2,000 | medium: E3 zonal-residual tests use `zonal_validation` | 5 | move the E3 functions first |
| 3 | Remove the three deprecated CLI commands and migrate `test_cli.py` to `gkx run/scan` | 0 | ~60 src, ~300 tests | low | 1 | none |
| 4 | Fold the diagnostics timesteppers (`solvers_nonlinear_diagnostics`, `_diagnostic_integration`, `_imex_diagnostics`, `solvers_time_explicit_diagnostics`, `solvers_linear_integrator_diagnostics`) into their integrators as sampled scans | −5 | ~1,500 | medium: bitwise trajectory fingerprints required | 8 | fingerprint harness (this PR) |
| 5 | Lazy registry → versioned migration table outside `__all__` (≤30 advertised names); delete unused compatibility rows | 0 | ~200 | public surface | 4 | release note |
| 6 | Parallel portfolio/identity (`parallel.decomposition`, `identity`, `independent`, `batch`) → one `numerics/parallel.py` | −3 | ~600 | low | 1 | CI smoke-test path update |
| 7 | One flux quadrature owner (`operators/fluxes.py` + `operators/moments.py` heat/particle pairs) | 0 | ~250 | low: E1 flux-moment tests | 4 | none |
| 8 | In-package VMEC/Boozer geometry (11 modules) per archived §9.2 | −8 | ~4,500 | high | 8 | §9.2 parity freeze on live-state/WOUT/EIK |
| 9 | `terms` → `physics/equations.py` + `physics/fields.py` (one field-equation owner) | −4 | ~800 | medium | 26 | rank 4 |
| 10 | `workflows` + `runtime.py` → `solve/`, `io/`, `cli` (breaks cycles 5, 6) | −12 | ~3,000 | medium | 25 | ranks 4, 9 |
| 11 | `artifacts` → `io/` (breaks cycle 4) | −7 | ~1,500 | low | 9 | rank 10 |
| 12 | Velocity-sharded linear routes (9 modules, 3,717 lines) → `numerics/parallel.py`; keep only measured routes | −6 | ~1,500 | medium | 6 | PERF lane measurement |

Ranks 1–7 are about 6,600 lines at low or medium risk; with ranks 8–12 the
tree reaches roughly 55,000 lines in about 90 files, and the §2.4 targets then
need the one-owner merges inside `solvers_*` and `operators/`.

## 6. Behaviour fingerprints (office host, JAX 0.10.2, float64)

`fingerprint.py` in this directory, run on the base (f005418bf) and the
contracted tree. The comparison is bitwise.

| Fingerprint | XLA:CPU base | XLA:CPU head | CUDA (A4000) base | CUDA head |
| --- | --- | --- | --- | --- |
| Cyclone linear gamma | `0x1.7ed2ffdd9f835p-4` | identical | `0x1.7ed2ffdd9f834p-4` | identical |
| Cyclone linear omega | `0x1.286b1286ae465p-2` | identical | `0x1.286b1286ae466p-2` | identical |
| eigenfunction SHA-256[:16] | `a92b98619ab58211` | identical | `ee55ad0e31a50d42` | identical |
| 100-step nonlinear heat-flux trace SHA | `f0c5711f6ad61c17` | identical | differs run to run (two base runs differ) | same class |
| final heat flux | `0x1.e0af0faac1b00p-18` | identical | `0x1.e0af0faac1afdp-18` / `...aff` | `0x1.e0af0faac1afdp-18` |
| window value | `0x1.6a3a9aa898351p-34` | identical | `0x1.6a3a9aa89833ep-34` | identical |
| window gradient d<Q>/d(tprim) | `-0x1.22ee865d4c2b7p-40` | identical | `-0x1.22ee865d4c30dp-40` | identical |

On XLA:CPU every fingerprint matches bitwise. On CUDA the linear eigenpair and
the window value and gradient match bitwise. The nonlinear trace is not
reproducible on CUDA even between two base runs; the head's final value equals
the first base run's bit for bit.
