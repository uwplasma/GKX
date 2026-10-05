# ARCH-C inventory and third contraction (2026-10-04)

Lane: ARCH-C (plan.md G.6 P4; archived plan §8). Base: `origin/main` at
`2987bb705` (2.5.0). Branch: `arch/contract-3`. Refreshes
`plan/research/2026-09-27-arch-b/INVENTORY.md`.

Reproduce: `python plan/research/2026-10-04-arch-c/inventory.py .` and
`python plan/research/2026-10-04-arch-c/fingerprint.py .` (ARCH-B harness plus
a kinetic-electron `imex-ars3` run).

## 1. Before and after

| Measure | Base `2987bb705` | This branch |
| --- | ---: | ---: |
| `src/gkx` Python files / lines | 136 / 75,904 | 129 / 67,191 |
| `tests` Python files / lines | 30 / 82,803 | 30 / 74,151 |
| `scripts` Python files / lines | 59 / 43,611 | 63 / 48,382 |
| Import cycles (function-local imports counted) | 1 | 0 |

Repository-wide Python lines fall by about 12.6k (src -8.7k, tests -8.7k,
scripts +4.8k). About 4.8k of the src reduction is relocation: report and
gate modules, campaign helpers and test-only reference kernels that nothing in
the package reached now live in `scripts/` (`scripts/checks/_gates/`, the
campaign and comparison scripts). The rest is deletion.

## 2. What was done

- **Last cycle broken; solve layer collapsed.** The dependency-injection
  records (`*Deps`, `_PATCHABLE_RUNTIME(_COMMAND)_GLOBALS`, `*_fn` parameters)
  are gone; owners call collaborators directly. `run_runtime_linear` lives in
  `workflows.linear`, `run_runtime_nonlinear` and `prepare` in
  `workflows.nonlinear`, `run_runtime_scan` in `orchestration_scan`;
  `gkx.runtime` is a 69-line facade. `orchestration_artifacts` merged into
  `workflows/runtime/artifacts.py`. Test monkeypatches were audited and now
  patch every solve-layer module that binds the name
  (`tests/support/helpers.py::patch_runtime`).
- **Report modules out of the package:** `diagnostics.{validation_gates,
  transport_windows, quasilinear_calibration, zonal_validation}` and
  `artifacts.zonal_plots` -> `scripts/checks/_gates/` (28 `gkx.*` lazy exports
  removed; no example or tutorial used them).
- **Dead code deleted** (reached by nothing outside tests): auto fit-signal
  selector family, Hermite helpers, windowed nonlinear metrics, eigenfunction
  comparison, nperiod contracts, `cfl_term_contributions`, cached shift-invert,
  and others; duplicate shift-axis, rk3/rk4 wrappers and collision pytree
  methods merged; the nonlinear bracket contexts, IMEX option bundles and the
  gradient-validation report collapsed.
- **Merged** `growth_windows` into `growth_rates` (single consumer, seven names).
- **Tests:** runtime, CLI, artifact, release-gate, benchmarking, comparison,
  geometry and nonlinear-helper tests parametrized, shared builders moved to
  `tests/support/helpers.py`; tests that only asserted re-exports or forwarding
  removed. No E1-E3 math or physics test removed.

## 3. Fingerprints (office, XLA:CPU, JAX 0.11.2, float64)

Linear Cyclone eigenpair, 100-step nonlinear heat-flux trace, window gradient,
quasilinear flux at two k_y, and a kinetic-electron `imex-ars3` run: bitwise
identical between base and branch (values in the PR body).

## 4. Remaining and next

- Cohesion gate: two single-consumer split modules remain
  (`solvers_nonlinear_diagnostics` -> `solvers_nonlinear_diagnostic_integration`,
  owned by the perf lane; `workflows/runtime/artifacts` -> `commands`, over the
  line budget together). Low-cohesion count rose 5 -> 8 because deleting dead
  helpers removed intra-module references (modes, cache_arrays,
  benchmarking_shared).
- Targets not met: src 129 files / 67k lines (target 45 / 45k), tests 74k
  (target 35k). Next ranks unchanged from ARCH-B: fold the diagnostics
  timesteppers into their integrators (needs the perf lane's files), artifacts
  -> `io/`, terms + operators -> `physics/`.
