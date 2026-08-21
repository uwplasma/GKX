# Tests

This tree protects the promoted GKX product: solver physics, numerical
methods, differentiability, executable workflows, release gates, and documented
artifact schemas. It is not an archive of every historical branch or one-off
tool wrapper.

The refactor target is fewer than 100 Python test files while preserving at
least 95% package-wide coverage and the physics/numerics gates used for release.

## Ownership

- `unit/`: fast contracts for pure package code: grids, geometry, operators,
  solvers, diagnostics, objectives, parallel helpers, and quasilinear formulas.
- `integration/`: user-facing executable/runtime behavior, examples, restart
  paths, plotting, progress output, and saved-output contracts.
- `validation/`: literature-anchored physics gates, convergence checks,
  benchmark comparisons, nonlinear transport windows, quasilinear calibration,
  and stellarator validation policy.
- `tools/`: tests for repository-maintenance scripts in `tools/artifacts`,
  `tools/campaigns`, `tools/comparison`, `tools/profiling`, and
  `tools/release`.
- `release/`: bounded repository-policy gates used by CI or release readiness.
- `support/`: shared test helpers that are not part of the package API.

Do not add flat `tests/test_*.py` files. Keep only `conftest.py` at the root.

## Consolidation Rules

Prefer one parametrized test module per physical or workflow contract. Avoid
one test file per script unless the script is complex enough to be a separate
product surface.

Good test-family names:

- `test_runtime_progress_and_outputs.py`
- `test_benchmark_fit_policy.py`
- `test_linear_validation_artifact_reports.py`
- `test_nonlinear_transport_window_gates.py`
- `test_quasilinear_calibration_reports.py`

Poor test-family names:

- names that preserve a deleted helper or historical branch;
- names that duplicate a tool script one-to-one without testing a broader
  contract;
- names that encode external-code terminology outside an explicit comparison
  or benchmark context.

When reducing test count, keep or strengthen assertions that check:

- conservation, symmetry, normalization, convergence, and observed-order
  behavior;
- serial-vs-parallel numerical identity;
- finite-difference/JVP/VJP agreement and conditioning diagnostics;
- artifact schemas and fail-closed release guards;
- documented executable behavior.

Remove tests that only preserve retired compatibility paths, obsolete examples,
or unpromoted reduced/synthetic workflows.
