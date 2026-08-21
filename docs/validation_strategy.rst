Validation And Coverage Strategy
================================

Purpose
-------

The refactor branch uses a traceability manifest to keep software coverage,
physics validation, and publication artifacts tied together. The package-wide
target is still 95% coverage, but coverage is treated as an engineering
guardrail. A test only counts as useful when it protects an equation, a
numerical method, a diagnostic convention, an artifact contract, or an
autodiff/performance guarantee.

The machine-readable manifest lives at
``tools/validation_coverage_manifest.toml`` and is checked by
``tools/release/check_validation_coverage_manifest.py``. Each critical module entry
records:

- the source file and owning refactor lane;
- reference anchors from the literature, independent-code comparisons, or
  documented numerical methods;
- physics contracts that should remain true across refactors;
- numerical contracts such as observed order, conservation, window policy, or
  finite-value handling;
- fast tests that run locally;
- shipped artifacts or gate reports that document the validation state;
- the next tests needed to reach the package-wide 95% target.

Claim-scope synchronization
---------------------------

Validation status and scientific claims are intentionally separate from raw
coverage. Use :doc:`release_scope` as the human-readable claim ledger, and keep
it synchronized with ``benchmarks/references/gkx_1_7_release_contract.json``
and the validation coverage manifest. A passing test or coverage line is not
enough to promote a physics or
performance claim unless the relevant artifact also records the observable,
reference, tolerance, and accepted claim level. Likewise, an example that runs
successfully is still only a release claim when it is labeled as release-gated
and tied to the relevant artifact; otherwise keep it framed as a stress lane,
pilot, or deferred manuscript lane.

How to check the manifest
-------------------------

Run:

.. code-block:: bash

   python tools/release/check_validation_coverage_manifest.py

For CI or release bookkeeping, write a JSON summary:

.. code-block:: bash

   python tools/release/check_validation_coverage_manifest.py \
     --out-json docs/_static/validation_coverage_manifest_summary.json

The wide-coverage CI job attaches measured coverage from the combined Cobertura
report:

.. code-block:: bash

   python tools/release/check_validation_coverage_manifest.py \
     --coverage-xml coverage-wide.xml \
     --enforce-package-coverage \
     --out-json docs/_static/validation_coverage_manifest_summary.json

This fails if total package coverage drops below the manifest target and records
direct/owned module coverage gaps for the next refactor tranche. Module-level
coverage enforcement is intentionally a separate switch so the release gate can
remain package-wide while the manifest still exposes specific debt.

The manifest checker's ``gate-index`` mode complements its coverage audit. The gate
index reports which validation artifacts currently pass. The coverage manifest
reports whether the remaining refactor and testing work is traceable to
physics, numerics, artifacts, and tests.

Finalization sequence
---------------------

The remaining work should be closed in this order.

1. **Freeze module contracts before moving code.**
   For each large file, write or update tests for current public behavior, then
   extract only cohesive helpers. Keep public facades stable until examples,
   docs, and benchmark scripts use the new module boundaries.

2. **Keep the closed complexity budgets enforced.**
   Runtime, linear, nonlinear, benchmark, validation, imported-VMEC, and
   nonlinear-gradient ownership are below their architecture budgets with no
   remaining named exceptions. New work must preserve those budgets and the
   package/file-count no-regression gates.

3. **Turn open or deferred physics lanes into explicit gates.**
   Literature-facing lanes should produce JSON/CSV/PNG/PDF artifacts with the
   same observable, window, tolerance, and source recorded in metadata. The
   current priority list is W7-X zonal recurrence/damping, W7-X
   fluctuation-spectrum experimental transfer functions, W7-X TEM /
   kinetic-electron and multi-flux-tube validation, production nonlinear
   transport-gradient gates, broader optimized-equilibrium nonlinear audits
   beyond the selected QA candidate, and any stricter case-specific nonlinear
   window-statistics retuning that should become a paper claim.

4. **Replace coverage gaps with physics or numerics tests.**
   Do not add shallow import-only tests to chase the number. Prefer tests for
   ladder identities, field-solve limits, bracket antisymmetry, diagnostic
   normalization, strict JSON output, observed-order gates, restart invariants,
   and artifact reload behavior.

5. **Validate differentiability explicitly.**
   Autodiff examples should carry finite-difference or tangent checks, inverse
   recovery diagnostics, and covariance/uncertainty estimates. The Phase-A
   ``vmex`` and ``booz_xform_jax`` bridge now carries a tracer-safe
   geometry-observable sensitivity check, a two-parameter inverse design, and
   local UQ covariance diagnostics. Reduced linear, quasilinear, and
   nonlinear-window-estimator derivatives now have AD/finite-difference gates,
   but production nonlinear transport derivatives still need long-window
   heat-flux convergence, local-gradient conditioning, and broader
   optimized-equilibrium audits before they are used for stellarator heat-flux
   optimization claims.

6. **Keep performance measurements separated from validation.**
   Performance panels should report cold compile, warm runtime, memory, output
   time, and parallelization speedup separately. Parallelization gates should
   first target independent scans, UQ ensembles, and sensitivity batches where
   strong scaling is scientifically useful and robust.

7. **Raise CI thresholds in phases.**
   Keep default tests under the five-minute local budget. Enforce fast critical
   modules first, then broad package coverage on the wide CI lane, then manual
   office/GPU parity and performance sweeps. The merge target is package-wide
   coverage at or above 95% with the validation manifest and gate index both
   passing.

Release readiness criteria
--------------------------

The refactor branch is ready to merge when:

- package-wide coverage is at least 95% on the wide lane;
- all high-priority manifest modules have their planned extraction tests;
- the tracked validation gate index is passing or open lanes are explicitly
  labeled as non-release exploratory artifacts;
- shipped examples still run and plot from output files;
- W7-X, HSX, Cyclone, Cyclone-Miller, KBM, ETG, Miller, and VMEC examples have
  current documentation that labels each lane as release-gated, stress, pilot,
  or deferred and matches the tracked artifacts;
- autodiff examples validate gradients and inverse/UQ outputs;
- the performance manifest points to current runtime/memory panels, CPU/GPU
  profiler artifacts, and numerical-identity gates for every performance
  claim made in the README/docs;
- docs build with warnings treated as errors;
- package build, release workflow, and PyPI metadata checks pass.
