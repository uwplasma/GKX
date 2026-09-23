Testing
=======

GKX tests protect contracts, not line counts. A test belongs in the suite when
it checks one of:

- an implemented equation or a reduced physical limit;
- a numerical method: convergence order, conservation, free-energy identity;
- a geometry, normalization, or diagnostic convention;
- a benchmark artifact and its documented fit/window policy;
- an autodiff contract against finite differences, tangents, or an adjoint
  identity;
- a regression for a bug found in parity, restart, runtime, plotting, or
  geometry-adapter work.

The package-wide coverage target is 95%, enforced on the combined wide-coverage
run (see `CI jobs`_). Coverage is a guardrail; the physics and numerics
contracts above are the objective.

Test tree
---------

``tests/README.md`` owns the layout. Do not add flat ``tests/test_*.py`` files;
prefer one parametrized module per physical or workflow contract.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Directory
     - Contents
   * - ``unit/``
     - Fast contracts for package code, by domain: ``core``, ``geometry``,
       ``linear``, ``nonlinear``, ``operators``, ``solvers``, ``diagnostics``,
       ``objectives``, ``parallel``, ``quasilinear``, ``api``.
   * - ``integration/``
     - Executable behavior: runtime TOML loading and runs
       (``runtime/test_runtime_config.py``, ``runtime/test_runtime_runner.py``),
       CLI, saved artifacts, shipped examples (``examples/test_examples.py``),
       adaptive eigenmodes.
   * - ``validation/``
     - Literature-anchored physics gates (``physics_gates/``), benchmark
       contracts, nonlinear transport windows, quasilinear calibration and
       guardrails, stellarator/VMEC policy.
   * - ``tools/``
     - Tests for ``tools/comparison`` and ``scripts/profiling`` scripts.
   * - ``release/``
     - Repository-policy gates (``test_release_gates.py``) and the evidence
       ledger contract (``test_evidence_ledger.py``).
   * - ``support/``
     - Shared helpers; not package API.

Representative contracts and where they live:

- Hermite/Laguerre ladder identities, quasineutrality, and streaming
  (``gkx.operators.linear.moments``): ``tests/unit/linear/test_linear.py``,
  ``tests/unit/linear/test_linear_moments_invariants.py``,
  ``tests/unit/operators/test_linear_streaming.py``.
- Term-wise RHS equivalence (``gkx.terms.assemble_rhs_cached`` against
  ``gkx.operators.linear.rhs.linear_rhs_cached``):
  ``tests/unit/operators/test_terms_assembly.py``.
- Term toggles, drift activation, diamagnetic drive vanishing at
  :math:`k_y=0`, ``rho_star`` scaling of cached :math:`k_y`, and the end-damping
  taper: ``tests/unit/linear/test_linear.py``.
- Collision matrices and free-energy identities, end damping, Hermite
  hierarchy, geometry conventions: ``tests/validation/physics_gates/``.
- Growth-rate fit windows (``gkx.diagnostics.growth_rates``):
  ``tests/unit/diagnostics/test_analysis.py``.
- Krylov/shift-invert internals, mode-family targeting, fallback policy:
  ``tests/unit/solvers/test_linear_krylov_core.py``.
- Nonlinear bracket sign, real-FFT path, flutter, electromagnetic split:
  ``tests/unit/nonlinear/test_nonlinear_exb.py``; Hermitian projection, RK
  variants, collision splitting, IMEX, flow-shear remap and window-gradient
  checks: ``tests/unit/nonlinear/test_nonlinear_helpers_extra.py``.
- Serial-vs-parallel numerical identity: ``tests/unit/parallel/``; the tracked
  large-run scaling artifacts: ``tests/unit/parallel/test_parallel_artifacts.py``.

Running tests
-------------

.. code-block:: bash

   JAX_ENABLE_X64=true GKX_X64=1 pytest

``pytest.ini`` sets ``addopts = -q --maxfail=1 --disable-warnings -m "not
slow"``, so a bare ``pytest`` runs everything except tests marked ``slow``.
CI sets ``JAX_ENABLE_X64=true`` and ``GKX_X64=1`` for every test step except the
two float32 steps in ``python-floor``; set them locally when reproducing CI.

Markers:

- ``slow``: excluded by default *and* by the wide-coverage shards, so a
  ``slow`` test leaves the coverage denominator. Do not use it for a test that
  only takes a minute.
- ``integration``: a label for end-to-end tests; it does not deselect anything.
- ``gpu``: exercises a device-specific execution path.

For a bounded local loop, use the per-file runner. It applies a per-file
timeout and a whole-run timeout (both 300 s by default) and reports files it
did not reach as ``not_run(total_timeout)``:

.. code-block:: bash

   python scripts/checks/run_test_gates.py fast
   python scripts/checks/run_test_gates.py fast --total-timeout 0   # full sequential pass

Opt-in parity gates skip unless their environment variable is set:

.. code-block:: bash

   # restart parity: resumed nonlinear run equals the continuous run
   pytest -q tests/integration/runtime/test_runtime_runner.py -k restart_gate
   # CPU vs GPU short nonlinear trajectory
   GKX_DEVICE_PARITY=1 pytest -q tests/unit/parallel/test_parallel_core.py -k cpu_gpu
   # VMEC -> *.eik.nc regenerated twice, arrays bitwise identical
   GKX_VMEC_FILE=/path/to/wout.nc pytest -q tests/unit/geometry/test_vmec_eik.py -k roundtrip

CI jobs
-------

``.github/workflows/ci.yml`` runs these jobs on every push and pull request.
All Python jobs use 3.11. Each job installs for itself; the only ``needs:``
edge is ``wide-coverage`` on ``wide-coverage-shards``.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Job
     - What it runs
   * - ``repo-hygiene``
     - ``ruff check`` and ``ruff format --check`` (ruff pinned from the
       ``dev`` extra); repository-size, package-architecture, parallel-scaling,
       quasilinear-guardrail, VMEC/Boozer differentiability-claim, and
       release-readiness checkers; then ``git diff --exit-code`` fails if the
       four regenerated tracked JSON artifacts (quasilinear guardrails,
       differentiability guard, technical status, release readiness) differ
       from the commit.
   * - ``python-floor``
     - Installs at ``requires-python`` (3.11), imports ``gkx`` and ``gkx.cli``,
       collects the whole suite, runs two float32 nonlinear gradient checks,
       and the ``tomllib`` portability gates.
   * - ``mypy``
     - ``mypy``.
   * - ``quick-tests``
     - Seven named shards: ``fundamentals-core``, ``release-artifacts``,
       ``model-artifacts``, ``linear-core``, ``runtime-core``,
       ``nonlinear-core``, ``parallel-autodiff`` (the last with four logical
       CPU devices).
   * - ``adaptive-eigensolver``
     - Paired eigenmode and implicit-AD gates against the released SOLVAX
       (``GKX_REQUIRE_PAIRED_SOLVAX=1`` turns a missing API into an error).
   * - ``fast-coverage``
     - Release-surface files; requires >= 95% line coverage on ``cli.py``,
       ``plotting.py``, ``workflows/runtime/artifacts.py``, and
       ``analysis.py``.
   * - ``wide-coverage-shards``
     - 24 shards of ``scripts/checks/run_test_gates.py wide-coverage`` with
       ``-o addopts= -m "not slow"``.
   * - ``wide-coverage``
     - Combines the shard data (every shard must report), enforces >= 95%
       package-wide, and writes
       ``docs/_static/validation_coverage_manifest_summary.json`` through
       ``check_validation_coverage_manifest.py --enforce-package-coverage``.
   * - ``docs-and-packaging``
     - Coverage-manifest summary, quasilinear ``calibration-inputs`` audit,
       ``sphinx -W``, sdist/wheel build and metadata check, installed-wheel
       smoke test and the documented first run.
   * - ``ci-required``
     - Runs with ``if: always()`` and fails if any job above reports
       ``failure``, ``cancelled``, or ``skipped``. Branch protection should
       require this single check.
   * - ``nightly-full``
     - Only on ``workflow_dispatch`` with ``run_nightly``: ``mypy``, docs
       linkcheck (fails on 404/410 only), the full ``pytest`` suite, and the two
       core coverage gates below.

Core coverage gates in ``nightly-full``:

.. code-block:: bash

   # gkx.terms >= 90%
   pytest -q tests/unit/operators/test_terms_assembly.py \
          tests/unit/operators/test_linear_streaming.py \
          tests/unit/operators/test_terms_fields.py \
          tests/unit/solvers/test_time_integrators.py \
          tests/unit/nonlinear/test_nonlinear_exb.py \
          tests/unit/nonlinear/test_nonlinear.py \
          --cov=src/gkx/terms --cov-fail-under=90
   # solvers_linear_krylov.py >= 90%, read from coverage-core.xml
   pytest -q tests/unit/solvers/test_linear_krylov_core.py \
          --cov=src/gkx --cov-report=xml:coverage-core.xml

The wide gate runs locally with the same helper. One process:

.. code-block:: bash

   python scripts/checks/run_test_gates.py wide-coverage \
     --shards 24 --timeout 1800 --fail-under 95 \
     --pytest-arg=-o --pytest-arg=addopts= \
     --pytest-arg=-m --pytest-arg="not slow"

Or shard by shard, then combine, as CI does:

.. code-block:: bash

   python -m coverage erase
   for shard in $(seq 1 24); do
     python scripts/checks/run_test_gates.py wide-coverage \
       --shards 24 --timeout 1800 --only-shard "${shard}" \
       --keep-existing-coverage --skip-combine \
       --pytest-arg=-o --pytest-arg=addopts= \
       --pytest-arg=-m --pytest-arg="not slow"
   done
   python scripts/checks/run_test_gates.py wide-coverage \
     --shards 24 --combine-only --fail-under 95

The shard that owns the logical-CPU file runs all of its device gates at four
devices in one command, which is why CI uses ``--timeout 1800``.

Contracts a local ``pytest`` cannot catch
-----------------------------------------

These are enforced by manifest scripts, the docs build, or a different
interpreter. Run them before pushing.

1. **Repo-wide line budget.** The physical line count of ``src/gkx/**/*.py``
   may not exceed the ``installable_source_python_lines`` baseline in
   ``tools/package_architecture_manifest.toml``. To raise it, edit the baseline
   with an inline ``# old -> new: reason`` comment.
2. **Per-module cap.** ``[complexity_policy]`` in the same manifest sets
   ``default_max_lines = 1200`` per module and ``public_facade_max_lines =
   500`` for the listed facades. Going over needs a reviewed
   ``[[complexity_policy.exceptions]]`` entry.
3. **Coverage-owner manifest.** Every module under ``src/gkx`` must appear in
   ``tools/validation_coverage_manifest.toml`` as a direct ``[[modules]]`` row,
   inside a row's ``owned_modules``, or in
   ``coverage_inventory.excluded_modules``. Modules of 2000 or more source lines
   need a direct row. ``fast_tests`` and ``artifact_paths`` name files, not
   directories, and no list repeats an entry.
4. **API reference.** Every ``.. automodule:: gkx...`` in ``docs/api.rst`` must
   resolve to a real module tracked by the coverage manifest, and the docs build
   with ``-W``.
5. **Python floor.** ``python-floor`` collects the suite on 3.11; TOML is read
   with the standard-library ``tomllib`` and a release gate forbids the
   ``tomli`` backport.

The checklist in CI order:

.. code-block:: bash

   python scripts/checks/check_repository_size_manifest.py
   python scripts/checks/check_package_architecture_manifest.py
   python scripts/checks/check_validation_coverage_manifest.py
   mypy
   pytest -q --collect-only --disable-warnings > /dev/null
   pytest tests/release/test_release_gates.py
   python -m sphinx -W -b html docs docs/_build/html

Focused checks for items 3 and 4:

.. code-block:: bash

   pytest tests/release/test_release_gates.py \
     -k "documented_public_api or large_modules_have_direct_manifest_rows"

Numerical stack
---------------

``pyproject.toml`` requires ``jax>=0.10.1``, ``jaxlib>=0.10.1``, and
``solvax>=0.22.0``. The JAX floor is hard: ``gkx.objectives.core`` calls
``lax_linalg.eig(..., enable_eigvec_derivs=True)``, which first shipped in JAX
0.10.1; older JAX fails the solver-objective tests with ``TypeError: eig() got
an unexpected keyword argument``. Every CI job prints the resolved ``jax``,
``jaxlib``, and ``numpy`` versions ("Print benchmark stack"); reproduce a parity
regression on that stack, with ``JAX_ENABLE_X64=1``, before changing solver
logic. Near-marginal lanes (TEM, ETG scans, some imported-linear stellarator
cases) can move under a different JAX/NumPy combination.

Validation gate tooling
-----------------------

Artifact gates share one JSON report convention (observable, reference,
absolute/relative tolerance, pass/fail) from
``gkx.diagnostics.validation_gates``: ``evaluate_scalar_gate``,
``observed_order_gate_report``, ``branch_continuity_gate_report``,
``eigenfunction_gate_report``, ``linear_metrics_gate_report``,
``nonlinear_window_gate_report``, and ``zonal_response_gate_report``. Artifacts
that set ``gate_index_include=false`` are documented diagnostics, not release
gates.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Command
     - Artifact and current result
   * - ``scripts/checks/check_validation_coverage_manifest.py gate-index``
     - ``docs/_static/validation_gate_index.json``: 17 of 18 tracked reports
       pass; ``quasilinear_model_selection_status.json`` is open.
   * - ``scripts/artifacts/generate_linear_reference_overlays.py kbm`` (retired; :ref:`retired-generators`) / ``w7x``
     - ``docs/_static/reference_modes/*_eigenfunction_reference_overlay_ky0p3000.json``;
       gate: overlap >= 0.95, relative :math:`L^2` <= 0.25. KBM 0.999985 /
       0.00721; W7-X 0.9999999994 / 3.33e-5.
   * - ``scripts/artifacts/build_linear_validation_artifacts.py observed-order`` (retired subcommand; :ref:`retired-generators`)
     - ``docs/_static/cyclone_resolution_observed_order.json`` (Cyclone
       :math:`k_y=0.30`, ``(Nl,Nm)`` = (4,8), (6,12), (12,24), (16,32)); passes.
   * - ``scripts/artifacts/build_linear_validation_artifacts.py kbm-branch`` (retired subcommand; :ref:`retired-generators`)
     - ``docs/_static/kbm_branch_gate_summary.json``; passes the adjacent
       jump and successive-overlap gates.
   * - ``scripts/comparison/compare_gx_nonlinear.py diagnostics --summary-json``
     - ``docs/_static/nonlinear_{cyclone,cyclone_miller,kbm,hsx,w7x}_gate_summary.json``;
       all pass the release-window gate.
   * - ``scripts/artifacts/build_zonal_flow_artifacts.py miller-panel`` (retired; :ref:`retired-generators`)
     - ``docs/_static/miller_zonal_response_pilot.json`` (Merlo Case III,
       ``Nm=144``, ``dt=0.0025``). ``gate_report`` pins GKX's converged values;
       ``literature_comparison`` fails on residual (0.206 vs 0.19, tolerance
       0.015) and GAM frequency (2.345 vs 2.24, tolerance 0.10) and passes on
       damping (-0.184 vs -0.17, tolerance 0.03).
   * - ``scripts/artifacts/build_w7x_zonal_validation_artifacts.py`` (retired; :ref:`retired-generators`) and
       ``scripts/artifacts/build_w7x_zonal_reference_artifacts.py compare`` (retired; :ref:`retired-generators`)
     - ``docs/_static/w7x_zonal_reference_compare.json``: open. Time coverage
       passes at all four :math:`k_x\rho_i`; the residual passes only at 0.05;
       the late-envelope gate fails at all four.
   * - ``scripts/comparison/build_exact_state_audit.py report`` (retired; :ref:`retired-generators`)
     - ``docs/_static/w7x_exact_state_audit.json``: max finite pointwise
       relative error 4.62e-5 against a 1e-4 gate.
   * - ``scripts/checks/check_nonlinear_transport_gates.py runtime-outputs``
     - Checks every ``*.out.nc`` for ``Grids/time``, the requested heat-flux
       diagnostic, finite monotone samples, and optional ``tmin/tmax``
       coverage; fails closed on restart-only output.
   * - ``scripts/checks/check_nonlinear_optimization_gates.py production-guard``
     - Blocks production nonlinear turbulent-flux optimization claims until
       optimized equilibria have replicated post-transient window audits.
       Reference negative case:
       ``docs/_static/strict_qa_top12_edge_matched_nonlinear_transport.json``
       (0.58% reduction, uncertainty z-score 0.20; matched gate fails).
   * - ``scripts/checks/check_nonlinear_transport_gates.py matrix-portfolio``
     - Selects only a passing broad matrix family; current negative ledger:
       ``docs/_static/broad_nonlinear_transport_matrix_negative_evidence.json``.
   * - ``scripts/checks/check_vmec_boozer_gates.py high-grid-admission``
     - Admits an external-VMEC holdout at high grid only when the low-grid
       failure is the sole failure and time-horizon and replicate gates pass;
       it does not claim full ``n48/n64/n80`` convergence.
   * - ``scripts/checks/check_quasilinear_promotion_guardrails.py``
     - ``docs/_static/quasilinear_promotion_guardrails.json``; its
       ``calibration-inputs`` mode runs in ``docs-and-packaging``.

``check_quasilinear_promotion_guardrails.py`` fails if a promoted quasilinear
report lacks train/holdout points, finite nonlinear window statistics, a
passed holdout gate, or calibration-policy metadata, and it scans the
claim-scope wording in the README and docs. This is not a runtime/TOML
absolute-flux predictor; it is a fast metadata and wording guard.

Every shipped TOML that reaches the nonlinear runtime either pins ``run_to`` in
its ``[time]`` block or is listed in ``tests/release/test_release_gates.py``
with its measured first-chunk stop decision. A zero-gradient relaxation run
under the default ``run_to = "saturation"`` would stop in the first chunk, so
the Merlo deck pins ``run_to = "t_max"`` and its artifact gates
``trace_completeness = 1``.
