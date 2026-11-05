Testing
=======

GKX tests protect contracts, not line counts. A test belongs in the suite when
it checks one of:

- an implemented equation or a reduced physical limit;
- a numerical method: convergence order, conservation, free-energy identity;
- a geometry, normalization, or diagnostic convention;
- an analysis or benchmark fit/window policy;
- an autodiff contract against finite differences, tangents, or an adjoint
  identity;
- a regression for a bug found in parity, restart, runtime, plotting, or
  geometry-adapter work.

The package-wide coverage target is 95%, enforced on the combined wide-coverage
run (see `CI and release checks`_). Coverage is a guardrail; the physics and numerics
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
       CLI, saved artifacts, shipped examples (``examples/test_examples.py``, which runs every gallery ``run.py``),
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
  ``tests/unit/linear/test_linear.py``,
  ``tests/unit/operators/test_terms_fields.py``.
- Term-wise RHS equivalence (``gkx.terms.assemble_rhs_cached`` against
  ``gkx.operators.linear.rhs.linear_rhs_cached``):
  ``tests/unit/operators/test_terms_fields.py``.
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
- Serial-vs-parallel numerical identity: ``tests/unit/parallel/``.

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
   GKX_VMEC_FILE=/path/to/wout.nc pytest -q tests/unit/geometry/test_geometry.py -k roundtrip

CI and release checks
---------------------

CI runs the full test groups and combines 24 wide-coverage shards. The package
coverage requirement is 95%; keep numerical and physical contracts in the
coverage run rather than replacing them with generated-result assertions.

After combining coverage data, enforce the source coverage gate:

.. code-block:: bash

   python scripts/check.py validation-coverage \
     --coverage-xml coverage-wide.xml --enforce-package-coverage

``python scripts/check.py readiness`` checks current source, package and CI
contracts. Report checkers accept explicitly supplied run outputs for
validation campaigns; a stored summary is not a substitute for the live
source and numerical tests. Keep test reports, coverage data and benchmark
outputs outside version control.
