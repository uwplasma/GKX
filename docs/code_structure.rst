Code Structure
==============

This page says where each equation and each public surface lives, and the rules
that keep the tree that way. :doc:`architecture` is the module map, and
:doc:`api` documents every module.

Package policy
--------------

``tools/package_architecture_manifest.toml`` is the package-topology policy:
module layout, the per-module line cap, and the repo-wide line budget (see
:doc:`testing`). Add a module only when it replaces several existing files or
isolates a measured JAX compilation or performance boundary. Prefer the domain
packages (``operators``, ``objectives``, ``parallel``, ``diagnostics``,
``workflows``, ``artifacts``) to new root-level ``runtime_*``,
``nonlinear_*``, ``vmec_*``, ``quasilinear_*``, or ``benchmark_*`` modules.
Reusable metrics belong in ``gkx.diagnostics``; campaign orchestration and
admission policy belong in ``tools``, not in the installable package. Source
modules use physics and numerical names; external-code names and raw reference
handling stay in root ``benchmarks/`` and ``tools/comparison``.

Repository roles
----------------

- ``examples/``: small, copyable user workflows.
- ``benchmarks/``: benchmark drivers and compact TOML inputs, plus
  ``benchmarks/results/manifest.toml``. Raw solver products stay out of git;
  promoted results appear in :doc:`benchmarks`.
- ``tools/``: maintainer commands, by purpose: ``artifacts``, ``campaigns``,
  ``comparison``, ``profiling``, ``release``. Artifact builders are grouped by
  validation family, one command with subcommands per family.
- ``tests/``: automated gates that run without raw generated outputs (see
  :doc:`testing`).

A file that fits none of these roles belongs on an experiment branch, not on
``main``.

Public and internal modules
---------------------------

Examples, scripts, and external users may rely on:

- the modules documented in :doc:`api`, starting with the root ``gkx``
  registry;
- ``gkx.config``, ``gkx.runtime``, ``gkx.geometry``, ``gkx.cli``;
- ``gkx.artifacts``, ``gkx.artifacts.plotting``,
  ``gkx.workflows.runtime.artifacts``;
- ``gkx.parallel`` and ``gkx.operators.nonlinear.parallel``;
- documented scripts under ``benchmarks/``, ``examples/``, and ``tools/``.

These may move as long as public behavior and tests are unchanged:

- ``gkx.terms.*``;
- ``gkx.workflows.runtime.*`` other than ``artifacts``;
- the low-level adapters and backend bridges inside ``gkx.geometry``.

Package facades (``gkx.geometry``, ``gkx.parallel``, ``gkx.operators.linear``,
``gkx.operators.nonlinear``, ``gkx.diagnostics.analysis``) re-export from the
owning modules; ``gkx``, ``gkx.api``, ``gkx.parallel``, and the two
``gkx.operators`` packages resolve names lazily from static tables.
Dependency-light contracts such as ``gkx.parallel.decomposition`` must not
import JAX-heavy solver stacks through a package initializer.

Equations to code
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 38 32

   * - Equation or method
     - Source (under ``src/gkx``)
     - Tests
   * - Hermite/Laguerre basis, gyroaverage :math:`J_\ell`, polarization
     - ``core_velocity.py``
     - ``tests/unit/core/test_core_numerics.py``
   * - Spectral grid, :math:`k_y` layout, dealiasing
     - ``core_grid.py``, ``core_ky_layout.py``
     - ``tests/unit/core/``
   * - Flux-tube geometry coefficients
     - ``geometry/`` (paths in :doc:`architecture`)
     - ``tests/unit/geometry/``,
       ``tests/validation/physics_gates/test_geometry_physics_contracts.py``
   * - Field equations: quasineutrality, Ampère, perpendicular pressure balance
     - ``terms/fields.py`` (``solve_fields``); ``operators/linear/moments.py``
       (``quasineutrality_phi``, ``build_H``)
     - ``tests/unit/operators/test_terms_fields.py``,
       ``tests/unit/linear/test_linear_moments_invariants.py``
   * - Parallel streaming, Hermite closure, mirror force
     - ``terms/linear_terms.py``, ``operators/linear/streaming.py``
     - ``tests/unit/operators/test_linear_streaming.py``,
       ``tests/validation/physics_gates/test_hermite_hierarchy_physics.py``
   * - Curvature and grad-B drifts, diamagnetic drive
     - ``terms/linear_terms.py``
     - ``tests/unit/linear/test_linear.py``
   * - Collision operators (Dougherty, Sugama and Coulomb six-moment,
       tabulated linearized matrices)
     - ``operators/linear/dissipation.py``, ``operators/linear/collisions.py``,
       ``operators/linear/collision_tables.py``,
       ``operators/linear/collision_factory.py``, ``operators/collision.py``
     - ``tests/validation/physics_gates/test_collision_physics.py``,
       ``tests/unit/operators/test_linear_collisions_coverage.py``
   * - Hypercollisions, hyperdiffusion, end damping
     - ``operators/linear/dissipation.py``
     - ``tests/validation/physics_gates/test_end_damping_physics.py``
   * - Linked (twist-shift) boundary
     - ``operators/linear/linked.py``, ``geometry/core.py``
       (``twist_shift_params``)
     - ``tests/unit/linear/test_linear.py``
   * - Linear RHS assembly
     - ``terms/assembly.py`` (``assemble_rhs_cached``),
       ``operators/linear/rhs.py``, ``operators/linear/cache_builder.py``
     - ``tests/unit/operators/test_terms_assembly.py``
   * - ExB and electromagnetic (flutter) nonlinearity
     - ``terms/nonlinear.py``, ``operators/nonlinear/brackets.py``,
       ``operators/nonlinear/rhs.py``
     - ``tests/unit/nonlinear/test_nonlinear_exb.py``
   * - Linear time integration and implicit solves
     - ``solvers_linear_integrators.py``, ``solvers_time_explicit*.py``,
       ``solvers_linear_implicit.py``
     - ``tests/unit/solvers/test_time_integrators.py``
   * - Eigenmodes: Krylov, shift-invert, propagator
     - ``solvers_linear_krylov.py``, ``solvers_linear_krylov_algorithms.py``,
       ``solvers_linear_precond_pr3.py``,
       ``solvers_linear_adaptive_propagator.py``
     - ``tests/unit/solvers/test_linear_krylov_core.py``
   * - Nonlinear time integration (explicit, IMEX)
     - ``solvers_nonlinear_explicit.py``, ``solvers_nonlinear_imex.py``,
       ``solvers_nonlinear_state_integration.py``,
       ``solvers_nonlinear_diagnostic_integration.py``
     - ``tests/unit/nonlinear/test_nonlinear.py``,
       ``tests/unit/nonlinear/test_nonlinear_helpers_extra.py``
   * - Free energy, spectra, heat and particle fluxes
     - ``operators/moments.py``, ``operators/fluxes.py``
     - ``tests/unit/nonlinear/test_nonlinear.py``
   * - Growth-rate and frequency fits, fit windows
     - ``diagnostics/growth_rates.py``, ``diagnostics/growth_windows.py``,
       ``diagnostics/modes.py``
     - ``tests/unit/diagnostics/test_analysis.py``
   * - Quasilinear weights and calibration
     - ``diagnostics/quasilinear_transport.py``,
       ``diagnostics/quasilinear_calibration.py``
     - ``tests/unit/quasilinear/``
   * - Zonal-flow residual and GAM metrics
     - ``diagnostics/zonal_validation.py``
     - ``tests/validation/physics_gates/test_validation_gates.py``
   * - Implicit eigenvalue derivatives
     - ``objectives/eigen.py``, ``objectives/autodiff_validation.py``
     - ``tests/unit/objectives/test_autodiff_solver_objectives.py``

The operator equations themselves are in :doc:`operators`.

Runtime layers
--------------

1. Configuration and startup: ``config.py``, ``workflows/runtime/toml.py``,
   ``workflows/runtime/startup.py``, ``workflows/runtime/policies.py``,
   ``workflows/runtime/initial_phi.py``, ``workflows/runtime/resolution.py``.
2. Execution: ``runtime.py``, ``workflows/linear.py``,
   ``workflows/nonlinear.py``, ``workflows/runtime/chunks.py``,
   ``workflows/runtime/orchestration_scan.py``,
   ``workflows/runtime/warm_start.py``,
   ``workflows/runtime/parallel_nonlinear.py``.
3. Diagnostics and results: ``workflows/runtime/diagnostics.py``,
   ``workflows/runtime/diagnostic_arrays.py``,
   ``workflows/runtime/results.py``.
4. Artifacts: ``workflows/runtime/artifacts.py``,
   ``workflows/runtime/orchestration_artifacts.py``, ``artifacts/``.
5. Executable: ``cli.py`` builds the parser; ``workflows/runtime/commands.py``
   runs the linear, scan, nonlinear, and ``--plot`` commands and owns the
   programmatic case helpers ``run_linear_case`` and ``run_nonlinear_case``.
   Command flags resolve once into typed option records, so the precedence of
   CLI flag, TOML section, and runtime default is inspectable.

Benchmark execution uses the same path: canonical TOML inputs through
``run_runtime_linear`` and ``run_runtime_scan``. ``gkx.benchmarking_shared``
holds reference tables, CSV loaders, normalization constants, and
comparison-only defaults; it launches no solves.

``run_runtime_parameter_scan`` runs ordered continuation scans: the caller
supplies a named scalar axis and a pure ``RuntimeConfig`` update callback, and
the previous state can seed the next solve. ``candidate_options`` declares
per-point solver targets and ``select_candidate`` picks the one continued.
These scans are sequential by construction; independent ensembles use
``gkx.parallel``. ``refit_runtime_linear_trajectory`` refits a stored
time-domain linear result with another fit window or mode extractor
(``z_index``, ``max``, ``project``, ``svd``) without re-running it.

Coverage-owner manifest
-----------------------

``tools/validation_coverage_manifest.toml`` accounts for every module under
``src/gkx`` and is checked by
``tools/release/check_validation_coverage_manifest.py``. A direct
``[[modules]]`` row names the source path, owning lane, reference anchors,
physics and numerics contracts, fast tests, artifacts, and next tests.

- Add a direct row when a module has public imports, its own physics or
  numerics contract, separate artifact traceability, or high change risk.
- List a module in an owner row's ``owned_modules`` when it is a narrow helper
  whose contract the owner's tests fully exercise.
- ``coverage_inventory.excluded_modules`` holds package ``__init__`` files,
  version metadata, and the small ``solvers_linear``, ``solvers_nonlinear``,
  and ``solvers_time`` policy facades.
- Record new coverage debt in the owner row's ``next_tests``.

``tests/release/test_release_gates.py`` cross-checks the manifest: every
``.. automodule:: gkx.*`` in :doc:`api` must resolve to a source module or
package ``__init__`` the manifest accounts for, and any non-``__init__`` module
of 2,000 or more non-comment source lines needs a direct row.

Repository artifact hygiene
---------------------------

Git keeps source inputs, tests, small gate reports, and lightweight
documentation previews. Heavy runtime outputs, profiler traces, raw NetCDF
comparisons, and high-resolution exports live in release artifacts with
checksums and replay commands.

.. code-block:: bash

   python tools/release/check_repository_size_manifest.py audit --top 30
   python tools/release/check_repository_size_manifest.py
   python tools/release/check_repository_size_manifest.py release-artifacts

The audit separates tracked file size from ignored local roots (``tools_out/``,
``docs/_build/``, ``dist/``, virtual environments, caches).
``tools/repository_size_manifest.toml`` sets the tracked-size budget, the
largest allowed unlisted file, and the whitelist of retained large files.
``tools/release_artifact_manifest.toml`` records checksums, replay commands, and
destinations for large assets. The checker validates provenance only; it
neither uploads nor deletes. A ``move_to_release`` entry may be absent from git
only when the manifest records its ``release_tag``, ``release_url``, original
size, and SHA-256.

Preview compression:

.. code-block:: bash

   python tools/release/check_repository_size_manifest.py compress-previews --mode release --max-width 2200 --colors 192
   python tools/release/check_repository_size_manifest.py compress-previews --mode docs --min-bytes 300000 --max-width 1800 --colors 192

The first touches only release-manifest previews, so update
``tools/release_artifact_manifest.toml`` with the new sizes and checksums
afterwards. The second skips release-manifest paths by default. Rerun both
manifest checkers after either. History rewrites need a coordinated
maintenance window, because every collaborator must reclone or reset after a
force push.
