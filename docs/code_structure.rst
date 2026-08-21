Code Structure
==============

Purpose
-------

This page documents where the main physics, numerics, runtime, and artifact
surfaces live in the source tree. It is meant to make refactoring safer by
keeping the boundary between public APIs and internal implementation modules
explicit.

The long-term consolidation target is encoded in
``tools/package_architecture_manifest.toml``. New
refactor work should reduce package
surface area by merging, deleting, or moving code out of the installable
package. Add a new module only when it replaces multiple existing files or
isolates a measured JAX compilation/performance boundary. Domain packages such
as ``operators``, ``solvers``, ``objectives``, ``parallel``, ``diagnostics``,
``workflows``, and ``artifacts`` are preferred over more root-level
``runtime_*``, ``nonlinear_*``, ``vmec_*``, ``quasilinear_*``, or
``benchmark_*`` modules. Validation and campaign implementation should not live
in the installable package; reusable metrics belong in ``diagnostics`` and
long-run orchestration belongs in ``tools`` or root ``benchmarks``.

Repository Roles
----------------

The repository uses four non-source-code areas with distinct jobs:

- ``examples/`` contains small, copyable user workflows.
- ``benchmarks/`` contains reproducible benchmark drivers and compact benchmark
  inputs that researchers can run directly.
- ``tools/`` contains maintainer commands for artifacts, campaigns,
  comparisons, profiling, and releases.
- ``tests/`` contains automated gates that CI can run without relying on raw
  generated outputs. Consolidated test owners keep shared imports once at the
  file boundary; section markers describe the retained physics families but do
  not recreate former files through repeated imports or module aliases.

Files that do not fit one of those roles should be deleted from ``main`` or
moved to a draft experiment branch. Benchmark or comparison references to other
codes are allowed only in explicit benchmark/comparison contexts.

Artifact builders under ``tools/artifacts`` should be organized by validation
family rather than by one output file per script. For example, the VMEC/Boozer
aggregate holdout evidence now uses one
``build_vmec_boozer_aggregate_holdout_gate.py`` command with ``alpha`` and
``surface`` subcommands while preserving separate documented artifact names.

Public API vs Internal Modules
------------------------------

Public surfaces that examples, scripts, and external users are expected to rely
on:

- documented module pages in :doc:`api`
- ``gkx.artifacts``
- ``gkx.geometry``
- ``gkx.cli``
- ``gkx.runtime``
- ``gkx.workflows.runtime.config``
- ``gkx.workflows.runtime.artifacts``
- ``gkx.artifacts.plotting``
- ``gkx.parallel``
- ``gkx.operators.nonlinear.parallel``
- documented benchmark/example scripts under ``benchmarks/`` and ``examples/``
- documented repository-maintenance entry points under purpose-specific
  ``tools/`` subfolders

Internal modules that are free to move as long as the public behavior and tests
remain unchanged:

- ``gkx.terms.*``
- ``gkx.workflows.runtime.startup``
- ``gkx.workflows.runtime.diagnostics``
- ``gkx.workflows.runtime.diagnostic_arrays``
- ``gkx.workflows.runtime.initial_phi``
- ``gkx.workflows.runtime.chunks``
- ``gkx.workflows.runtime.results``
- ``gkx.workflows.runtime.commands``
- low-level geometry adapters and import bridges inside ``gkx.geometry``

Large refactor status for this push: the split runtime, diagnostics,
validation-gate, zonal-validation, and parallelization-policy modules are
documented as behavior-preserving infrastructure. They should not be cited as
new physics validation, production nonlinear optimization, or broad performance
claims unless the corresponding artifact gate in :doc:`release_scope` promotes
that claim separately.

Runtime Flow
------------

The executable-facing runtime path is split conceptually into four layers:

1. **configuration and startup**
   - ``workflows/runtime/config.py``
   - ``workflows/runtime/startup.py``
   - ``workflows/runtime/policies.py``
2. **solver execution**
   - ``runtime.py``
   - ``solvers/time/explicit.py``
   - ``solvers/time/explicit_steps.py``
   - ``solvers/time/explicit_cfl.py``
   - ``solvers/time/diffrax_*`` owner modules exported by ``solvers/time/__init__.py``
   - ``solvers/time/runners.py``
3. **diagnostics and artifacts**
   - ``diagnostics/moments.py``
   - ``diagnostics/metadata.py``
   - ``diagnostics/transport.py``
   - ``diagnostics/modes.py``
   - ``diagnostics/growth_rates.py``
   - ``diagnostics/growth_windows.py``
   - ``workflows/runtime/diagnostics.py``
   - ``workflows/runtime/diagnostic_arrays.py``
   - ``workflows/runtime/results.py``
   - ``workflows/runtime/orchestration_scan.py`` and ``workflows/runtime/orchestration_artifacts.py``
   - ``workflows/runtime/artifacts.py``
   - ``artifacts/``
   - ``artifacts/plotting.py``
4. **executable workflows**
   - ``workflows/runtime/commands.py``
   - ``workflows/linear.py``
   - ``workflows/nonlinear.py``
   - ``workflows/cases.py``
   - ``workflows/demo.py``
   - ``cli.py``
5. **benchmark and validation tooling**
   - ``gkx.benchmarking.shared``
   - root ``benchmarks/`` drivers
   - purpose-specific ``tools/`` commands
   - ``tests/validation`` physics and benchmark gates

Physics / Numerics / IO Map
---------------------------

.. list-table::
   :header-rows: 1

   * - Responsibility
     - Primary files
     - Typical tests
   * - Basis and spectral grids
     - ``core/velocity.py``, ``core/grid.py``
     - orthonormality, indexing, symmetry
   * - Geometry and imported equilibria
     - ``geometry/analytic.py``, ``geometry/flux_tube.py``, ``geometry/core.py``, ``geometry/imported_miller.py``, ``geometry/imported_vmec.py``, ``geometry/vmec_field_line_sampling.py``, ``geometry/vmec_boozer_derivatives.py``, and ``geometry/vmec_state_controls.py``
     - parser, remap, normalization, geometry-response tests, Miller/VMEC finite-difference geometry and NetCDF writeout gates
   * - Linear operators and fields
     - ``operators/linear/rhs.py``, ``operators/linear/cache_builder.py``, ``operators/linear/collisions.py``, ``operators/linear/collision_tables.py``, ``operators/linear/dissipation.py``, ``solvers/linear/``, ``terms/linear_terms.py``, ``terms/fields.py``, and ``terms/assembly.py``
     - manufactured solutions, observed-order, eigenfunction and branch tests; cache-builder tests cover staged grid, geometry, twist-shift, gyro/moment, drift, and linked-boundary packing
   * - Solver objectives and eigen-AD gates
     - top-level ``gkx`` objective exports, ``objectives/core.py``, ``objectives/eigen.py``, ``objectives/portfolio.py``, ``objectives/portfolio_guard.py``, ``objectives/zonal.py``, ``objectives/stellarator.py``, ``objectives/solver_vmec.py``, ``objectives/gradient_gates.py``, and the retained ``objectives/vmec_*`` modules
     - core linear/quasilinear observables, implicit eigenpair VJP, branch locality, portfolio reduction/covariance, zonal and stellarator objectives, VMEC/Boozer geometry gradients, admission gates, and finite-difference line searches
   * - Nonlinear operators
     - ``solvers/nonlinear/state_integration.py``, ``solvers/nonlinear/diagnostic_integration.py``, ``operators/nonlinear/rhs.py``, ``operators/nonlinear/brackets.py``, ``operators/nonlinear/diagnostic_state.py``, ``operators/nonlinear/diagnostics.py``, ``operators/nonlinear/projection.py``, ``operators/nonlinear/collisions.py``, ``solvers/nonlinear/explicit.py``, ``solvers/nonlinear/diagnostics.py``, ``solvers/nonlinear/imex.py``, ``solvers/nonlinear/imex_diagnostics.py``, ``core/velocity.py``, and ``terms/nonlinear.py``
     - RHS routing, bracket payload, explicit stepping, explicit diagnostic orchestration, IMEX diagnostic orchestration, cached IMEX operator/state policy, diagnostic tuple assembly, fixed-mode and Hermitian projection, collision-split, staged electrostatic/electromagnetic nonlinear contribution helpers, transport-window tests
   * - Parallelization policy and helpers
     - ``parallel.py``, ``sharding.py``, ``parallel/velocity.py``, ``parallel/velocity_plan.py``, ``parallel/velocity_hermite.py``, ``parallel/velocity_streaming.py``, ``parallel/velocity_drive.py``, ``operators/nonlinear/parallel.py``, ``operators/nonlinear/parallel_contracts_domain.py``, ``operators/nonlinear/parallel_contracts_spectral.py``, ``operators/nonlinear/parallel_contracts_strategy.py``, ``operators/nonlinear/domain_decomposition.py``, ``operators/nonlinear/spectral_core.py``, ``operators/nonlinear/spectral_layout.py``, ``operators/nonlinear/spectral_identity_reports.py``, ``operators/nonlinear/spectral_identity_rhs.py``, ``operators/nonlinear/spectral_identity_integrator.py``, ``operators/nonlinear/device_z.py``
     - identity gates, one-device fallback, velocity-space plan/exchange/streaming/field-reduction microkernels, domain/spectral/strategy contracts, spectral state/layout/work-model/bracket/tolerance helpers, logical spectral reports/RHS/integrator gates, device-z routing gates with explicit RHS and transport-trace/report policies, diagnostic-only nonlinear sharding policy
   * - Runtime/executable behavior
     - ``runtime.py``, ``workflows/runtime/startup.py``, ``workflows/runtime/policies.py``, ``workflows/runtime/execution.py``, ``workflows/runtime/diagnostics.py``, ``workflows/runtime/diagnostic_arrays.py``, ``workflows/runtime/initial_conditions.py``, ``workflows/runtime/initial_phi.py``, ``workflows/runtime/chunks.py``, ``workflows/runtime/results.py``, ``workflows/runtime/orchestration_scan.py``, ``workflows/runtime/orchestration_artifacts.py``, ``workflows/runtime/commands.py``, ``workflows/linear.py``, ``workflows/nonlinear.py``, ``workflows/cases.py``, ``workflows/demo.py``, ``cli.py``
     - runtime contract, startup/restart, output-path, full-GK linear/nonlinear workflows, runtime TOML case dependency defaults, saved-output plot command routing, executable artifact path display and progress/summary printing, linear-fit diagnostics, electrostatic-potential initializers, quasilinear finalization, diagnostic-array validation/composition, chunking, result assembly, runtime command workflows, executable smoke tests
   * - Public import registry
     - ``api/__init__.py``
     - compact lazy public export registry, top-level ``gkx`` export membership/order checks, public-object identity tests, API documentation build
   * - Diagnostic extraction and growth-rate fitting
     - ``diagnostics/analysis.py``, ``diagnostics/modes.py``, ``diagnostics/growth_rates.py``, ``diagnostics/growth_windows.py``, ``diagnostics/quasilinear_transport.py``
     - mode selection, eigenfunction extraction, least-squares growth/frequency fitting, automatic fit-window selection, quasilinear transport weights and saturation helpers, late-time growth/frequency tests
   * - Artifacts and plots
     - ``workflows/runtime/artifacts.py``, ``artifacts/``, ``artifacts/spectral_layout.py``, ``artifacts/runtime_plots.py``, ``artifacts/benchmark_plots.py``, ``artifacts/diagnostic_plots.py``, ``artifacts/zonal_plots.py``, ``artifacts/plotting.py``
     - serialization, reload, restart append schema, dealiased-axis contracts, runtime-output plots, benchmark/scan panels, diagnostic/eigenfunction figures, zonal-response figures, plotting contract tests
   * - Benchmark harness
     - ``config.py``, ``gkx.benchmarking.shared``, ``diagnostics/modes.py``, ``diagnostics/validation_gates.py``, ``diagnostics/zonal_validation.py``
     - late-time/windowed gate tests, eigenfunction reference/phase utilities, diagnostics time-series loading, benchmark case presets, physics metric extraction, scan/eigenmode orchestration, reference loading, fallback policy tests

Refactor Mapping
----------------

The current modularization branch is preserving the public runtime surface while
extracting internal responsibilities out of ``runtime.py`` and other large
modules.

Completed extractions:

- compact public API registry in ``gkx.api``. The root
  ``gkx`` package and ``gkx.api`` registry are lazy facades that
  derive the stable documented ``__all__`` order from one authoritative lazy
  target mapping while exporting directly from the
  owning configuration, geometry, solver, validation, parallelization,
  objective, and plotting modules. Pure-contract imports such as
  ``gkx.parallel.decomposition`` must remain dependency-light and must
  not import NumPy/JAX-heavy solver stacks through package initializers.
- zero-shear boundary promotion, analytic s-alpha/slab geometry models, and
  sampled/imported flux-tube geometry data/loading:
  ``geometry/analytic.py`` and ``geometry/flux_tube.py``. Imported NetCDF/eik loading keeps schema
  selection, scalar/profile reads, root-level terminal-theta inference,
  mirror-term reconstruction, drift/Jacobian normalization, and
  ``FluxTubeGeometryData`` packing as separate private stages so geometry-file
  variants can be tested without one large loader body.
- focused imported-geometry owners. ``geometry.imported_miller`` owns the
  complete Miller imported-geometry pipeline. ``geometry.imported_vmec`` is a
  compact VMEC/Boozer-to-EIK orchestrator; backend loading, field-line
  sampling, metric derivatives, and state construction live respectively in
  ``geometry.backend_discovery``, ``geometry.vmec_field_line_sampling``,
  ``geometry.vmec_boozer_derivatives``, and
  ``geometry.vmec_state_controls``. Shared finite-difference and
  period-extension kernels live in ``geometry.kernels``. Imported Miller
  profile assembly keeps central-surface normalization, period extension,
  Bishop coefficients, metric coefficients, magnetic drifts, target-grid
  interpolation, ballooning conversion, and final EIK profile packing as
  explicit stages inside ``geometry.imported_miller``. Ownership tests import
  the VMEC orchestrator once and compare only its genuinely shared backend and
  field-line contracts against those focused owners; former section names are
  not represented by duplicate aliases of the same module.
- mode selection/eigenfunction extraction and late-time growth/frequency
  fitting:
  ``diagnostics/modes.py``, ``diagnostics/growth_rates.py``, and
  ``diagnostics/growth_windows.py``. The
  public ``diagnostics.analysis`` and
  ``diagnostics.growth_rates`` modules remain small facades over focused
  diagnostic owners. Fit-window selection keeps argument validation,
  least-squares scoring, amplitude/slope thresholds, candidate-window search,
  and fallback policy as named stages inside ``diagnostics/growth_windows.py``
  so benchmark auto-windowing can be tested without duplicating fit logic.
- scalar energy, species transport/heating, and resolved spectral diagnostics:
  ``diagnostics/moments.py``, ``diagnostics/transport.py``, and
  ``diagnostics/metadata.py``. Quadrature/mode weights, channel kernels,
  energy, and resolved spectra share the same physical owner and are
  re-exported directly by ``gkx.diagnostics``.
- explicit linear step kernels, diagnostics-rich linear IVP integration,
  explicit CFL/frequency-bound policy, and progress formatting:
  ``solvers/time/explicit_steps.py``, ``solvers/time/explicit_diagnostics.py``,
  and ``solvers/time/explicit_cfl.py``. The public
  ``solvers.time.explicit`` module remains the import facade for existing
  debug tools and tests. Its linear IVP facade now keeps method validation,
  adaptive CFL timing, JIT stepper construction, sample-history collection,
  progress emission, and array packaging as named private stages. The
  diagnostics owner separately stages method/time-policy validation, JIT stepper
  construction, energy/transport sampling, progress rendering, and
  ``SimulationDiagnostics`` construction so saved explicit-time benchmark paths
  exercise named numerical pieces instead of one monolithic loop.
- Diffrax time-integration internals. ``solvers/time/__init__.py`` remains the
  package-level public facade while optional dependency/policy helpers, linear save-path
  integration, streaming growth/frequency fits, and nonlinear integration live
  in ``solvers/time/diffrax_core.py``, ``diffrax_linear.py``,
  ``diffrax_streaming.py``, and ``diffrax_nonlinear.py``. The streaming owner
  stages state/cache preparation, monitored-mode extraction, optimized
  density-mode extraction, weighted log-derivative accumulation, Diffrax RHS
  construction, IMEX zero-term routing, and saved-result finalization so
  differentiable streaming fits can be audited without changing the public
  ``integrate_linear_diffrax_streaming`` contract.
  The saved-field linear owner stages state/cache preparation, packed-state
  sharding, RHS construction, save-field/mode extraction, save-time policy,
  Diffrax solve execution, and final unpacking while keeping the public
  ``integrate_linear_diffrax`` contract stable.
  The fixed-step cached linear integrator stages method validation,
  state/damping preparation, serial/parallel RHS routing, RK/IMEX/SSPX3
  stepping, progress callbacks, and sample-stride scans separately while
  keeping the public ``integrate_linear`` and donated-buffer wrappers stable.
  The nonlinear explicit solver owns fixed-step RK/SSP/K10 scan policy, progress
  callbacks, state projection, and diagnostic explicit stepping in
  ``solvers/nonlinear/explicit.py``.
  The nonlinear owner stages state/cache preparation, packed-state sharding,
  linear and nonlinear RHS construction, IMEX term routing, saved-``phi``
  extraction, solve execution, and final ``FieldState`` packing while keeping
  the public ``integrate_nonlinear_diffrax`` contract stable.
- term-wise RHS assembly internals. ``terms/assembly.py`` owns public RHS
  assembly, cached RHS composition, per-term diagnostic decomposition,
  field-only solves, external-field sources, electrostatic-field policy, and
  collision-skip policy. The production RHS and diagnostic decomposition share one staging
  layer for state/species normalization, scalar parameter expansion, field and
  Hamiltonian construction, drift/drive/dissipation contribution assembly,
  fixed-order term summation, and species-axis restoration. This keeps
  ``assemble_rhs_cached`` and ``assemble_rhs_terms_cached`` numerically aligned
  without duplicating RHS policy. Linear
  electromagnetic field solves in ``terms/fields.py`` are staged as
  coefficient casting, gyrocenter moments, electrostatic potential,
  compressional ``bpar`` coupling, parallel ``apar`` solve, and final
  ``FieldState`` packing, while the public custom-VJP wrapper keeps the same
  differentiability boundary. Linear
  Streaming and diamagnetic contribution kernels keep Hermite-mode drive
  insertion centralized in ``terms/linear_terms.py``. Collisional,
  hypercollisional, hyperdiffusive, and end-damping kernels have one physical
  owner in ``operators/linear/dissipation.py``; assembly imports that owner
  directly rather than retaining a second implementation under ``terms``.
- nonlinear public-driver internals. Cached RHS/state integration lives in
  ``solvers/nonlinear/state_integration.py`` and explicit/IMEX diagnostic entry
  points live in ``solvers/nonlinear/diagnostic_integration.py``.
  Lower-level nonlinear RHS, diagnostic-state, policy, explicit-step, explicit
  and IMEX diagnostic scan preparation/finalization, and IMEX mechanics remain
  owned by ``operators/nonlinear/*`` and
  ``solvers/nonlinear/*`` modules. The explicit nonlinear diagnostic
  implementation keeps the broad public signature as a facade, then packs
  method, timestep, stride, Fourier-layout, selected-mode, collision, and
  output-resolution knobs into a private options object before constructing
  state/policy/closure scan components. This keeps runtime diagnostics
  patchable while making the numerical stages testable without a monolithic
  integration body.
- startup/loading/initial-condition helpers:
  ``workflows/runtime/startup.py``
- runtime mode-index, nonlinear step-count, external-source, parallel-scan,
  and nonlinear diagnostics keyword policies:
  ``workflows/runtime/policies.py``
- runtime linear fit/eigenfunction extraction and quasilinear finalization:
  ``workflows/runtime/diagnostics.py``
- finite-value checks plus runtime diagnostic slicing, truncation, striding,
  and concatenation:
  ``workflows/runtime/diagnostic_arrays.py``. These helpers are imported
  directly from this owner rather than through the runtime fit module.
- adaptive chunk execution used by runtime and comparison artifacts:
  ``workflows/runtime/chunks.py``. Progress/ETA formatting is imported from the
  canonical runtime progress formatter rather than wrapped in the chunk
  owner.
- runtime result containers and nonlinear result assembly:
  ``workflows/runtime/results.py``
- runtime progress formatting, combined-``ky`` scan batching, serial/worker
  scan orchestration, progress formatting, and nonlinear artifact handoff policy:
  ``workflows/runtime/orchestration_scan.py``,
  ``workflows/runtime/chunks.py``, and
  ``workflows/runtime/orchestration_artifacts.py``. Scan dependency-bundle
  builders live with the scan owner and read the public ``runtime.py`` facade
  only as a patchable symbol source. Scan routing keeps combined-``ky``
  eligibility, independent-worker task generation, ordered result packing,
  batch initial-state assembly, diagnostic extraction, and fit selection as
  named stages behind one private scan-options contract. Nonlinear artifact handoff keeps restart
  input resolution, append-history loading, checkpoint chunk sizing, and
  diagnostic-history merging as explicit policies in the artifact owner.
- saved runtime-output plotting command routing:
  ``workflows/runtime/commands.py`` plus ``workflows/runtime/orchestration_artifacts.py``.
  The public ``cli.py`` facade still owns executable parser dispatch and
  renderer/runtime-command patch seams, while the command workflow owns
  ``gkx --plot`` argument validation and runtime command dependency
  construction. Command artifact display, executable headers, and nonlinear
  summaries live with the runtime artifact orchestration policy so saved-output
  behavior and restart/checkpoint handoff share one owner.
- runtime restart-state dispatch: the public ``runtime.py`` facade keeps the
  patchable NetCDF/raw state loaders but shares one shape-keyword payload
  between both paths, so restart dimensions cannot drift between loader calls.
- runtime linear time/fit option forwarding: ``run_runtime_linear``,
  ``run_runtime_scan``, and the combined-``ky`` batch wrapper share one private
  option bundle in the public ``runtime.py`` facade, keeping method, timestep,
  sample-stride, fit-window, mode-method, and fit-signal forwarding aligned.
- runtime TOML case workflows: ``workflows/cases.py`` owns the stable
  ``run_linear_case`` and ``run_nonlinear_case`` signatures and their default
  wiring; the package-level API exports them directly from that owner.
- root benchmark drivers and result pointers:
  ``benchmarks/``. This directory is the canonical lightweight benchmark
  entry point at repository root. It stores drivers, TOML inputs, and
  ``benchmarks/results/manifest.toml`` only; raw solver products stay in
  scratch directories and promoted benchmark results are displayed from
  :doc:`benchmarks`.
- full-GK executable linear runtime workflow:
  ``workflows/linear.py``. Context preparation, time integration, Krylov
  fallback, linear fitting, and quasilinear finalization are separate private
  stages so runtime dispatch can stay compact while preserving the public
  ``gkx.runtime`` facade and monkeypatch seams.
- full-GK executable nonlinear runtime workflow:
  ``workflows/nonlinear.py``
- shared plot style plus runtime-output, benchmark/scan, diagnostic, and
  zonal-response figure families:
  ``artifacts/runtime_plots.py``, ``artifacts/benchmark_plots.py``,
  ``artifacts/diagnostic_plots.py``, and ``artifacts/zonal_plots.py``.
  The shared publication style and public imports live in
  ``artifacts/plotting.py``, which remains the stable facade for examples and
  user scripts.
- validation acceptance contracts and JSON-ready gate helpers:
  ``diagnostics/validation_gates.py`` owns scalar tolerance evaluation, JSON
  serialization, and report builders. Metric extraction and its result
  contracts live in ``diagnostics/analysis.py``; eigenfunction comparison and
  reference-bundle contracts live in ``diagnostics/modes.py``. One private
  upper-limit scalar-gate policy is shared for convergence, mismatch, deficit,
  and branch-jump thresholds so tolerance semantics remain auditable.
- autodiff validation helpers:
  ``objectives/autodiff_validation.py`` owns finite-difference Jacobians,
  Gauss-Newton covariance diagnostics, dense operator materialization, and
  isolated eigenbranch sensitivity reports. Eigenbranch AD gates share
  selector/gap classification, complex-observable realification, unsupported-AD
  fallback packing, implicit left/right eigenpair tangent solves, split
  observable chain-rule assembly, and
  finite-difference comparison stages.
- benchmark-harness physics metric extraction and scan/mode orchestration:
  ``diagnostics/analysis.py`` and ``workflows/runtime``. Reviewed
  reference tables and comparison-only policies live in
  ``benchmarking/shared.py``. Eigenfunction normalization,
  phase alignment, comparison metrics, and reference-bundle IO live in
  ``diagnostics/modes.py``; diagnostic time-series loading lives in
  ``artifacts/nonlinear_diagnostics.py``; late/leading windows and analytic
  signals live in ``diagnostics/growth_windows.py``; real-FFT grid inference
  lives in ``artifacts/spectral_layout.py``. Late-time linear metrics,
  nonlinear transport windows, heat-flux convergence, observed-order checks,
  and branch-continuity metrics live in ``diagnostics/analysis.py``.
  Zonal-flow residual/GAM metric extraction lives in
  ``diagnostics/zonal_validation.py``. Zonal-response metrics are staged as trace
  coercion, initial-level normalization, tail residual extraction,
  extrema/envelope detection, damping/frequency fits, and final
  ``ZonalFlowResponseMetrics`` packing, keeping Rosenbluth-Hinton/GAM
  conventions explicit.
- zonal-response reference/trace normalization helpers:
  ``diagnostics/zonal_validation.py``
- dominant-eigenvalue custom VJP and branch-locality diagnostics:
  ``objectives/eigen.py``
- core solver-objective constants plus value-level linear/quasilinear
  observables:
  ``objectives/core.py``
- solver-objective sampling axes, physical-``ky`` grid mapping, and aggregate
  weights:
  ``objectives/sampling.py``
- solver-ready geometry objective gates, reduced nonlinear-window metrics,
  solver-ready gradient gates, mode-21 VMEC/Boozer gradient gates,
  backend-free portfolio row/weight contracts, portfolio AD/FD sensitivity
  gates, artifact promotion guards, geometry-owned VMEC/Boozer state coefficient helpers,
  VMEC/Boozer objective-table plumbing, and VMEC/Boozer finite-difference/
  line-search gates:
  ``objectives/geometry.py``,
  ``objectives/gradient_gates.py``,
  ``objectives/vmec_boozer_gradients.py``,
  ``objectives/portfolio.py``,
  ``objectives/portfolio_guard.py``,
  ``objectives/vmec_boozer.py``,
  ``objectives/vmec_boozer_fd.py``,
  ``objectives/vmec_boozer_line_search.py``. Scalar and aggregate
  VMEC/Boozer finite-difference reports share one settings validator, one
  VMEC-state coefficient context, one perturbation helper, and one three-point
  response/curvature diagnostic so scalar and aggregate sensitivity gates use
  the same finite-value, response-resolution, and curvature policy. Scalar and
  aggregate reports now keep aggregate surface/field-line/``ky`` planning,
  three-point VMEC-state evaluation, weighted-sample metadata, and payload
  assembly in separate helpers so optimizer-facing sensitivity reports remain
  auditable without changing their public JSON schema. Mode-21 VMEC/Boozer
  gradient reports now share context construction, observable-vector assembly,
  implicit sensitivity-gate execution, and common payload metadata, while the
  public report functions remain the physics-facing entry points for linear,
  quasilinear, and reduced nonlinear-window differentiability claims. Scalar and
  aggregate VMEC/Boozer line-search reports share one private curvature-gated
  one-parameter search loop plus focused scalar/aggregate probe builders,
  common payload assembly, and explicit held-out training/probe helpers,
  keeping finite-difference, training, and held-out aggregate gates on the same
  accept/reject policy. Reduced QA low-turbulence comparison artifacts keep
  optimized-design generation, per-design diagnostic payloads, gate booleans,
  comparison metrics, and differentiability-plumbing metadata in separate
  helpers so publication JSON structure and optimization diagnostics can evolve
  independently. Solver-ready
  gradient gates share one normalized heat/particle transport-weight helper
  for eigenmode observables so linear-growth, quasilinear, and particle-flux
  AD checks use the same quadrature and normalization path. Solver-ready
  branch-continuity and linear-RHS gradient reports now stage parameter
  validation, objective construction, eigensystem branch checks, implicit
  AD/FD gate rows, value-evaluator checks, and report packing separately.
  The top-level
  ``gkx.objectives`` API re-exports the portfolio helpers directly from
  these owner modules.
- production nonlinear turbulent-flux optimization guardrails now live in
  ``diagnostics/nonlinear_transport_optimization.py``. The diagnostics owner
  stages optimization-scope normalization, reduced-artifact scope checks,
  replicated long-window transport extraction, matched baseline-to-optimized
  audit gates, safety gates, promotion gates, evidence-gap accounting, and
  summary assembly so release safety and production-claim promotion cannot be
  conflated. Persisted pass flags and named gate rows require explicit Boolean
  values, and replicate counts require finite nonnegative integers; malformed
  evidence therefore produces blockers rather than promotion or exceptions.
  Explicit finite comparison metrics take precedence over fallback
  statistics, including a physical zero; zero uncertainty separation can
  therefore never be replaced by a stale positive fallback and accidentally
  promote an audit. Replicate-spread diagnostics now live in
  ``diagnostics/nonlinear_replicates.py`` and stage ensemble row normalization,
  high/low variant selection, state classification, replicate-row packing, and
  summary assembly. The same owner builds seed/timestep artifact-readiness
  manifests, while ``diagnostics/transport_windows.py`` owns individual-window
  statistics and ensemble uncertainty gates. These decisions remain testable
  without rerunning nonlinear simulations.
- quasilinear nonlinear-window convergence metadata is consolidated in
  ``diagnostics/transport_windows.py`` for statistics, CSV/summary IO,
  promotion readiness, and ensemble uncertainty; replicate readiness belongs
  to ``diagnostics/nonlinear_replicates.py``. The public API re-exports the
  documented transport-window helpers directly from these diagnostics owners.
  Persisted report, gate, and row pass flags must be explicit booleans; strings
  and numeric lookalikes fail closed. The statistics owner stages validated
  late-window selection, finite-sample counts, drift/terminal-window metrics,
  block/bootstrap uncertainty, and gate-report assembly in one file so
  nonlinear transport admission rules remain auditable. The ensemble owner
  stages replicate-row normalization, uncertainty statistics, gate packing,
  artifact grouping, missing-replicate hints, and readiness-manifest packing
  so seed/timestep promotion evidence can be tested without rerunning
  simulations.
- quasilinear model-selection claim boundaries live in
  ``diagnostics/quasilinear_model_selection.py``. This owner separates
  candidate-skill gate rows, absolute-flux overclaim guardrails, optional
  optimized-equilibrium audit gates, and final ledger assembly, while
  artifact loading and required-candidate metric normalization.
- nonlinear parallelization policy metadata, local domain prototypes, and
  spectral-core work models/RHS primitives plus device-z shard-map routes:
  ``operators/nonlinear/parallel.py``,
  ``operators/nonlinear/parallel_contracts_domain.py``,
  ``operators/nonlinear/parallel_contracts_spectral.py``,
  ``operators/nonlinear/parallel_contracts_strategy.py``,
  ``operators/nonlinear/domain_decomposition.py``,
  ``operators/nonlinear/spectral_core.py``,
  ``operators/nonlinear/spectral_layout.py``,
  ``operators/nonlinear/spectral_identity_reports.py``,
  ``operators/nonlinear/spectral_identity_rhs.py``,
  ``operators/nonlinear/spectral_identity_integrator.py``,
  ``operators/nonlinear/device_z.py``
  The local-domain transport gate stages trace collection, trace-error scoring,
  fail-closed blockers, and report packing in
  ``operators/nonlinear/domain_decomposition.py``. The spectral fixed-window
  and pencil-transport gates use the same structure in
  ``operators/nonlinear/spectral_identity_integrator.py`` so future
  profiler-backed nonlinear decomposition work can change routing without
  weakening the numerical-identity policy. The device-z route keeps sharding
  setup, fail-closed topology checks, shard-map RHS execution, transport-window
  sampling, compute-only final-state identity, and final report packing as
  separate stages inside
  ``operators/nonlinear/device_z.py`` while report-schema construction remains
  in ``operators/nonlinear/device_z_reports.py``.
- velocity-space parallelization is split into decomposition metadata
  (``parallel/velocity_plan.py``), Hermite exchange and velocity-field
  reductions (``parallel/velocity_hermite.py``), streaming/magnetic-drift
  microkernels (``parallel/velocity_streaming.py``), and electrostatic/
  diamagnetic field-drive microkernels (``parallel/velocity_drive.py``). The
  public ``parallel/velocity.py`` module remains the stable facade used by
  tools and performance gates.
- independent-work parallelization is split into numerical-identity reports
  (``parallel/identity.py``), JAX-array batch mapping
  (``parallel/batch.py``), and ordered Python ensemble execution
  (``parallel/independent.py``). The public ``parallel/core.py`` and
  ``parallel/__init__.py`` modules remain stable facades for user imports.
  The package facade uses a static lazy-export registry rather than reading
  sibling source files at import time. This keeps imports dependency-light and
  compatible with installed wheels and non-filesystem Python loaders; a unit
  test requires the registry to match each owning module's public exports.
  ``parallel/independent.py`` keeps indexed payload collection, reconstruction
  contracts, identity report construction, exception provenance, ordered
  executor routing, and metadata packing as separate helpers so UQ and
  optimization ensembles stay auditable without changing solver layout.
- nonlinear RHS composition and state-to-diagnostic tuple assembly:
  ``operators/nonlinear/rhs.py`` and
  ``operators/nonlinear/diagnostic_state.py``. The diagnostic-state owner
  separates field defaulting, growth/frequency mode extraction, scalar
  diagnostics, resolved field/transport group evaluation, and resolved
  spectra/channel schema packing while preserving the
  explicit/IMEX scan tuple schema. The old root nonlinear helper shims were
  removed; normal users should use the top-level ``gkx`` public API and
  developer helpers should import from ``gkx.solvers.nonlinear`` and
  ``gkx.operators.nonlinear``.
- full-GK nonlinear executable orchestration lives in
  ``workflows/nonlinear.py`` behind the public ``gkx.runtime`` facade.
  The owner separates runtime context construction, fixed-mode/source policy,
  diagnostic keyword forwarding, adaptive/fixed diagnostic execution,
  final-state integration, and result assembly so runtime branch tests can
  exercise each policy without duplicating executable wiring.
- fixed-step linear integration keeps public dispatch in
  ``solvers/linear/integrators.py`` while diagnostic sampling lives in
  ``solvers/linear/integrator_diagnostics.py``. The diagnostic owner now keeps
  sampling validation, cache/state setup, damping assembly, explicit/IMEX/RK
  step policy, density/Hermite-Laguerre observables, progress callbacks, and
  every-step versus strided scans in named stages. The facade preserves the
  public ``integrate_linear_diagnostics`` import and test monkeypatch seams.
- velocity-parallel linear RHS routing is split into common eligibility/device
  policy (``solvers/linear/parallel_common.py``), Hermite streaming routes
  (``solvers/linear/parallel_streaming.py``), and electrostatic slice/fused
  shard-map routes (``solvers/linear/parallel_electrostatic.py``). The public
  ``solvers/linear/parallel.py`` module remains the stable dispatcher. The
  electrostatic fused route is staged as route validation, sharding specs,
  closure constants, Hermite exchange, phi solve, streaming/mirror/drift/drive
  term builders, and JIT cache lookup; the serial single-device path has a
  separate helper so identity tests can exercise both routes.
- explicit RK/SSP/K10 one-step policy, cached explicit scan policy, explicit
  diagnostic step and scan-selection policy, explicit diagnostic integration
  orchestration, IMEX diagnostic integration orchestration, cached IMEX scan
  policy, IMEX diagnostic step and scan-execution policy, and IMEX
  fixed-point/GMRES solve policy: ``solvers/nonlinear/explicit.py``,
  ``solvers/nonlinear/diagnostics.py``, ``solvers/nonlinear/imex_diagnostics.py``,
  and ``solvers/nonlinear/imex.py``. Developer helpers should import from
  ``gkx.solvers.nonlinear``.
- linear cache, linked-boundary maps, Hermite-Laguerre moments, parameter
  pytrees, cache-backed RHS assembly, implicit linear GMRES/preconditioner
  policy, fixed-step/diagnostic integration policy, eigenmode policy/operator/
  branch-selection/preconditioner/Krylov algorithms, and velocity-parallel RHS
  dispatch live under ``operators/linear/`` and ``solvers/linear/``. The cache
  path is split into ``cache_model.py`` for the JAX pytree, ``cache_arrays.py``
  for moment/damping/gyroaverage array factories, ``cache_builder.py`` for
  geometry-dependent construction, and ``collisions.py`` for collision-matrix
  assembly, table interpolation, and runtime application. This keeps the
  collision equations independent of cache allocation without introducing a
  compatibility facade. The builder itself now has explicit private
  stages for twist-shift policy, perpendicular wavenumber/drift arrays,
  Laguerre gyroaverage construction, and linked-boundary metadata so extension
  work can test one numerical policy at a time; ``operators.linear`` remains
  the package-level public cache facade. The public Krylov import path remains
  ``solvers/linear/krylov.py``;
  that facade now keeps option normalization, user-facing progress messages,
  shift-invert seed selection, shift-selection flags, and fallback policy as
  explicit private stages while delegating compiled kernels to the focused
  owner modules;
  ``solvers/linear/implicit.py`` keeps implicit state normalization, damping/
  drift diagonal assembly, linked Hermite-line solves, coarse kx projection,
  preconditioner selection, and matrix-free matvec construction as separate
  private stages;
  matrix-free operator application and compiled iterations share the focused
  ``krylov_algorithms.py`` owner. The old root
  ``linear_*`` helper shims were
  removed; normal users should use the top-level ``gkx`` public API for
  the public linear surface or import focused developer helpers from the domain
  packages.
- nonlinear turbulence-gradient paired-seed statistics, control-variate
  construction, uncertainty propagation, and independent control-mean gates
  live in ``diagnostics/nonlinear_gradient_statistics.py`` as pure reusable
  functions. Historical candidate selection and launch policy remains in a
  temporary campaign owner pending deletion; it no longer owns scientific
  uncertainty arithmetic.
- nonlinear turbulence-gradient evidence scope markers, acceptance config
  dataclasses, JSON-safe parsing, finite-difference conditioning gates, and
  artifact classification live in ``diagnostics/metadata.py``. Replicated
  window summaries live in ``diagnostics/nonlinear_replicates.py`` and central
  finite-difference transport response/uncertainty lives in
  ``diagnostics/transport.py``. The compact
  ``diagnostics/nonlinear_gradient_evidence.py`` facade owns bracket and
  candidate report orchestration plus production evidence-gap reports. It has
  one final export contract; ``diagnostics/metadata.py`` and
  ``diagnostics/transport.py`` likewise each have one complete owner contract
  covering their shared data, transport, and evidence interfaces. Intermediate
  section exports and alias-based ownership tests were removed so each helper
  is tested at its real owner.
  Candidate ranking and evidence-gap reporting remain fail-closed so production
  nonlinear-gradient promotion cannot be inferred from startup, pilot, reduced,
  or single-window artifacts.
- runtime artifact read/write, generic I/O and finite-value validation helpers,
  linear/quasilinear artifact writers, generic nonlinear table writers, dealiased-axis
  layout, unified NetCDF output-bundle and diagnostic-history schema writing
  with Phi2, species-history, split electromagnetic, zonal-field, resolved
  spectra, geometry, restart, and final-field helpers, nonlinear diagnostic
  reload helpers, and restart-append schema coverage:
  ``workflows/runtime/artifacts.py``, ``artifacts/io.py``,
  ``artifacts/spectral_layout.py``, and ``artifacts/nonlinear_netcdf.py``.
  The old root
  ``runtime_artifact_*`` helper modules were removed; import implementation
  helpers from ``gkx.artifacts`` instead.

Next planned extractions:

- benchmark-family runners and fit-policy helpers
- remaining linear result-assembly helpers
- runtime output/artifact handoff helpers

Traceability For Refactors
--------------------------

Refactor work is tracked in ``tools/validation_coverage_manifest.toml``. The
manifest is checked by ``tools/release/check_validation_coverage_manifest.py`` and
requires every high-priority module to name its source path, owning lane,
reference anchors, physics and numerics contracts, fast tests, artifacts, and
next coverage tests. Update it whenever a source extraction changes module
ownership or validation responsibility.

The manifest also owns the package inventory. Direct rows cover the public or
high-risk refactor modules. Smaller implementation modules are listed in an
owner row's ``owned_modules`` field when their behavior is validated by the
same fast tests and artifacts. Package plumbing such as ``__init__.py`` and
version metadata is the only normal exclusion. Adding a new
``src/gkx/*.py`` file without one of those declarations should fail the
manifest checker.

Use this rule of thumb when changing ownership:

- add a direct row when a module has public imports, independent physics or
  numerics contracts, separate artifact traceability, or high refactor risk;
- add it to ``owned_modules`` when it is a narrow helper split whose contract is
  still fully exercised by the owning row's tests;
- update the owning row's ``next_tests`` when the helper split creates coverage
  debt that is not closed in the same change.

The refactor manifest tests also cross-check public and size-sensitive
surfaces. Every ``.. automodule:: gkx.*`` entry in :doc:`api` must
resolve to a source file or package ``__init__`` that is accounted for by the
manifest, and public package ``__init__`` entries must be explicit exceptions
rather than generic plumbing. The same test requires any non-``__init__``
source module with at least 2,000 non-comment source lines to have a direct
manifest row, so large modules cannot be hidden under another owner's
``owned_modules`` list.

``tools/package_architecture_manifest.toml`` is the authoritative package-topology
policy and enforces the module layout and aggregate
line budgets. Differentiability, physics, parity, and performance requirements
are acceptance gates in the root plan rather than a second migration ledger.

Refactor-only metadata containers have been removed from the runtime package.
The one structural protocol used by the equations is the compact
``gkx.operators.collision`` interface; validation and migration policy
remain in release manifests and tests rather than becoming runtime objects.

Runtime command dispatch now keeps parser construction in ``gkx.cli``,
with parser registration split by command family, and moves runtime linear,
runtime scan, and runtime nonlinear command execution into
``gkx.workflows.runtime.commands``. The generic ``run`` executable path
attaches the already-loaded runtime config/data to the parser namespace before
dispatch, so command execution does not parse the same TOML twice.
Linear, scan, and nonlinear command flags are resolved once into typed
command-option records before solver calls are made, which keeps precedence
between CLI flags, TOML sections, and runtime defaults inspectable without
spreading executable policy across command bodies. Saved-artifact display order
and optional artifact writing for linear, scan, quasilinear, and nonlinear
commands also live in focused command-output helpers so user-facing executable
messages can be tested without launching solver runs.
Programmatic TOML case helpers and executable command dispatch now share
``gkx.workflows.runtime.commands``. Path override, progress,
quasilinear override, dependency injection, and preload-reuse policies therefore
have one canonical workflow owner rather than two forwarding modules.
Runtime initial-condition construction is staged inside
``gkx.workflows.runtime.initial_conditions``. Validation of initializer
options, restart-state scaling, kinetic-species targeting, single-mode profile
assembly, random/Gaussian multimode seeding, electrostatic-potential seeding,
Hermitian full-``ky`` completion, and restart merge policy now have named helper
stages. The startup facade still exposes the same ``_build_initial_condition``
entry point for runtime workflows and tests, but the implementation no longer
mixes restart I/O, moment normalization policy, and phi-inversion cache setup in
one long branch.

The benchmark stack has two explicit roles:

- ``gkx.benchmarking.shared`` owns compact reference containers, CSV
  loaders, normalization constants, and comparison-only defaults.
- Canonical TOML inputs plus ``run_runtime_linear`` and ``run_runtime_scan``
  own promoted solver execution. ETG and Cyclone use only this path; artifact
  figures consume reviewed tables rather than launching a second hidden solve.

Reviewed reference data and comparison policies live in
``gkx.benchmarking.shared``. KBM time histories and fixed-beta ``k_y`` scans use
generic runtime orchestration. The former ``kbm_beta_scan.py`` was removed
after audit showed that it incorrectly interpreted a ``k_y`` reference table
as beta values. TEM and kinetic-electron execution also use canonical runtime
inputs and scan paths. New examples must not add named-case execution APIs.
External-code names and raw reference handling stay in root ``benchmarks/`` or
``tools/comparison``; reusable source modules use physics and numerical names.

Ordered continuation scans use ``run_runtime_parameter_scan``. The caller
supplies a named scalar axis and a pure ``RuntimeConfig`` update callback; the
runtime records one ``RuntimeLinearResult`` per point and can pass the previous
state into the next solve. Case-specific branch targets and acceptance rules
remain callback policy rather than hidden runtime behavior. Continuation scans
are intentionally sequential because adjacent points depend on one another;
independent parameter ensembles should instead use the parallel batch APIs.
For spectra with competing branches, ``candidate_options`` declares the
per-point solver targets and ``select_candidate`` chooses the retained result;
only that result is continued.

Time-domain linear results retain the selected-``k_y`` field history.
``refit_runtime_linear_trajectory`` applies another fit window or mode
extractor to that stored trajectory without repeating the simulation. This is
the preferred path for branch audits: integration cost is paid once, while
``z_index``, ``max``, ``project``, and ``svd`` diagnostics remain independently
reviewable.


Quasilinear calibration now lives in
``gkx.diagnostics.quasilinear_calibration``. It owns calibration-point
schemas, spectrum integration, train/holdout scale fitting, nonlinear-window
CSV/NetCDF ingestion, and report writing behind one diagnostics owner. Each
nonlinear trace is loaded once and reused for both the selected window and its
convergence report; persisted ensemble pass flags must be explicit booleans.
Late-window transport gates live in ``gkx.diagnostics.transport_windows``.
The public validation API re-exports user-facing helpers while
campaign launch and artifact-building policy stays in ``tools``. Model-selection status construction keeps scoped candidate-skill gates,
input normalization, optimized-equilibrium audit summaries, and absolute-flux
claim guardrails in ``diagnostics/quasilinear_model_selection.py``.

VMEC-JAX candidate and transport admission gates now have explicit owners.
``gkx.objectives.vmec_candidate_admission`` owns solved-equilibrium,
authoritative-WOUT, and WOUT-reproducibility candidate gates. It keeps aspect,
iota, iota-profile, quasisymmetry, and pass/fail helpers together so optimizer
state and WOUT gates share one JSON schema and threshold semantics.
``gkx.objectives.vmec_transport_admission`` owns transport-admission
policy dataclasses, reduced transport metric selection, multi-surface/
field-line/``k_y`` sample coverage, and promoted transport-candidate selection.
The ``gkx.objectives.vmec_transport`` module owns objective
configuration, optional-backend path policy, differentiable sample tables,
reductions, and the optimizer callback. Eigenbranch-locality gates remain in
``vmec_transport_branch`` because they evaluate a distinct three-state
continuation contract.
``gkx.diagnostics.stellarator_transport_reports`` owns report-style
nonlinear transport diagnostics: landscape admission, reduced prelaunch gates,
next-campaign admission, and matched nonlinear audit redesign. Persisted gate
flags must be explicit booleans and replicate counts must be finite,
nonnegative integers; malformed values fail closed into report blockers rather
than becoming truthy or raising during report construction. The public
``gkx.api`` re-exports user-facing admission helpers directly
from these owners, while installable validation-campaign subpackages have
been removed.

The first differentiable-geometry split keeps
``gkx.geometry.differentiable`` as the public facade while
moving optional backend lookup and strict AD/finite-difference gate utilities
into ``gkx.geometry.backend_discovery`` and
``gkx.geometry.autodiff_checks``. The solver-ready in-memory flux-tube
mapping and geometry-observable contract lives in
``gkx.geometry.flux_tube_contract``. Pure parity metrics,
interpolation, radial derivative, Boozer half-mesh, Fourier field-line, and
periodic sampling helpers live in ``gkx.geometry.numerics``. This
separates import-side effects, validation-report plumbing, public contract
validation, and small numerical kernels from the VMEC/Boozer field-line bridge.
The AD/FD validation owner stages parameter validation, observable flattening,
Jacobian construction, tangent checks, conditioning gates, failure reasons, and
strict JSON report assembly so differentiability tests can target each
research-grade gate directly.
Imported VMEC/Boozer geometry generation enters through
``gkx.geometry.imported_vmec``. Its focused owners separate optional
backend discovery, radial spline and Boozer-mode sampling, Hegna-Nakajima and
metric derivatives, and VMEC field-line state construction. The orchestrator
retains only flux-tube cutting, equal-arc remapping, atomic EIK output, and the
high-level request path. This separation keeps the formulas discoverable and
lets backend, sampling, derivative, and orchestration tests fail independently.
Zero-shear boundary policy and analytic s-alpha/slab geometry models live in
``gkx.geometry.analytic``. Sampled solver-ready geometry data, analytic
sampling, imported-NetCDF loading, and periodic mirror-term reconstruction live
in ``gkx.geometry.flux_tube``. Twist-shift and grid-default geometry
policy live in ``gkx.geometry.core``. The
``gkx.geometry`` package remains a thin public facade that
re-exports the same classes and functions for existing user code.
Geometry sensitivity, inverse-design, and local UQ reports live in
``gkx.geometry.sensitivity`` so backend bridge modules can depend on
the report contract without importing the public facade. Bounded VMEC
boundary and Boozer-spectrum bridge checks, Boozer ``|B|`` field-line
evaluation, and Boozer-to-flux-tube sensitivity diagnostics live in
``gkx.geometry.booz_xform_bridge``. Pure helper imports retain object
identity; backend-discovery-dependent bridge functions use thin facade wrappers
so existing monkeypatch-based optional-backend tests still target
``gkx.geometry.differentiable``. VMEC-state-to-Boozer, VMEC metric
tensor, and VMEC field-line tensor AD/FD sensitivity reports live in
``gkx.geometry.vmec_state_sensitivity``; the public facade uses the
same hook-preserving wrapper pattern, while the implementation owns shared
VMEC example loading, coefficient-index validation, perturbation policy, and
the common tensor-observable AD/finite-difference payload builder used by
metric and field-line gates. Metric and field-line tensor reports also share
one VMEC geometry context/index helper before diverging into their
observable-specific sampling paths.
Boozer constant preparation and equal-arc cache prewarm helpers live in
``gkx.geometry.vmec_boozer_constants``. Core Boozer equal-arc profile
construction, radial Boozer-profile interpolation, and equal-arc remapping
live in ``gkx.geometry.vmec_boozer_core``. That owner now stages the
bridge as radial Boozer-profile interpolation, equal-arc field-line
construction, zero-beta metric/drift profile assembly, and final solver-ready
mapping assembly; optional backend execution and Boozer radial-grid validation
remain private helper seams so the public state-to-profile bridge stays
focused on physics profile assembly. Metric/drift profile assembly is further
split into differential-geometry evaluation, raw metric coefficients, raw
curvature-drift coefficients, open equal-arc remapping, and final
``_MetricDriftProfiles`` packing so the tensor algebra and solver-normalized
coefficients can be reviewed independently. The core bridge delegates Boozer
field-line spectral sums, cylindrical derivatives, and coordinate-gradient
algebra to
``gkx.geometry.vmec_boozer_derivatives`` so the differentiable geometry
path has a small, unit-testable tensor-algebra owner rather than one long
VMEC/Boozer orchestration function. The core
profile assembly shares one dtype-aware numerical floor across ``|B|``,
``gradpar``, Jacobian, metric, curvature, and ``q`` denominators so
regularization policy is visible and consistent. Boozer metric-gradient terms
use a separate float32-safe toroidal-flux denominator floor before
``grad(theta)``, ``grad(phi)``, and ``grad(alpha)`` divisions. Direct
``vmex`` tensor sampling and conversion into the solver-ready flux-tube
mapping contract lives in ``gkx.geometry.vmec_tensor_mapping``. That
bridge is staged as surface/reference-scale validation, shared VMEC field-line
coordinate construction, raw tensor loading, periodic line sampling,
perpendicular metric assembly, local grad-``B`` drift closure, and final mapping
packaging. VMEC flux-tube sensitivity and array-parity report orchestration
lives in
``gkx.geometry.vmec_flux_tube_reports``; it reuses the shared
VMEC-state example loading, coefficient-index validation, and perturbation
policy from ``gkx.geometry.vmec_state_sensitivity`` so the flux-tube,
Boozer, metric-tensor, and field-line AD/FD gates stay on one setup contract.
Direct-array parity, imported-EIK loading, optional Boozer equal-arc parity,
production parity metrics, and final JSON packing now have separate private
stages in that owner. The public report functions remain schema-preserving
facades over those stages, which keeps the VMEC/Boozer differentiability and
parity gates easier to review without adding another module.
VMEC boundary-gradient
probe classification, collection row assembly, and projected-transport
line-search admission summaries live in
``gkx.geometry.vmec_boundary_chain``. Boundary-chain scalar error
construction and pass/fail policy helpers are kept explicit in that owner so
VMEC/Boozer finite-difference, JVP, and VJP convention gates can be tested
without launching expensive VMEC solves.

Release-scope synchronization for refactors is tracked separately in
:doc:`release_scope`. In particular, the current restartable NetCDF append
contract normalizes diagnostics loaded from ``*.out.nc`` to the persisted
schema before concatenation; transient in-memory traces that are not stored in
the NetCDF artifact are not treated as release data on continuation.

Testing Taxonomy
----------------

The source tree should be validated through five distinct test classes:

1. **unit tests**
   - cheap, deterministic, local behavior
2. **numerical verification tests**
   - observed-order, manufactured solutions, invariants
3. **benchmark/validation tests**
   - growth rates, frequencies, eigenfunctions, transport windows
4. **autodiff tests**
   - finite-difference and complex-step gradient checks, tangent/adjoint consistency
5. **regression tests**
   - runtime contracts, saved artifacts, reference-lane preservation

Every future source extraction should update this page if module ownership
changes materially.

Repository Artifact Hygiene
---------------------------

The repository should keep reproducible source inputs, tests, small gate
reports, and lightweight documentation previews in Git. Heavy runtime outputs,
profiler traces, raw NetCDF comparisons, and high-resolution publication
exports should live in release artifacts with checksums and replay commands.

Use the non-destructive audit helper before release cleanups:

.. code-block:: bash

   python tools/release/check_repository_size_manifest.py audit --top 30
   python tools/release/check_repository_size_manifest.py
   python tools/release/check_repository_size_manifest.py release-artifacts

The report separates tracked file size from ignored local artifact roots such
as ``tools_out/``, ``docs/_build/``, ``dist/``, virtual environments, and caches.
The checked manifest is ``tools/repository_size_manifest.toml``. It defines the
tracked-size budget, the maximum size of any unlisted tracked file, and the
temporary whitelist for any intentionally retained large files.

The release migration manifest is ``tools/release_artifact_manifest.toml``. It
records checksums, replay commands, and planned destinations for high-resolution
panels and other large assets. The checker validates provenance only; it does
not upload or delete artifacts. ``move_to_release`` entries may be absent from
Git after migration, but only when the manifest records the immutable
``release_tag`` and ``release_url`` together with the original size and SHA-256
checksum.

Documentation figures should use lightweight checked-in previews, with
high-resolution publication exports regenerated from the replay commands or
hosted as release assets. The reproducible preview-compression command is:

.. code-block:: bash

   python tools/release/check_repository_size_manifest.py compress-previews --mode release --max-width 2200 --colors 192
   python tools/release/check_repository_size_manifest.py compress-previews --mode docs --min-bytes 300000 --max-width 1800 --colors 192

The first command only touches release-manifest previews, so update
``tools/release_artifact_manifest.toml`` with the new sizes and checksums after
running it. The second command skips release-manifest paths by default and is
intended for ordinary checked-in documentation previews. Rerun both manifest
checkers after either cleanup.

History rewrites are not part of routine development; they require a coordinated
maintenance window because every collaborator must reclone or reset local
branches after a force push.
