API Reference
=============

Public API Registry
-------------------

The top-level ``gkx`` package advertises only the current product contracts:
``load``, ``solve``, ``scan``, ``plot``, ``prepare``, ``Case``, ``LinearResult``,
``NonlinearResult``, and ``ScanResult``. Five integration names remain
advertised for the maintained VMEX turbulence coupling:
``flux_tube_geometry_from_mapping``, ``solver_objective_vector_from_geometry``,
``solver_linear_operator_matrix_from_geometry``,
``solver_scalar_objective_from_vector``, and
``VMEXTransportObjectiveConfig``. Together with ``__version__``, this makes a
15-name root surface.

Implementation ownership remains in the domain modules below. GKX 1.x root
names outside this advertised set stay available by direct named import
through GKX 2.x, but no longer appear in ``gkx.__all__``, ``dir(gkx)``, or
wildcard imports. They may be removed no earlier than GKX 3.0; migrate to the
owning subpackage shown in this reference before then. The complete legacy
lazy registry is retained during that window, so this reduction does not
eagerly import numerical dependencies or wrap the underlying objects.

GKX 3 Product Contracts
-----------------------

The first GKX 3 product-surface names are immutable, zero-copy views of the
established runtime contracts:

``Case``
   The frozen runtime configuration, identical to ``RuntimeConfig``.
``LinearResult``
   A linear solve result, identical to ``RuntimeLinearResult``.
``NonlinearResult``
   A nonlinear solve result, identical to ``RuntimeNonlinearResult``.
``ScanResult``
   A linear scan result, identical to ``RuntimeLinearScanResult``.

The identity relationship is intentional during migration: existing workflows
already return the new contract types, result arrays are never wrapped or
copied, and historical ``Runtime*`` imports remain valid. Later Phase 1 changes
can therefore add high-level workflows without moving numerical kernels into
this public facade.

The first high-level workflow is deliberately small::

   case = gkx.load("cyclone.toml")
   result = gkx.solve(case)
   figure, axes = gkx.plot(result)
   spectrum = gkx.scan(case, [0.1, 0.2, 0.3])
   simulation = gkx.prepare(case)
   time, diagnostics, state, fields = simulation.run()

``load`` resolves case-relative paths through the established TOML owner.
``solve`` dispatches to the established nonlinear runner when nonlinear physics
is enabled and otherwise to the linear runner. ``scan`` is the established
runtime scan function itself, so these names add no numerical implementation or
result conversion. ``plot`` dispatches those result containers to the existing
linear, scan, and nonlinear figure builders; it returns ``(Figure, axes)`` and
does not save or display implicitly.

``prepare`` reuses the nonlinear runtime's established explicit diagnostic
scan and fixes its grid, geometry layout, numerical method, and output schema.
Repeated ``simulation.run(initial_state)`` calls reuse the compiled scan for
matching shapes and dtypes; ``simulation.run_arrays`` is the differentiable
array-only boundary. Prepared execution currently rejects linear, IMEX,
parallel-sharded, and active run-to-saturation cases with explicit errors.

.. automodule:: gkx.api
   :members:
   :exclude-members: KrylovConfig
   :no-index:

Collision Operator Interface
----------------------------

.. automodule:: gkx.operators.collision
   :members:

Velocity-Space Core
-------------------

.. automodule:: gkx.core_velocity
   :members:

Geometry
--------

.. automodule:: gkx.geometry
   :members:
   :no-index:

Geometry Core
-------------

.. automodule:: gkx.geometry.core
   :members:
   :private-members:

Analytic Geometry
-----------------

.. automodule:: gkx.geometry.analytic
   :members:
   :private-members:

Flux-Tube Geometry
------------------

.. automodule:: gkx.geometry.flux_tube
   :members:
   :private-members:

Miller EIK Generation
---------------------

.. automodule:: gkx.geometry.miller_eik
   :members:
   :private-members:

VMEC EIK Generation
-------------------

.. automodule:: gkx.geometry.vmec_eik
   :members:
   :private-members:

Differentiable Geometry
-----------------------

.. automodule:: gkx.geometry.differentiable
   :members:

Differentiable Geometry Backend Discovery
-----------------------------------------

.. automodule:: gkx.geometry.backend_discovery
   :members:
   :private-members:

Differentiable Flux-Tube Contract
---------------------------------

.. automodule:: gkx.geometry.flux_tube_contract
   :members:
   :private-members:

Differentiable Geometry AD Checks
---------------------------------

.. automodule:: gkx.geometry.autodiff_checks
   :members:
   :private-members:

Differentiable Geometry Sensitivity
-----------------------------------

.. automodule:: gkx.geometry.sensitivity
   :members:
   :private-members:

Differentiable Boozer Bridge
----------------------------

.. automodule:: gkx.geometry.booz_xform_bridge
   :members:
   :private-members:

Differentiable VMEC Boozer Core
-------------------------------

.. automodule:: gkx.geometry.vmec_boozer_core
   :members:
   :private-members:

Differentiable VMEC Boozer Drifts
---------------------------------

.. automodule:: gkx.geometry.vmec_boozer_drifts
   :members:
   :private-members:

Differentiable VMEC Boozer Constants
------------------------------------

.. automodule:: gkx.geometry.vmec_boozer_constants
   :members:
   :private-members:

Differentiable VMEC Tensor Mapping
----------------------------------

.. automodule:: gkx.geometry.vmec_tensor_mapping
   :members:
   :private-members:

Differentiable Geometry Numerics
--------------------------------

.. automodule:: gkx.geometry.numerics
   :members:
   :private-members:

Grids
-----

.. automodule:: gkx.core_grid
   :members:

Species and Linear Parameters
-----------------------------

.. automodule:: gkx.operators.linear.params
   :members: Species, build_linear_params

Operators
---------

.. automodule:: gkx.operators
   :members:

Linear Operators
----------------

.. automodule:: gkx.operators.linear

Linear Linked Boundaries
------------------------

.. automodule:: gkx.operators.linear.linked
   :members:
   :private-members:

Linear Cache
------------

.. automodule:: gkx.operators.linear.cache_model
   :members:
   :no-index:

.. automodule:: gkx.operators.linear.cache_arrays
   :members:
   :private-members:
   :no-index:

.. automodule:: gkx.operators.linear.cache_builder
   :members:
   :no-index:

Linear Collisions
-----------------

.. automodule:: gkx.operators.linear.collisions
   :members:
   :no-index:

.. automodule:: gkx.operators.linear.collision_tables
   :members:
   :no-index:

Linear Moments
--------------

.. automodule:: gkx.operators.linear.moments
   :members:
   :private-members:

Linear Parameters
-----------------

.. automodule:: gkx.operators.linear.params
   :members:
   :private-members:
   :no-index:

Linear RHS
----------

.. automodule:: gkx.operators.linear.rhs
   :members:

Linear Dissipation
------------------

.. automodule:: gkx.operators.linear.dissipation
   :members:
   :private-members:

Linear Implicit Solvers
-----------------------

.. automodule:: gkx.solvers_linear_implicit
   :members:
   :private-members:

Linear Integrators
------------------

.. automodule:: gkx.solvers_linear_integrators
   :members:
   :private-members:

Linear Diagnostic Integration
-----------------------------

.. automodule:: gkx.solvers_linear_integrator_diagnostics
   :members:
   :private-members:

Linear Parallel RHS
-------------------

.. automodule:: gkx.solvers_linear_parallel
   :members:
   :private-members:

Linear Parallel Policy
----------------------

.. automodule:: gkx.solvers_linear_parallel_common
   :members:
   :private-members:

Linear Parallel Streaming
-------------------------

.. automodule:: gkx.solvers_linear_parallel_streaming
   :members:
   :private-members:

Linear Parallel Electrostatic Slices
------------------------------------

.. automodule:: gkx.solvers_linear_parallel_electrostatic
   :members:
   :private-members:

Linear Krylov Solvers
---------------------

.. automodule:: gkx.solvers_linear_krylov
   :members:
   :private-members:

Linear Eigenmode Solver Internals
---------------------------------

``gkx.solvers_linear_krylov`` owns ``KrylovConfig`` and the public
status-reporting wrapper, while the focused helper modules below own operator
application, branch selection, shift-invert preconditioning, and Arnoldi iterations.

.. automodule:: gkx.solvers_linear_krylov_algorithms
   :members:
   :private-members:
   :no-index:

Nonlinear Diagnostics
---------------------

.. automodule:: gkx.operators.nonlinear.diagnostics
   :members:
   :private-members:
   :no-index:

Nonlinear Diagnostic State
--------------------------

.. automodule:: gkx.operators.nonlinear.diagnostic_state
   :members:
   :private-members:

Nonlinear Collision Split Helpers
---------------------------------

.. automodule:: gkx.operators.nonlinear.collisions
   :members:
   :private-members:

Nonlinear Helpers
-----------------

.. automodule:: gkx.operators.nonlinear.policies
   :members:
   :private-members:
   :no-index:

Nonlinear Projection Helpers
----------------------------

.. automodule:: gkx.operators.nonlinear.projection
   :members:
   :private-members:

Nonlinear RHS
-------------

.. automodule:: gkx.operators.nonlinear.rhs
   :members:
   :private-members:

Nonlinear Bracket Kernels
-------------------------

.. automodule:: gkx.operators.nonlinear.brackets
   :members:
   :private-members:

Nonlinear Term Assembly
-----------------------

.. automodule:: gkx.terms.nonlinear
   :members:
   :private-members:

Nonlinear Explicit Step
-----------------------

.. automodule:: gkx.solvers_nonlinear_explicit
   :members:
   :private-members:

Nonlinear State Integration
---------------------------

.. automodule:: gkx.solvers_nonlinear_state_integration
   :members:
   :private-members:
   :no-index:

Nonlinear Diagnostic Drivers
----------------------------

.. automodule:: gkx.solvers_nonlinear_diagnostics
   :members:
   :private-members:

Nonlinear Diagnostic Integration
--------------------------------

.. automodule:: gkx.solvers_nonlinear_diagnostic_integration
   :members:
   :private-members:
   :no-index:

Nonlinear IMEX
--------------

.. automodule:: gkx.solvers_nonlinear_imex
   :members:
   :private-members:

Explicit Time Integrators
-------------------------

.. automodule:: gkx.solvers_time_explicit
   :members:
   :private-members:

Explicit Step Kernels
---------------------

.. automodule:: gkx.solvers_time_explicit_steps
   :members:
   :private-members:

Explicit Diagnostic Integrators
-------------------------------

.. automodule:: gkx.solvers_time_explicit_diagnostics
   :members:
   :private-members:

Explicit CFL Policy
-------------------

.. automodule:: gkx.solvers_time_explicit_cfl
   :members:
   :private-members:

Config-Driven Time Runners
--------------------------

.. automodule:: gkx.solvers_time_runners
   :members:
   :private-members:


Nonlinear Transport Optimization Diagnostics
--------------------------------------------

.. automodule:: gkx.diagnostics.nonlinear_transport_optimization
   :members:
   :private-members:

Nonlinear Gradient Statistics
-----------------------------

.. automodule:: gkx.diagnostics.nonlinear_gradient_statistics
   :members:
   :private-members:

Benchmarks
----------

.. automodule:: gkx.benchmarking_shared
   :members:
   :no-index:

Validation Gates
----------------

.. automodule:: gkx.diagnostics.validation_gates
   :members:

Autodiff Validation
-------------------

.. automodule:: gkx.objectives.autodiff_validation
   :members:

Parallelization
---------------

.. automodule:: gkx.parallel
   :members:

Parallel Identity Reports
-------------------------

.. automodule:: gkx.parallel.identity
   :members:
   :private-members:

Parallel Batch Mapping
----------------------

.. automodule:: gkx.parallel.batch
   :members:
   :private-members:

Parallel Independent Tasks
--------------------------

.. automodule:: gkx.parallel.independent
   :members:
   :private-members:

Nonlinear Parallel Spectral Core
--------------------------------

.. automodule:: gkx.operators.nonlinear.spectral_core
   :members:
   :private-members:

Nonlinear Domain Decomposition
------------------------------

.. automodule:: gkx.operators.nonlinear.domain_decomposition
   :members:
   :private-members:

Nonlinear Parallel Device-Z Core
--------------------------------

.. automodule:: gkx.operators.nonlinear.device_z
   :members:
   :private-members:

Velocity Sharding Plans
-----------------------

.. automodule:: gkx.parallel.velocity
   :members:
   :no-index:

.. automodule:: gkx.parallel.velocity_plan
   :members:

.. automodule:: gkx.parallel.velocity_hermite
   :members:
   :private-members:

.. automodule:: gkx.parallel.velocity_streaming
   :members:
   :private-members:

.. automodule:: gkx.parallel.velocity_drive
   :members:
   :private-members:

State Sharding Policy
---------------------

.. automodule:: gkx.parallel.state
   :members:

Sharded Integrators
-------------------

.. automodule:: gkx.parallel.integrators
   :members:

Zonal Validation
----------------

.. automodule:: gkx.diagnostics.zonal_validation
   :members:

Zonal Flow Objectives
---------------------

.. automodule:: gkx.objectives.zonal
   :members:

Analysis
--------

.. automodule:: gkx.diagnostics.analysis
   :members:
   :no-index:

Mode Diagnostics
----------------

.. automodule:: gkx.diagnostics.modes
   :members:

Growth-Rate Diagnostics
-----------------------

.. automodule:: gkx.diagnostics.growth_rates
   :members:
   :private-members:

Growth-Rate Fit Windows
-----------------------

.. automodule:: gkx.diagnostics.growth_windows
   :members:

Zonal Response Plots
--------------------

.. automodule:: gkx.artifacts.zonal_plots
   :members:

Publication Plotting
--------------------

.. automodule:: gkx.artifacts.plotting
   :members:
   :no-index:

Transport Figures
-----------------

.. automodule:: gkx.artifacts.transport_figures
   :members:
   :no-index:

Real-Space Snapshots
--------------------

.. automodule:: gkx.artifacts.snapshots
   :members:
   :no-index:

Automatic Run Figures
---------------------

.. automodule:: gkx.artifacts.run_figures
   :members:
   :no-index:

Whole-Run Summary Figure
------------------------

.. automodule:: gkx.artifacts.run_summary
   :members:
   :no-index:

Foreign Output Dispatch
-----------------------

.. automodule:: gkx.artifacts.foreign_output
   :members:
   :no-index:

GX Output Plotting
------------------

.. automodule:: gkx.artifacts.gx_output
   :members:
   :no-index:

Persistent Compilation Cache
----------------------------

.. automodule:: gkx.compilation_cache
   :members:
   :no-index:

Config
------

.. automodule:: gkx.config
   :members:

Normalization
-------------

.. automodule:: gkx.diagnostics.normalization
   :members:

Moment And Energy Diagnostics
-----------------------------

.. automodule:: gkx.diagnostics.moments
   :members:
   :private-members:

Transport Diagnostics
---------------------

.. automodule:: gkx.diagnostics.transport
   :members:
   :private-members:

Runtime Config
--------------

.. automodule:: gkx.workflows.runtime.config
   :members:

Runtime Startup
---------------

.. automodule:: gkx.workflows.runtime.startup
   :members:
   :private-members:

Runtime Policies
----------------

.. automodule:: gkx.workflows.runtime.policies
   :members:
   :private-members:

Runtime Diagnostics
-------------------

.. automodule:: gkx.workflows.runtime.diagnostics
   :members:
   :private-members:

Runtime Diagnostic Arrays
-------------------------

.. automodule:: gkx.workflows.runtime.diagnostic_arrays
   :members:
   :private-members:

Runtime Phi Initializer
-----------------------

.. automodule:: gkx.workflows.runtime.initial_phi
   :members:
   :private-members:

Runtime Resolution Estimator
----------------------------

.. automodule:: gkx.workflows.runtime.resolution
   :members:
   :private-members:

Runtime Chunks
--------------

.. automodule:: gkx.workflows.runtime.chunks
   :members:
   :private-members:

Runtime Results
---------------

.. automodule:: gkx.workflows.runtime.results
   :members:
   :private-members:

Runtime Orchestration
---------------------

.. automodule:: gkx.workflows.runtime.orchestration_scan
   :members:
   :private-members:

.. automodule:: gkx.workflows.runtime.orchestration_artifacts
   :members:
   :private-members:

.. automodule:: gkx.workflows.runtime.warm_start
   :members:
   :private-members:

Runtime Commands
----------------

.. automodule:: gkx.workflows.runtime.commands
   :members:
   :private-members:

Runtime TOML Inputs
-------------------

.. automodule:: gkx.workflows.runtime.toml
   :members:
   :private-members:

Runtime Artifact Package
------------------------

.. automodule:: gkx.artifacts
   :members:

Runtime Artifact IO
-------------------

.. automodule:: gkx.artifacts.io
   :members:
   :private-members:

NetCDF Spectral Layout
----------------------

.. automodule:: gkx.artifacts.spectral_layout
   :members:
   :private-members:

Nonlinear Output NetCDF
-----------------------

.. automodule:: gkx.artifacts.nonlinear_netcdf
   :members:
   :private-members:

Quasilinear Transport Diagnostics
---------------------------------

.. automodule:: gkx.diagnostics.quasilinear_transport
   :members:

Quasilinear Calibration
-----------------------

.. automodule:: gkx.diagnostics.quasilinear_calibration
   :members:
   :private-members:

Quasilinear Nonlinear-Window Gates
----------------------------------

.. automodule:: gkx.diagnostics.transport_windows
   :members:
   :private-members:

Run-to-Saturation Stop Policy
-----------------------------

.. automodule:: gkx.diagnostics.saturation
   :members:
   :private-members:

Quasilinear Model Selection
---------------------------

.. automodule:: gkx.diagnostics.quasilinear_model_selection
   :members:

Solver Eigen Objectives
-----------------------

.. automodule:: gkx.objectives.eigen
   :members:

Solver Objective Core
---------------------

.. automodule:: gkx.objectives.core
   :members:

Solver Objective Sampling
-------------------------

.. automodule:: gkx.objectives.sampling
   :members:

Solver Geometry Objectives
--------------------------

.. automodule:: gkx.objectives.geometry
   :members:
   :private-members:

Solver VMEC/Boozer Gradient Gates
---------------------------------

.. automodule:: gkx.objectives.vmec_boozer_gradients
   :members:
   :private-members:

Solver VMEC/Boozer Objectives
-----------------------------

.. automodule:: gkx.objectives.vmec_boozer
   :members:
   :private-members:

Solver VMEC/Boozer Finite-Difference Gates
------------------------------------------

.. automodule:: gkx.objectives.vmec_boozer_fd
   :members:
   :private-members:

Solver VMEC/Boozer Line-Search Gates
------------------------------------

.. automodule:: gkx.objectives.vmec_boozer_line_search
   :members:
   :private-members:

Parallel Decomposition Contracts
--------------------------------

.. automodule:: gkx.parallel.decomposition
   :members:

VMEC-JAX Transport Objective
----------------------------

.. automodule:: gkx.objectives.vmec_transport
   :members:
   :private-members:

VMEC-JAX Transport Branch Gates
-------------------------------

.. automodule:: gkx.objectives.vmec_transport_branch
   :members:

VMEC-JAX Transport Optimization
-------------------------------

.. automodule:: gkx.objectives.vmec_transport_optimization
   :members:

VMEC-JAX Boundary Chain
-----------------------

.. automodule:: gkx.geometry.vmec_boundary_chain
   :members:

Stellarator ITG Objectives
--------------------------

.. automodule:: gkx.objectives.stellarator
   :members:
   :no-index:




Stellarator Objective Portfolios
--------------------------------

.. automodule:: gkx.objectives.portfolio
   :members:
   :private-members:

Runtime Runner
--------------

.. automodule:: gkx.runtime
   :members:
