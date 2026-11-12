API Reference
=============

Public API Registry
-------------------

``gkx.__all__`` holds 16 names: ``__version__`` and the 15 names of
``gkx.api.__all__``.

- Workflow: ``load``, ``solve``, ``scan``, ``plot``, ``prepare``.
- Contracts: ``PreparedSimulation``, ``Case``, ``LinearResult``,
  ``NonlinearResult``, ``ScanResult``.
- VMEX turbulence coupling: ``flux_tube_geometry_from_mapping``,
  ``solver_objective_vector_from_geometry``,
  ``solver_linear_operator_matrix_from_geometry``,
  ``solver_scalar_objective_from_vector``, ``VMEXTransportObjectiveConfig``.

Every name resolves lazily from ``gkx.api._EXPORT_TARGETS``, so ``import gkx``
does not import the solver stack. The same table keeps the GKX 1.x root names:
they still import by name (``from gkx import KrylovConfig``) but are not in
``gkx.__all__``, ``dir(gkx)``, or wildcard imports. They may be removed no
earlier than GKX 3.0; import them from the owning module listed on this page.

Contract types
--------------

``Case``, ``LinearResult``, ``NonlinearResult``, and ``ScanResult`` are aliases,
not wrappers, of ``RuntimeConfig``, ``RuntimeLinearResult``,
``RuntimeNonlinearResult``, and ``RuntimeLinearScanResult``. Result arrays are
never copied, and the ``Runtime*`` names remain valid imports.

Workflow
--------

::

   case = gkx.load("cyclone.toml")
   result = gkx.solve(case)
   figure, axes = gkx.plot(result)
   spectrum = gkx.scan(case, [0.1, 0.2, 0.3])
   simulation = gkx.prepare(case, steps=64)
   result = simulation.solve()

``load``
   Reads a TOML deck through ``gkx.workflows.runtime.toml`` and resolves
   case-relative paths.
``solve``
   ``gkx.runtime.solve``: runs the nonlinear runtime when
   ``physics.nonlinear`` is set, otherwise the linear runtime.
``scan``
   ``gkx.runtime.run_runtime_scan``: a linear :math:`k_y` scan.
``plot``
   ``gkx.artifacts.plotting.plot``: returns ``(Figure, axes)`` for a linear,
   scan, or nonlinear result, and neither saves nor shows the figure.
``prepare``
   ``gkx.api.prepared.prepare_simulation``: validates the case and returns a
   frozen ``PreparedSimulation``.

When the deck has no ``[run]`` table and no ``Nl``/``Nm`` argument is given,
``(Nl, Nm)`` defaults to (12, 24) for a linear case and (4, 8) for a nonlinear
case, the same fallback the runtime and the CLI use.

Prepared simulations
--------------------

``PreparedSimulation`` accepts linear and nonlinear cases and owns no numerics:
every method delegates to the runtime path the case would take anyway.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Member
     - Behavior
   * - ``solve(parameters=None, initial_state=None)``
     - Nonlinear: returns ``(time, diagnostics, state, fields)`` from the
       prepared backend. ``diagnostics``
       may be ``None`` when disabled; ``fields`` is a ``FieldState``.
       Linear: calls ``gkx.runtime.solve`` and returns its ``LinearResult``.
       ``parameters`` must be ``None``; a linear ``initial_state`` raises
       ``NotImplementedError``.
   * - ``scan(parameter, values, *, parallel="auto")``
     - Only ``parameter="ky"``; delegates to ``run_runtime_scan``. An integer
       ``parallel`` sets the worker count.
   * - ``value_and_grad(objective, parameters)``
     - ``jax.value_and_grad(objective)(parameters)``.
   * - ``warmup()``
     - Nonlinear: completes one solve so XLA compiles it, and returns ``self``;
       a second call does nothing. Linear: no-op, because the linear runtime
       picks its solver per call.
   * - ``estimate_memory()``
     - Bytes for the complex state and the stage copies an explicit step keeps
       live. A floor: diagnostic buffers are not counted.
   * - ``summary()``, ``print_summary()``, ``compilation_metadata()``
     - Kind, state shape, ``(Nl, Nm)``, geometry model, precision, devices,
       working set, persistent-cache state, and whether it was warmed.
       ``compiled_at_prepare`` is always ``False``: XLA compiles on first
       execution.
   * - ``precision``, ``devices``
     - ``"float64"`` or ``"float32"`` from ``jax_enable_x64``; visible JAX
       devices as ``platform:id``.

Preparing a nonlinear case builds its diagnostic backend through
``gkx.runtime.prepare``; see :doc:`parallelization` for supported named policies.
Explicit ``steps=N`` fixes a finite window. Without that budget, adaptive
``solve`` and ``warmup`` reuse compiled chunks until ``t_max``; every call resets
the clock and controller, retaining executable caches rather than physical state.
A saturation-stopping deck raises unless ``steps=N`` is passed or
``run_to = "t_max"`` is selected. Use ``gkx.solve`` or the CLI to retain
saturation stopping. Adaptive chain IMEX and generic ``imex`` preparation
remain unsupported.

With ``time.diagnostics_interval`` set, prepared ``run`` selects retained
rows during host-side result assembly, including an adaptive full-horizon run.
That ``run`` is host-only and refuses traced state or model inputs.

For a fixed timestep and known step count, ``run_arrays`` gathers the interval
samples with a static index schedule. This preserves fixed output shapes,
``jax.jit`` and autodiff for state and matched model inputs. The schedule follows
accumulated step times and includes the final record; it does not interpolate.
It uses the prepared horizon and refuses a ``time_horizon`` override. Use
host-side ``run`` when overriding the horizon.

An adaptive trajectory has a variable number of interval samples, so its
``run_arrays`` refuses an active physical-time interval. Leave the interval
unset for the existing adaptive fixed-shape array contract and pass explicit
``steps`` for a finite differentiated window. ``output.time_interval`` belongs
to file writing and does not change returned backend diagnostics.
``chunk_steps`` controls the host loop of an inferred adaptive full-horizon
run. Fixed-window preparation refuses an explicit chunk length; omit it to
retain the fixed compiled scan.

The lower-level ``gkx.runtime.prepare`` accepts nonlinear cases and returns a
backend with ``run`` and ``run_arrays``. ``run`` returns the same four-part
result as the nonlinear wrapper. ``run_arrays`` returns
``(state, (diagnostic_arrays, time, dt_series), fields)`` without host
conversion or diagnostic assembly. Prepare with explicit ``steps`` for this
finite-window interface; the host loop for an adaptive full horizon cannot
be transformed with JAX.

For fixed-step model differentiation, ``run_arrays`` accepts ``cache`` and
``params`` together, with optional matched ``geometry``. A geometry override
requires both cache and parameters. Named-collective model inputs must retain
the prepared tree structure, array shapes and dtypes. Adaptive model overrides
are refused, and ``time_horizon`` cannot be combined with model overrides.
The diagnostics-off named final-state path also refuses model overrides.

For example, the finite-window nonlinear backend exposes arrays directly:

.. code-block:: python

   import gkx.runtime as runtime

   nonlinear_case = gkx.load("examples/03_nonlinear_tokamak/case.toml")
   backend = runtime.prepare(nonlinear_case, steps=64)
   state, (diagnostic_arrays, time, dt_series), fields = backend.run_arrays()

Keep the case topology and numerical policy fixed when reusing this backend.
A full-horizon call restarts its clock and timestep controller on each run;
continuing a physical trajectory requires an explicit state and a suitable
continuation policy.

For the same finite window, the prepared route and
``integrate_nonlinear_explicit_diagnostics_state`` compile the same graph
and return bitwise identical arrays
(``tests/unit/nonlinear/test_nonlinear.py``). Prefer ``prepare`` when a case
runs more than once: it keeps the compiled graph, while each call to the
function entry point compiles its own. :doc:`solvers` gives the contract and
the state-dtype caveat.

On a linked (twist-shift) deck, every nonlinear state these entry points are
given -- the one ``prepare`` is built with and any ``initial_state`` passed to
``PreparedSimulation.solve`` -- is projected onto the linked chain cover before
it is integrated, exactly as the runtime projects a supplied ``initial_state``
at intake. The rows outside the chains are decoupled, and the ExB bracket would
alias whatever sits there back onto the chain rows. The projection is a mask
fixed by the deck's topology, not a check on the state, so it holds under
``jit`` and under reverse-mode AD: the cotangent of a supplied state is exactly
zero on those rows and unchanged elsewhere. Periodic decks and full-cover
linked grids are untouched. The raw drivers underneath, ``integrate_nonlinear``
and ``integrate_nonlinear_cached``, project a supplied state the same way.
:doc:`solvers` states the rule and what it costs.

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

Differentiable VMEC Boozer Core
-------------------------------

.. automodule:: gkx.geometry.vmec_boozer_core
   :members:
   :private-members:

Differentiable VMEC Boozer Drifts
---------------------------------

.. automodule:: gkx.geometry.vmec_boozer_derivatives
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

.. automodule:: gkx.core_ky_layout
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

.. automodule:: gkx.solvers.linear.implicit
   :members:
   :private-members:

Linear Integrators
------------------

.. automodule:: gkx.solvers.linear.integrators
   :members:
   :private-members:

Linear Parallel RHS
-------------------

.. automodule:: gkx.solvers.linear.parallel
   :members:
   :private-members:

Linear Parallel Policy
----------------------

.. automodule:: gkx.solvers.linear.parallel_common
   :members:
   :private-members:

Linear Parallel Streaming
-------------------------

.. automodule:: gkx.solvers.linear.parallel_streaming
   :members:
   :private-members:

Linear Parallel Electrostatic Slices
------------------------------------

.. automodule:: gkx.solvers.linear.parallel_electrostatic
   :members:
   :private-members:

Linear Krylov Solvers
---------------------

.. automodule:: gkx.solvers.linear.krylov
   :members:
   :private-members:

Linear Eigenmode Solver Internals
---------------------------------

``gkx.solvers.linear.krylov`` owns ``KrylovConfig`` and the public
status-reporting wrapper; ``gkx.solvers.linear.krylov_algorithms`` owns operator
application, branch selection, shift-invert preconditioning, and the Arnoldi
iterations.

.. automodule:: gkx.solvers.linear.krylov_algorithms
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

.. automodule:: gkx.solvers.nonlinear.explicit
   :members:
   :private-members:

Nonlinear Integrating-Factor Step
---------------------------------

.. automodule:: gkx.solvers.nonlinear.lawson
   :members:
   :private-members:

Nonlinear State Integration
---------------------------

.. automodule:: gkx.solvers.nonlinear.diagnostics
   :members:
   :private-members:
   :no-index:

Nonlinear Diagnostic Drivers
----------------------------

.. automodule:: gkx.solvers.nonlinear.diagnostics
   :members:
   :private-members:

Nonlinear Diagnostic Integration
--------------------------------

.. automodule:: gkx.solvers.nonlinear.diagnostics
   :members:
   :private-members:
   :no-index:

Nonlinear IMEX
--------------

.. automodule:: gkx.solvers.nonlinear.imex
   :members:
   :private-members:

Explicit Time Integrators
-------------------------

.. automodule:: gkx.solvers.time_explicit
   :members:
   :private-members:

Config-Driven Time Runners
--------------------------

.. automodule:: gkx.solvers.time_runners
   :members:
   :private-members:

Benchmarks
----------

.. automodule:: gkx.benchmarking_shared
   :members:
   :no-index:

Autodiff Validation
-------------------

.. automodule:: gkx.objectives.autodiff_validation
   :members:

Parallelization
---------------

.. automodule:: gkx.parallel
   :members:

Parallel Independent Tasks
--------------------------

.. automodule:: gkx.parallel.independent
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

.. automodule:: gkx.operators.moments
   :members:
   :private-members:

Runtime Config
--------------

.. automodule:: gkx.config
   :members:
   :no-index:

Runtime Startup
---------------

.. automodule:: gkx.workflows.runtime.startup
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

Runtime Resolution Estimator
----------------------------

.. automodule:: gkx.workflows.runtime.wout
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

.. automodule:: gkx.workflows.runtime.artifacts
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

Run-to-Saturation Stop Policy
-----------------------------

.. automodule:: gkx.diagnostics.saturation
   :members:
   :private-members:

Solver Eigen Objectives
-----------------------

.. automodule:: gkx.objectives.eigen
   :members:

Solver Objective Core
---------------------

.. automodule:: gkx.objectives.core
   :members:

VMEC-JAX Transport Objective
----------------------------

.. automodule:: gkx.objectives.vmec_transport
   :members:
   :private-members:

Runtime Runner
--------------

.. automodule:: gkx.runtime
   :members:
