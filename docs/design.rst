Design and Code Map
===================

Why GKX is built the way it is, and where each part lives. Each choice has a
cost; the cost is stated next to it. Measurements are in :doc:`algorithms`.

Design choices
--------------

**Flux tube.** The gyrokinetic equation is solved in a thin tube that follows
one field line. It resolves the turbulence scale at a small fraction of the
cost of a surface or volume. The price: profile variation across the tube is
neglected, and the parallel ends need a boundary condition. GKX uses
twist-and-shift linking, solved on the modes each chain couples. For a
stellarator, tubes at several ``alpha`` sample a surface.

**Hermite-Laguerre velocity space.** The distribution is expanded in Hermite
polynomials in :math:`v_\parallel` and Laguerre polynomials in :math:`\mu`.
The gyroaverage becomes a sum and low-order moments are fluid quantities, so a
coarse truncation is a model, not an under-resolved grid. The cost: a
truncated Hermite ladder reflects free energy from its last moment, so GKX
damps the highest moments with hypercollisions and says so. GX uses the same
representation, which makes it the closest parity reference.

**JAX.** One code path for CPU and GPU, and exact derivatives of the
discretized equations, including the equilibrium when VMEX supplies it.
Reverse mode through an iterative solve is replaced by implicit rules
(eigenpairs) and checkpointed adjoints (time windows). The costs: compile time
on the first call (cached afterwards), static shapes, and ``float32`` graphs
that are not always bitwise reproducible.

**Time stepping.** The nonlinear step is explicit Runge-Kutta with a CFL
controller, because the :math:`E\times B` bracket changes every step. Stiff
linear terms can be implicit (``imex``, ``imex-ars3``); see :doc:`guidance`
for when. Linear growth rates come from the eigensolvers, which are faster and
certified.

**Collisions.** The default is the linearized Landau operator, built from
Rosenbluth potentials in the Hermite-Laguerre basis. It conserves particles,
momentum and energy, satisfies the H-theorem, and works at any resolution and
species set. Lenard-Bernstein/Dougherty, Sugama and Coulomb variants remain
selectable (:doc:`operators`).

**End damping.** Linked chains need an absorber at the field-line ends. One
rate, ``[time] damp_ends_rate``, applies on every route and no time step
enters it (:ref:`end-damping`).

**Eigensolver.** The default is ``auto``: a Hermite-banded direct shift-invert where its
blocks fit in memory, else the residual-certified Arnoldi on the time-advance
map (``adaptive``). Matrix-free shift-invert is opt-in for a chosen branch
(:doc:`solvers`, :doc:`algorithms`).

**Certify, do not trust.** Every eigenpair is checked against the unprojected
operator, and a failed check is an error. Under-resolved or unsaturated runs
warn. Published numbers are tied to artifacts and recomputed in CI through the
evidence ledger (:doc:`validation`).

**:math:`k_y` layouts.** Fields are real, so negative :math:`k_y` is the
conjugate of positive. ``full`` (default) stores both halves; ``half`` stores
:math:`k_y\ge0`. On CPU ``half`` runs an RK3 step in 0.54--0.60 of the time,
but the checkpointed gradient takes 1.28 times as long. One module,
:mod:`gkx.core_ky_layout`, owns the conversions and reduction weights.

Code map
--------

Start at ``gkx.load`` and ``gkx.solve`` and follow the call down:

- ``config.py``, ``workflows/runtime/{toml,wout,startup}.py``: decks,
  defaults, geometry, initial state.
- ``runtime.py`` dispatches to ``workflows/linear.py`` or
  ``workflows/nonlinear.py``; ``cli.py`` wraps both.
- ``terms/`` and ``operators/``: the gyrokinetic equation (fields, streaming,
  drifts, collisions, :math:`E\times B`).
- ``solvers_*.py``: time integrators, Krylov and shift-invert eigensolvers,
  nonlinear explicit and IMEX steps.
- ``geometry/``: s-alpha, slab, Miller, VMEC/VMEX flux tubes and the
  differentiable bridge.
- ``core_grid.py``, ``core_velocity.py``, ``core_ky_layout.py``: spectral grid
  and velocity basis.
- ``diagnostics/``, ``objectives/``: growth fits, quasilinear weights,
  saturation windows; differentiable objectives.
- ``parallel/``, ``artifacts/``: independent maps and state sharding; NetCDF,
  summaries, figures.

A linear solve builds the grid and geometry, the ``LinearCache``, the term
configuration, then evaluates the right-hand side (field solve, :math:`H` from
:math:`G`, per-term kernels, ``terms.assembly.assemble_rhs_cached``) and either
integrates it in time or hands it to the eigensolver. A nonlinear solve adds
the brackets of ``terms/nonlinear.py`` to the same right-hand side.

Geometry paths
--------------

Every path ends in one solver-ready object: ``SAlphaGeometry``,
``SlabGeometry`` or ``FluxTubeGeometryData``.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``[geometry] model``
     - Path
   * - ``s-alpha`` (default), ``slab``
     - Analytic, ``gkx.geometry.analytic``.
   * - ``miller``
     - ``gkx.geometry.miller_eik`` writes an ``*.eiknc.nc`` file; the runtime
       loads it as ``imported-eik``.
   * - ``vmec``
     - ``gkx.geometry.vmec_eik`` writes an ``*.eik.nc`` file; the runtime
       loads it as ``vmec-eik``. ``gkx wout_XXX.nc`` builds the deck.
   * - ``imported-netcdf``, ``imported-eik``, ``vmec-eik``, ``desc-eik``
     - Read from ``geometry_file``.
   * - in memory (differentiable)
     - ``gkx.geometry.differentiable``; from a VMEX state through Boozer
       coordinates or directly from VMEX tensors (``from_vmex``,
       ``from_vmex_wout``).

Equations to code
-----------------

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Equation or method
     - Source (under ``src/gkx``)
     - Tests (under ``tests``)
   * - Hermite/Laguerre basis, gyroaverage, polarization
     - ``core_velocity.py``
     - ``unit/core/test_core_numerics.py``
   * - Field equations: quasineutrality, Ampère, pressure balance
     - ``terms/fields.py``, ``operators/linear/moments.py``
     - ``unit/operators/test_terms_fields.py``
   * - Streaming, mirror, curvature and grad-B drift, drive
     - ``terms/linear_terms.py``, ``operators/linear/streaming.py``
     - ``unit/linear/test_linear.py``
   * - Collisions (Landau, Dougherty, Sugama), hypercollisions, end damping
     - ``operators/linear/{collisions,collision_tables,dissipation}.py``
     - ``validation/physics_gates/test_collision_physics.py``
   * - Linked (twist-shift) boundary
     - ``operators/linear/linked.py``
     - ``unit/linear/test_linear.py``
   * - :math:`E\times B` and flutter nonlinearity
     - ``terms/nonlinear.py``, ``operators/nonlinear/``
     - ``unit/nonlinear/test_nonlinear_exb.py``
   * - Linear integrators and implicit solves
     - ``solvers_linear_integrators.py``, ``solvers_time_explicit*.py``,
       ``solvers_linear_implicit.py``
     - ``unit/solvers/test_time_integrators.py``
   * - Eigenmodes: Krylov, shift-invert, adaptive propagator
     - ``solvers_linear_krylov*.py``, ``solvers_linear_precond_pr3.py``,
       ``solvers_linear_adaptive_propagator.py``
     - ``unit/solvers/test_linear_krylov_core.py``
   * - Nonlinear integrators (explicit, IMEX)
     - ``solvers_nonlinear_*.py``
     - ``unit/nonlinear/test_nonlinear.py``
   * - Free energy, spectra, fluxes
     - ``operators/moments.py``, ``operators/fluxes.py``
     - ``unit/nonlinear/test_nonlinear.py``
   * - Growth-rate fits and windows
     - ``diagnostics/growth_rates.py``, ``diagnostics/modes.py``
     - ``unit/diagnostics/test_analysis.py``
   * - Implicit eigenvalue derivatives
     - ``objectives/eigen.py``
     - ``unit/objectives/test_autodiff_solver_objectives.py``

Operator equations are in :doc:`operators`.

Repository rules
----------------

- ``examples/``: small workflows to copy. ``benchmarks/``: drivers, compact
  TOML inputs and ``benchmarks/results/manifest.toml``; raw products stay out
  of git. ``tools/``: manifests and the evidence ledger. ``tests/``: gates that
  need no raw outputs (:doc:`testing`).
- ``tools/package_architecture_manifest.toml`` fixes the module layout, the
  per-module line cap and the repository line budget. Add a module only when it
  replaces several files or isolates a measured compilation boundary. Prefer
  the domain packages to new root-level modules.
- Source modules use physics and numerical names; external-code names stay in
  ``benchmarks/`` and ``tools/comparison``.
- Public: the modules in :doc:`api`, ``gkx.config``, ``gkx.runtime``,
  ``gkx.geometry``, ``gkx.cli``, ``gkx.artifacts``, ``gkx.parallel``. Free to
  move if behavior and tests hold: ``gkx.terms.*``, most of
  ``gkx.workflows.runtime``, geometry adapters. Package initializers resolve
  names lazily and import no JAX-heavy stack.
- ``tools/validation_coverage_manifest.toml`` accounts for every module under
  ``src/gkx`` (``scripts/checks/check_validation_coverage_manifest.py``). Give a
  module a direct row if it has public imports, its own physics contract or
  high change risk; otherwise list it under an owner row. Every
  ``.. automodule:: gkx.*`` in :doc:`api` must resolve to a listed module.
- Git keeps inputs, tests, small gate reports and lightweight previews. Heavy
  outputs live in release artifacts with checksums
  (``tools/release_artifact_manifest.toml``,
  ``scripts/checks/check_repository_size_manifest.py``). A history rewrite needs
  a coordinated window: every clone must reset afterwards.
