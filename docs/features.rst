Features
========

What GKX can be asked to do, with the switch that turns each feature on. The
input keys are defined in :doc:`inputs`; the physics behind them in
:doc:`theory` and :doc:`operators`.

Geometry
--------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Geometry
     - Select with
     - Notes
   * - s-alpha tokamak
     - ``[geometry] model = "s-alpha"`` (default)
     - Cyclone base case benchmarks
   * - Miller tokamak
     - ``model = "miller"``
     - in-package flux-surface construction
   * - VMEC or VMEX equilibrium
     - ``model = "vmec"``, or ``gkx wout_*.nc``
     - any radius ``s`` and field line ``alpha``; tokamaks and stellarators
   * - VMEX in memory
     - ``gkx.geometry.from_vmex(...)``
     - differentiable through the equilibrium
   * - Imported flux tube
     - ``model = "imported-netcdf"``
     - reads a precomputed ``*.eik.nc`` file
   * - Slab
     - ``model = "slab"``
     - secondary-instability and Landau tests

Physics
-------

- Any number of kinetic species; adiabatic (Boltzmann) electrons or ions.
- Electrostatic, or electromagnetic with :math:`A_\parallel` and
  :math:`B_\parallel` (KAW, KBM).
- Five collision operators: Lenard-Bernstein/Dougherty, Sugama, improved
  Sugama, drift-kinetic and gyrokinetic linearized Coulomb
  (``[time] collision_operator``); see :doc:`operators`.
- Hypercollisions and hyperdiffusion, declared as regularizations.
- Twist-and-shift linked parallel boundaries, or periodic.

Linear runs
-----------

- Initial-value runs with explicit (``rk2``, ``rk4``, ``sspx3``), IMEX or
  backward-Euler GMRES time stepping.
- Matrix-free restarted Krylov eigensolver, shift-invert Arnoldi, and a sparse
  direct shift-invert route; every returned eigenpair is certified against the
  full operator (:doc:`solvers`).
- ``k_y`` scans in serial or across devices, identical to the serial result
  (:doc:`parallelization`).

Nonlinear runs
--------------

- Pseudo-spectral :math:`E\times B` bracket with the two-thirds rule, in the
  full or half ``k_y`` layout (``[grid] ky_layout``).
- ``run_to = "saturation"`` stops when heat flux, field and free energy are
  stationary; a missed saturation is reported.
- Restart from a saved bundle (``[init] init_file``), NetCDF output with
  spectra, fluxes and energies (:doc:`outputs`).
- ``gkx --estimate`` sizes the grid from the geometry and prints the reason
  for each entry.

Derivatives and optimization
----------------------------

- Implicit derivatives of eigenvalues and eigenvector observables
  (:doc:`differentiable_eigensolver`).
- A checkpointed discrete adjoint of the nonlinear heat flux over a window
  (:doc:`nonlinear_autodiff`).
- Quasilinear heat-flux spectra and their sensitivities (:doc:`quasilinear`).
- Growth-rate, quasilinear and nonlinear objectives in VMEX shape
  optimization (:doc:`stellarator_optimization`).

Hardware and precision
----------------------

- CPU and GPU through JAX; ``float32`` by default, ``float64`` with
  ``JAX_ENABLE_X64=true``.
- A persistent compilation cache; ``gkx.prepare`` compiles a nonlinear route
  once for repeated calls.
