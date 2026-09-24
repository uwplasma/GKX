Architecture
============

This page is the module map of ``src/gkx``. :doc:`code_structure` adds the
repository roles, the public/internal boundary, and the equation-to-code
table; :doc:`api` documents each module.

Core modules
------------

- ``gkx.core_velocity``: Hermite/Laguerre basis, gyroaverage coefficients, and
  polarization helpers.
- ``gkx.core_grid`` and ``gkx.core_ky_layout``: spectral grids and the
  :math:`k_y` axis layout contract.
- ``gkx.geometry``: every flux-tube geometry path (see `Geometry paths`_).
- ``gkx.terms``: term-wise RHS kernels (``linear_terms``: streaming, mirror,
  curvature/grad-B drift, diamagnetic drive; ``fields``: field solves;
  ``nonlinear``: ExB and electromagnetic brackets; ``assembly``: the summed
  RHS).
- ``gkx.operators.linear``: the ``LinearCache`` (model, array factories,
  builder), linked-boundary maps, Hermite/Laguerre moments and ``build_H``,
  parameter pytrees, streaming kernels, collisions and dissipation
  (collisions, hypercollisions, hyperdiffusion, end damping).
- ``gkx.operators.nonlinear``: nonlinear RHS, bracket kernels, diagnostic
  state and sampling, Hermitian/fixed-mode projection, collision splitting,
  and the spectral/domain/device-z parallel contracts.
- ``gkx.operators.moments`` and ``gkx.operators.fluxes``: quadrature, free
  energy, spectra, and heat/particle fluxes.
- ``gkx.solvers_linear_*``: fixed-step and diagnostic integrators, implicit
  GMRES/preconditioner solves, Krylov/shift-invert eigenmodes (with the
  ``pr3-cm`` preconditioner and the adaptive propagator), and velocity-parallel
  RHS routes.
- ``gkx.solvers_nonlinear_*``: explicit RK/SSP/K10 and IMEX integration, state
  integration, and diagnostic drivers.
- ``gkx.solvers_time_*``: explicit linear integrators, CFL policy, and
  config-driven runners.
- ``gkx.diagnostics``: mode extraction, growth-rate fits and windows,
  transport windows, quasilinear transport and calibration, zonal validation,
  saturation stop policy, validation gate reports.
- ``gkx.objectives``: solver objectives, implicit eigenvalue AD, VMEC/Boozer
  and VMEC-JAX transport objectives, zonal and stellarator objectives,
  portfolios, autodiff validation.
- ``gkx.parallel``: batch and independent-task parallelism, velocity-space
  sharding plans and kernels, sharded integrators, identity reports.
- ``gkx.config`` and ``gkx.runtime``: the ``RuntimeConfig`` (``Case``) schema
  and the runtime facade (``solve``, ``prepare``, scans).
- ``gkx.workflows``: the executable linear and nonlinear workflows and the
  ``workflows.runtime`` support package (TOML loading, startup, policies,
  chunks, scan orchestration, results, artifacts, commands, ``wout``
  shorthand).
- ``gkx.artifacts``: NetCDF/CSV/JSON writers and readers, restart state,
  plotting, run figures, and readers for GX output.
- ``gkx.api`` and ``gkx.cli``: the lazy public registry, the
  ``PreparedSimulation`` object, and the ``gkx`` executable.
- ``gkx.benchmarking_shared``: reviewed reference tables, normalization
  constants, and comparison-only branch policies.

Geometry paths
--------------

Every path ends in one solver-ready object: ``SAlphaGeometry``,
``SlabGeometry``, or ``FluxTubeGeometryData``.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - ``[geometry] model``
     - Path
   * - ``s-alpha`` (default), ``slab``
     - Analytic models in ``gkx.geometry.analytic``, built by
       ``gkx.geometry.core.build_flux_tube_geometry``.
   * - ``miller``
     - ``gkx.geometry.miller_eik`` writes a ``*.eiknc.nc`` file through
       ``gkx.geometry.imported_miller`` (with the shared kernels in
       ``gkx.geometry.kernels``); the runtime then loads it as
       ``imported-eik``.
   * - ``vmec``
     - ``gkx.geometry.vmec_eik`` writes a ``*.eik.nc`` file through
       ``gkx.geometry.imported_vmec`` (backend lookup in
       ``backend_discovery``, sampling in ``vmec_field_line_sampling``,
       Boozer derivatives in ``vmec_boozer_derivatives``, state controls in
       ``vmec_state_controls``); the runtime then loads it as ``vmec-eik``.
       ``gkx wout_XXX.nc`` builds such a deck from a WOUT file
       (``gkx.workflows.runtime.wout``).
   * - ``imported-netcdf``, ``imported-eik``, ``vmec-eik``, ``desc-eik``
     - ``geometry_file`` is read by
       ``gkx.geometry.flux_tube.load_imported_geometry_netcdf``.
   * - in memory (differentiable)
     - ``gkx.geometry.differentiable`` is the facade;
       ``flux_tube_geometry_from_mapping`` turns a mapping that satisfies
       ``gkx.geometry.flux_tube_contract`` into ``FluxTubeGeometryData``.
       Mappings come from a VMEX state through Boozer coordinates
       (``vmec_boozer_core``, ``vmec_boozer_drifts``,
       ``vmec_boozer_constants``) or directly from VMEX tensors
       (``vmec_tensor_mapping``: ``from_vmex``, ``from_vmex_wout``,
       ``from_vmex_mirror``). ``autodiff_checks``, ``sensitivity``,
       ``booz_xform_bridge``, and ``numerics`` hold the
       AD/finite-difference checks and their helpers.

``gkx.geometry.core`` also owns twist-shift parameters and the grid defaults a
geometry imposes.

Data flow
---------

A linear solve:

1. build the spectral grid and the geometry;
2. build the ``LinearCache`` (gyroaverage coefficients, drifts, linked maps);
3. convert ``LinearTerms`` into one ``TermConfig``
   (``linear_terms_to_term_config``);
4. solve the fields :math:`(\phi, A_\parallel, B_\parallel)`
   (``gkx.terms.fields.solve_fields``);
5. build :math:`H` from :math:`G` and the fields
   (``gkx.operators.linear.moments.build_H``);
6. sum the per-term kernels (``gkx.terms.assembly.assemble_rhs_cached``);
7. integrate in time (``integrate_linear``) or solve the eigenproblem
   (``gkx.solvers_linear_krylov``) with the same RHS.

A nonlinear solve adds the bracket terms from ``gkx.terms.nonlinear`` to the
same RHS and integrates it with ``gkx.solvers_nonlinear_*``. The full operator
equations are in :doc:`operators`.
