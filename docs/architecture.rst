Architecture
============

Core modules
------------

- ``gkx.core.velocity``: Hermite/Laguerre basis functions, gyroaverage
  coefficients, and polarization helpers.
- ``gkx.geometry``: analytic s-alpha flux-tube geometry.
- ``gkx.terms``: term-wise RHS kernels (streaming, mirror, drifts, drive, collisions, fields).
- ``gkx.operators.linear``: cache construction, linked-boundary maps,
  Hermite/Laguerre moment operators, linear parameter pytrees, and cached RHS
  assembly entry points.
- ``gkx.solvers.linear``: matrix-free eigensolver policy, linear
  fixed-step/diagnostic integration policy, implicit GMRES/preconditioner
  policy, and gated velocity-parallel linear RHS dispatch.
- ``gkx.solvers.nonlinear``: explicit RK/SSP/K10 and IMEX fixed-point,
  GMRES, and stage-composition policy.
- ``gkx.operators.nonlinear.diagnostics``: sampling, resolved-diagnostic
  packing, and ``SimulationDiagnostics`` construction shared by nonlinear
  diagnostic scans.
- ``gkx.operators.nonlinear.projection``: Hermitian and fixed-mode state
  projections used by compressed-real-FFT nonlinear scans and fixed-mode
  diagnostics.
- ``gkx.operators.nonlinear.collisions``: diagonal collision and
  hypercollision split policies shared by explicit and IMEX nonlinear scans.
- ``gkx.operators.nonlinear.policies``: diagnostic cache/weight/projection
  setup, adaptive time-step policy, fixed-mode omega masks used by comparison
  parity audits, reusable nonlinear IMEX operator construction, and public
  facades for the focused projection/collision owners.
- ``gkx.runtime`` / ``gkx.workflows.runtime.config``: user-facing runtime entrypoints and configuration schema.
- ``gkx.workflows.runtime.policies``: pure runtime selection policies for solver names, scan modes, nonlinear monitored modes, external fields, and step-count inference.
- ``gkx.workflows.runtime.orchestration_scan``, ``gkx.workflows.runtime.chunks``, and ``gkx.workflows.runtime.orchestration_artifacts``: runtime scan batching, progress/ETA formatting, and nonlinear restart/checkpoint artifact handoff behind injectable seams.
- ``gkx.benchmarking.shared``: reviewed reference tables,
  normalization constants, and comparison-only branch policies.
- ``gkx.diagnostics.growth_rates``: reusable growth/frequency fitting.
- ``gkx.artifacts.plotting``: reusable, publication-ready plotting utilities.

Term-level source mapping
-------------------------

- streaming, mirror, curvature, grad-B, diamagnetic, collisions,
  hypercollisions, hyperdiffusion, end damping:
  ``src/gkx/terms/linear_terms.py``
- field solves:
  ``src/gkx/terms/fields.py``
- nonlinear E×B, flutter, and Bessel-grid transforms:
  ``src/gkx/terms/nonlinear.py``
- assembled RHS:
  ``src/gkx/terms/assembly.py``

For the full operator equations, see :doc:`operators`.

Data flow
---------

The linear solve is structured as:

1. build the spectral grid and geometry
2. compute gyroaverage coefficients
3. convert ``LinearTerms`` into one canonical ``TermConfig``
4. solve the field equations for :math:`(\\phi, B_\\parallel, A_\\parallel)`
5. build the gyrokinetic variable ``H``
6. assemble RHS by summing per-term kernels from ``gkx.terms``
7. advance in time using ``integrate_linear``/diffrax/Krylov with the same RHS

This structure is intentionally modular so that nonlinear terms, collisions,
geometry adapters, and electromagnetic extensions can be inserted with minimal
refactoring.
