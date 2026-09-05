Research roadmap
================

The maintained execution plan is :download:`plan.md <../plan.md>`, revised
on 2026-09-04 against GKX 2.0.0 at ``a99dac89``. It includes the current audit,
physics and numerical benchmarks, CPU/GPU profiling and sharding, source
slimming, documentation, VMEX/ESSOS optimization and publication milestones.

Destination and present boundary
--------------------------------

The goal is a research-grade ES/EM, linear/quasilinear/nonlinear, multispecies
code with physical collision/closure models, radial electric fields, auditable
derivatives and CPU/GPU parallelism. It includes vacuum and finite-beta devices,
VMEX equilibria, and direct ESSOS coil fields with islands. Compact source is a
maintainability objective, not a restriction of the scientific destination.

The current local Maxwellian delta-f model is a starting point. The roadmap
separates implemented, verified and benchmarked combinations; it does not claim
that all species, fields, collisions, geometries and derivative paths compose.
Predictive use requires independent validation within a stated domain.

.. list-table:: Physics development gates
   :header-rows: 1
   :widths: 20 40 40

   * - Track
     - Target
     - Evidence before promotion
   * - R1 / C0–C4
     - Model collisions through linearized Frei and nonlinear Jorge Coulomb
     - Support ranges, pair conservation, weighted entropy, published coefficients,
       linearization limits and a consistent background model
   * - R1 / E0–E2
     - Local shear, stellarator equilibrium Er, transport-consistent Er
     - Remap/stage/AD checks, domain-aware physics and ambipolar-root conditions;
       shear alone is not the complete Er model
   * - R1–R2 / EM
     - Coupled phi, Apar, Bpar and kinetic multispecies at finite beta
     - Energy/field identities, real-mass-ratio KAW/KBM/microtearing and nonlinear
       transport; distinguish equilibrium and fluctuation beta
   * - R8a–e
     - Nested coil realization and direct coil-field/island turbulence
     - Non-flux geometry, conservative spatial/field operators, background
       residual, boundaries, nested limit, transport convergence and derivatives

Execution order
---------------

#. Resolve damping semantics and precision/backend regressions.
#. Verify the equations, fields, boundaries, collisions and discrete derivatives.
#. Validate nonlinear statistics and resolution across tokamak and stellarator cases.
#. Establish working distributed primal and derivative paths, then measure scaling.
#. Consolidate source, data, documentation and examples without removing test coverage.
#. Demonstrate linear and quasilinear design before accepting nonlinear transport reduction.
#. Test ESSOS coil realization and transport robustness on the realized equilibrium.
#. Develop non-flux geometry/island and background models alongside the local
   validation work; require their own equations and benchmarks before direct-coil optimization.
#. Publish only claims supported by reproducible, independently checked results.

The previous phase lists are retained in Git history, not maintained in parallel
here. For present scientific limitations, see :doc:`research_grade_plan` and
:doc:`release_scope`. For the existing preliminary QA campaign and its rejection
criteria, see :doc:`stellarator_optimization`.
