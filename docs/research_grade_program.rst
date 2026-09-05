Research roadmap
================

The maintained execution plan is :download:`plan.md <../plan.md>`, revised
on 2026-09-04 against GKX 2.0.0 at ``a99dac89``. It includes the current audit,
physics and numerical benchmarks, CPU/GPU profiling and sharding, source
slimming, documentation, VMEX/ESSOS optimization and publication milestones.

Execution order
---------------

#. Resolve damping semantics and precision/backend regressions.
#. Verify the equations, fields, boundaries, collisions and discrete derivatives.
#. Validate nonlinear statistics and resolution across tokamak and stellarator cases.
#. Establish working distributed primal and derivative paths, then measure scaling.
#. Consolidate source, data, documentation and examples without removing test coverage.
#. Demonstrate linear and quasilinear design before accepting nonlinear transport reduction.
#. Test ESSOS coil realization and transport robustness on the realized equilibrium.
#. Publish only claims supported by reproducible, independently checked results.

The previous phase lists are retained in Git history, not maintained in parallel
here. For present scientific limitations, see :doc:`research_grade_plan` and
:doc:`release_scope`. For the existing preliminary QA campaign and its rejection
criteria, see :doc:`stellarator_optimization`.
