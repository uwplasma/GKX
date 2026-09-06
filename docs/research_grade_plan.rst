Research status
===============

Reviewed against main a99dac89 on 2026-09-06.
The active execution queue is
`plan.md <https://github.com/uwplasma/GKX/blob/plan/focused-research-20260906/plan.md>`_;
:doc:`research_grade_program` records the technical decisions and sources.
Older checklists are historical, not additional active roadmaps.

.. list-table:: Claim boundaries
   :header-rows: 1
   :widths: 30 70

   * - Feature
     - Current scope
   * - Nonlinear derivatives
     - Implemented for a detached-state finite RK window with checkpointing.
       This does not establish the derivative of long-time mean transport.
   * - QA transport reduction
     - **Not statistically resolved.** The nominal campaign has 4/48 traces
       failing final drift; the previous claim that all gates passed was wrong.
   * - Linear and quasilinear
     - Eigenmode sensitivities and ranking tools exist. Residual/mode-selection
       checks and reference provenance are required; QL is not an absolute-flux model.
   * - Coulomb collisions
     - Drift-kinetic low-moment and finite-wavelength tables have different
       validation status. Finite-wavelength repairs remain under review; no
       general unlike-species/arbitrary-moment claim.
   * - CPU/GPU parallelism
     - Independent scans/ensembles are the qualified path. Species–Hermite
       execution exists with restrictions; full nonlinear scaling/AD is not
       generally established.
   * - Electromagnetic and coil fields
     - Equations and selected tests/interfaces exist. Broad finite-beta
       nonlinear transport is unqualified and its core qualification is
       **required**, not deferred. Island-containing coil-field turbulence
       requires a separate model/locality program.

Required targets are verified ES **and three-field electromagnetic** local
gyrokinetics, including kinetic-electron tokamak and finite-beta stellarator
cases. ES QA optimization is an early application, not the full code milestone.
The plan's EM0–EM5 sequence covers independent field residuals, wave/KBM checks,
nonlinear channel-resolved transport, stellarator geometry, AD and execution.
An ES-only result cannot close the research-grade release.
See :doc:`stellarator_optimization` for the retained initial/final conditions,
failed gates and reproduction commands.

Required evidence
-----------------

- Mathematics: independently constructed identities, invariants and derivatives.
- Numerics: time, domain, spectral/velocity and closure convergence.
- Physics: literature-anchored cases and matched independent-code observables.
- Statistics: stationary individual traces, correlation-corrected uncertainty
  and genuinely independent validation seeds.
- Performance: compilation, spin-up, value/VJP, memory and uncertainty at matched
  accuracy on the actual CPU/GPU workload.
- Usability: working examples and explicit supported/unsupported contracts.

No new release is scheduled. Review open PR dispositions and the essential
admitted goals before tagging or publishing.
