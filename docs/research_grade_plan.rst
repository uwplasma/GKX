Research readiness
==================

GKX distinguishes implemented algorithms from validated scientific claims.
The :download:`execution roadmap <../plan.md>` gives the ordered work;
this page summarizes the acceptance boundary at ``a99dac89`` (2026-09-04).

.. warning::

   GKX 2.0.0 has an open end-damping compatibility regression
   (`issue 192 <https://github.com/uwplasma/GKX/issues/192>`_).
   The proposed repair (`PR 197 <https://github.com/uwplasma/GKX/pull/197>`_)
   was not merged at this audit. Affected benchmark claims require a repaired
   operator, explicit damping units and regenerated evidence.

.. list-table:: Capability and evidence still required
   :header-rows: 1
   :widths: 22 30 48

   * - Capability
     - Current boundary
     - Acceptance
   * - Nonlinear derivatives
     - Checkpointed AD of a fixed post-spin-up window
     - Same-map value/gradient parity, Taylor tests, CPU/GPU checks and useful
       direction across independent saturated states
   * - QA transport optimization
     - Preliminary campaign; not promotion-ready
     - Per-trace stationarity, correlated uncertainty, new seeds, timestep,
       box/spatial/moment/closure convergence and equilibrium constraints
   * - Parallelism
     - Independent work; diagnostic single-trajectory decompositions
     - Actual distributed primal/VJP, fields/flux/conservation checks and
       matched CPU/GPU performance; metadata identity alone is insufficient
   * - Collisions and EM
     - Model- and resolution-specific support
     - Correct entropy/invariant identities, supported species/moment ranges,
       kinetic-electron and electromagnetic benchmark closure
   * - Research release
     - Runnable product and scoped historical evidence
     - Current physics gates, clean-wheel examples, reproducible data and
       explicit unsupported regimes; CI coverage alone is insufficient

What the nonlinear derivative means
-----------------------------------

.. math::

   J_N(x;G_0)=\frac{1}{N_w}\sum_{n\in W}Q(G_n,x),\qquad
   G_{n+1}=\Phi_{\Delta t}(G_n,x),\qquad
   G_0=\operatorname{stop\_gradient}(G_{sat}).

The derivative is exact for this declared discrete window, within numerical
accuracy. It excludes the derivative of spin-up and adaptive stopping. It is
not automatically the derivative of infinite-time mean turbulent transport.
Block checkpointing reduces retained state storage to :math:`O(\sqrt N)`;
caches, temporaries and requested outputs have additional costs.
See :doc:`nonlinear_autodiff` for implementation and measurements.

The retained QA results do **not** pass all transport gates: some traces fail
stationarity and the compact data cannot establish spectral convergence.
:doc:`stellarator_optimization` preserves the measurements and rejected cases.

Tests should falsify the implementation: analytic identities, independently
manufactured forcing, numerical convergence, literature physics, matched-code
observables, derivative checks and regression/performance limits. A passing
small test establishes only its tested model, precision and resolution.
