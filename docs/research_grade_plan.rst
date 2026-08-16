Research-grade checklist
========================

GKX separates algorithm verification from physics claims. A feature is ready
only when its numerical result, physical observable, and performance envelope
are all tested.

Nonlinear derivatives
---------------------

Status: **implemented**. :func:`gkx.nonlinear_heat_flux_window` differentiates
the physical post-saturation heat flux through the exact projected Runge--Kutta
map. Block checkpointing retains :math:`O(\sqrt N)` states.

Required gates:

* blocked and plain values and derivatives agree;
* automatic differentiation agrees with centered finite differences below the
  measured trajectory-divergence knee;
* CPU and GPU results agree within the selected precision;
* the optimizer uses the physical heat flux, not a state norm or linear
  saturation rule.

See :doc:`nonlinear_autodiff` for equations, measurements, and usage.

Transport claims
----------------

An optimizer window gives a local design direction. A reported transport
reduction additionally requires:

* stationary post-transient running means;
* correlation-corrected uncertainty using an effective sample count;
* independent seeds and timestep repeats;
* perpendicular, parallel, Laguerre, and Hermite convergence;
* matched baseline and candidate equilibria that satisfy aspect, iota, and
  quasisymmetry constraints.

The selected vacuum QA direction passes these gates. Its scope, including the
rejected short-horizon control, is in :doc:`stellarator_optimization`.

Open numerical work
-------------------

Electromagnetic parity
   Close the KBM benchmark discrepancy against an independent code.

Velocity-space recurrence
   Report :math:`t_{rec}` beside each nonlinear averaging interval and verify
   closure convergence.

Precision
   Keep single precision as the fast default; test ensemble observables in
   double precision where cancellation or long integrations demand it.

Performance
   Measure Hermite--Laguerre block preconditioners, mixed-precision iterative
   refinement, and distributed windows on production CPU and GPU cases. Adopt
   a method only when wall time and peak memory improve without changing the
   validated observable.

Testing rule
------------

Each test must name a result that could falsify the implementation:

* mathematics: identities, manufactured maps, and derivative comparisons;
* numerics: order, convergence, conditioning, and conservation;
* physics: literature benchmarks and independent-code parity;
* regression: stable public outputs and performance budgets.
