Stellarator optimization
========================

GKX composes with VMEX without intermediate geometry files:

.. math::

   x_{boundary}\rightarrow u_{VMEC}(x)\rightarrow
   \mathcal G(u)\rightarrow \langle Q_{GKX}\rangle.

VMEX differentiates its converged three-dimensional equilibrium
[HirshmanWhitson83]_ implicitly. GKX then differentiates the nonlinear
Runge--Kutta window with its checkpointed discrete adjoint. The chain rule is
handled by JAX.

QA nonlinear transport
----------------------

The executable example targets quasi-axisymmetry [LandremanPaul22]_ and follows
the nonlinear-transport optimization pattern of [Kim24]_. It is
:download:`QA_optimization.py <../examples/optimization/QA_optimization.py>`.
It follows VMEX's QA mode ladder and adds one objective tuple:

.. code-block:: python

   objective_function_terms = [
       (qs, 0.0, 1.0),
       (opt.aspect_ratio, 6.0, 1.0),
       (opt.mean_iota, 0.42, 10.0),
       (turbulent_transport, 0.0, transport_weight),
   ]

The VMEC equilibrium is vacuum (``AM=0`` and ``PRES_SCALE=0``). GKX retains
finite ITG drive with :math:`a/L_T=2.49` and :math:`a/L_n=0.8`.

Algorithm
---------

For each boundary-mode stage:

#. solve the VMEX equilibrium;
#. run GKX to a saturated state on the selected flux tube;
#. minimize QA, aspect, iota, and the physical heat-flux window using analytic
   VMEX and GKX derivatives; VMEX's ``auto`` policy selects the lower-memory
   block-factorized Jacobian for this four-residual problem;
#. refresh the saturated state after accepting the stage.

The transport residual is normalized by the seed flux,

.. math::

   w_Q = \frac{w_{priority}}{Q_{seed}^2},

so its priority is not set accidentally by diagnostic units. The default
``TRANSPORT_PRIORITY=2`` is deliberately visible beside the physics and
resolution constants.

Run
---

.. code-block:: bash

   pip install -e /path/to/VMEX
   python examples/optimization/QA_optimization.py

Use a GPU for the default saturation and 1024-step differentiated windows. A short
``VMEX_EXAMPLES_CI=1`` mode checks the complete solver/derivative path.

Outputs are the optimized VMEC input and WOUT plus standard VMEX equilibrium
plots. SciPy prints the objective history during the run.

Controls
--------

``A_OVER_LT``, ``A_OVER_LN``
   GKX gradient drives. These are :math:`a/L`, not :math:`R/L`.

``SATURATION_STEPS``
   Spin-up length. Increase until the heat-flux running mean is stationary.

``WINDOW_STEPS``
   Differentiated window. Use several autocorrelation times but remain below
   the measured tangent-divergence knee.

``TRANSPORT_PRIORITY``
   Relative least-squares cost assigned to the seed-normalized heat flux.

``NX, NY, NZ, NL, NM``
   Perpendicular, parallel, Laguerre, and Hermite resolution.

Validation policy
-----------------

An optimizer post-saturation window supplies a design direction, not the final
physics claim.
Promote a result only after matched baseline/candidate runs pass:

* saturation and running-mean convergence;
* independent random-seed and timestep repeats;
* resolved heat-flux reduction with uncertainty;
* aspect, iota, and quasisymmetry constraints on the solved WOUT.

The derivative mathematics, memory profile, and limitations are in
:doc:`nonlinear_autodiff`. Reduced linear and quasilinear objectives remain
useful screening diagnostics, but they are development diagnostics only and
are not substitutes for the saturated nonlinear objective.
