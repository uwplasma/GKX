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
       (qs, 0.0, QA_PRIORITY),
       (opt.aspect_ratio, 6.0, ASPECT_PRIORITY),
       (opt.mean_iota, 0.42, IOTA_PRIORITY),
       (turbulent_transport, 0.0, transport_weight),
   ]

The VMEC equilibrium is vacuum (``AM=0`` and ``PRES_SCALE=0``). GKX retains
finite ITG drive with :math:`a/L_T=3` and :math:`a/L_n=1`.

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

so its priority is not set accidentally by diagnostic units. The QA, aspect,
iota, and transport priorities are deliberately visible beside
the physics and resolution constants.

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

Matched validation
------------------

The differentiated window selects a boundary direction; independent forward
runs establish the transport result. The validation candidate uses one
``max_mode=1`` stage from VMEX's optimized QA input. It holds the seed aspect
and iota targets, moves eight boundary coefficients, and uses the same four
residuals and priorities as the example. The search differentiates a 16-step
post-saturation window; all acceptance statistics come from new long runs.
The coefficients are
:download:`available as CSV <_static/qa_transport_boundary_delta.csv>`.

Both boundaries are re-solved as vacuum equilibria with ``Ns=101``. Forward
GKX runs use :math:`s=0.64`, :math:`\alpha=0`,
``(Nx, Ny, Nz, Nl, Nm)=(16,16,24,4,8)``, ``Lx=Ly=62.8``, and
:math:`\Delta t=0.05`. Each matched pair shares one random multimode seed,
runs to :math:`t=1500`, and averages :math:`1100\leq t\leq1500`.

The positive-window integrated autocorrelation time [Sokal97]_ gives

.. math::

   n_{eff}=\min\!\left(n,\frac{n\,\Delta t_s}{2\tau_{int}}\right),\qquad
   r_i=100\left(1-\frac{\bar Q_{c,i}}{\bar Q_{b,i}}\right).

The reported standard error is the larger of the paired-seed scatter and the
propagated autocorrelation-corrected trace errors. A Student-:math:`t`
interval uses the number of independent seed pairs. Timestep checks use
:math:`\Delta t=0.05,0.04,0.025`; resolution checks use
``Nx=Ny=12,16,20``, ``Nz=16,24,32``, and
``(Nl,Nm)=(3,6),(4,8),(6,12)``. The Hermite hypercollision exponent follows
GKX's resolution-aware ``min(20, Nm//2)`` policy.

The derivative mathematics, memory profile, and limitations are in
:doc:`nonlinear_autodiff`. Reduced linear and quasilinear objectives remain
screening diagnostics, not substitutes for the saturated nonlinear objective.
