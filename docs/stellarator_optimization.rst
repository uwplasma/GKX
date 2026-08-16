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
post-saturation window; centered finite-difference checks validate that local
derivative, while all acceptance statistics come from new long runs.
The coefficients are
:download:`available as CSV <_static/qa_transport_boundary_delta.csv>`.

Both boundaries are re-solved as vacuum equilibria with ``Ns=101``,
``ftol=1e-10``, and ``niter=15000``:

.. list-table::
   :header-rows: 1

   * - quantity
     - baseline
     - candidate
   * - aspect ratio
     - 5.006406
     - 5.006984
   * - mean iota
     - 0.419017
     - 0.418831
   * - QA residual
     - :math:`5.88\times10^{-4}`
     - :math:`1.54\times10^{-3}`

Forward GKX runs use :math:`s=0.64`, :math:`\alpha=0`,
``(Nx, Ny, Nz, Nl, Nm)=(16,16,24,4,8)``, ``Lx=Ly=62.8``, and
:math:`\Delta t=0.05`. Each pair shares one fresh random Hermitian multimode
state; seeds are independent between pairs. Nominal traces run to
:math:`t=1500`, are sampled every :math:`\Delta t_s=1`, and average
:math:`1100\leq t\leq1500`.

The positive-window integrated autocorrelation time [Sokal97]_ gives

.. math::

   n_{eff}=\min\!\left(n,\frac{n\,\Delta t_s}{2\tau_{int}}\right),\qquad
   r_i=100\left(1-\frac{\bar Q_{c,i}}{\bar Q_{b,i}}\right).

The reported standard error is the larger of the paired-seed scatter and the
propagated autocorrelation-corrected trace errors. A Student-:math:`t`
interval uses the number of independent seed pairs. Heat flux is reported in
gyro-Bohm units.

.. list-table:: Matched post-saturation transport
   :header-rows: 1

   * - check
     - pairs
     - :math:`\bar Q_b`
     - :math:`\bar Q_c`
     - reduction [95% CI]
   * - nominal
     - 24
     - 11.16
     - 9.78
     - 12.26% [10.64, 13.88]
   * - :math:`\Delta t=0.04`
     - 8
     - 11.27
     - 9.58
     - 14.79% [10.52, 19.05]
   * - :math:`\Delta t=0.025`
     - 4
     - 10.76
     - 9.40
     - 12.57% [6.51, 18.64]
   * - :math:`N_x=N_y=12` (coarse)
     - 4
     - 18.73
     - 14.38
     - 23.13% [14.22, 32.04]
   * - :math:`N_x=N_y=20`
     - 16
     - 10.05
     - 9.53
     - 4.95% [2.12, 7.78]
   * - :math:`N_x=N_y=24` (short, rejected)
     - 4
     - 9.27
     - 9.24
     - 0.04% [-14.87, 14.96]
   * - :math:`N_x=N_y=24` (long)
     - 16
     - 9.92
     - 9.07
     - 8.50% [6.34, 10.66]
   * - :math:`N_z=16`
     - 4
     - 10.74
     - 9.76
     - 9.06% [2.37, 15.74]
   * - :math:`N_z=32`
     - 4
     - 11.60
     - 10.15
     - 12.51% [5.76, 19.25]
   * - :math:`(N_l,N_m)=(3,6)` (coarse)
     - 4
     - 13.78
     - 11.33
     - 17.12% [0.42, 33.81]
   * - :math:`(N_l,N_m)=(6,12)`
     - 16
     - 10.93
     - 9.56
     - 12.32% [9.62, 15.03]

The first :math:`24\times24` horizon failed its stationarity check: the
baseline half-window shift was :math:`6.94\pm2.43\%` and its interval included
zero. The replacement runs to :math:`t=2500` and averages
:math:`1900\leq t\leq2500`; its half-window shifts are
:math:`1.03\pm1.82\%` and :math:`1.54\pm1.58\%`. Its shortest trace still spans
14.0 autocorrelation times.

The 20x20 and long 24x24 intervals overlap. Their absolute baseline and
candidate means differ by 1.3% and 4.9%, respectively, while both reductions
remain resolved above zero. Timestep and :math:`N_z`-refinement intervals also
remain positive. At :math:`(N_l,N_m)=(6,12)`, all 16 pairs reduce transport,
the interval overlaps nominal, and the absolute means agree with nominal to
2.3%. Its half-window shifts are :math:`0.75\pm1.24\%` and
:math:`-0.92\pm1.72\%`. The coarse 12x12 and :math:`(N_l,N_m)=(3,6)` cases are
controls, not converged estimates. In total, the campaign contains 104 matched
pairs (208 traces) and 15.32 measured GPU integration hours on one RTX A4000.

.. figure:: _static/qa_transport_reduction.svg
   :width: 900px
   :alt: matched QA heat-flux traces and transport-reduction convergence

   Nominal ensemble mean plus seed SEM, followed by paired-seed reductions and
   autocorrelation-corrected 95% intervals. The short 24x24 run is retained as
   a failed-horizon control.

Download the :download:`case summary <_static/qa_transport_summary.csv>`,
:download:`per-trace statistics <_static/qa_transport_traces.csv>`, or
:download:`boundary displacement <_static/qa_transport_boundary_delta.csv>`.
The Hermite hypercollision exponent follows the resolution-aware
``min(20, Nm//2)`` policy.

The derivative mathematics, memory profile, and limitations are in
:doc:`nonlinear_autodiff`. Reduced linear and quasilinear objectives remain
screening diagnostics, not substitutes for the saturated nonlinear objective.
