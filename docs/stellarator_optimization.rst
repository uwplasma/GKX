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

Mathematical model
------------------

The optimized boundary uses VMEC's stellarator-symmetric Fourier map,

.. math::

   R(\theta,\zeta)=\sum_{m,n}R_{mn}\cos(m\theta-nN_{fp}\zeta),\qquad
   Z(\theta,\zeta)=\sum_{m,n}Z_{mn}\sin(m\theta-nN_{fp}\zeta).

VMEX solves :math:`F(u,x)=0` for the equilibrium state :math:`u` and boundary
coefficients :math:`x`. Its implicit derivative is

.. math::

   \frac{du}{dx}=-F_u^{-1}F_x.

For QA helicity :math:`(m,n)=(1,0)`, the optimized surface norm is built from
the pointwise residual

.. math::

   f_{QS}=\frac{(\mathbf B\!\times\!\nabla B\cdot\nabla\psi)
      (nN_{fp}-\iota m)
      -(\mathbf B\cdot\nabla B)(mG+nN_{fp}I)}{B^3},

where :math:`G` and :math:`I` are the Boozer covariant field averages.
:math:`f_{QS}=0` when :math:`|B|` depends only on
:math:`m\theta-nN_{fp}\zeta`.

GKX expands the nonadiabatic distribution in Laguerre--Hermite moments,

.. math::

   g_s=\sum_{\ell=0}^{N_l-1}\sum_{j=0}^{N_m-1}
       G_{s\ell j}\,\psi_\ell(\mu B)\,\phi_j(v_\parallel),

and advances the projected nonlinear system

.. math::

   \dot{\mathbf G}=\mathcal L(\mathcal G,p)\mathbf G
      +\mathcal N\!\left(\mathbf G,\mathcal F(\mathbf G;\mathcal G)\right)
      +\mathcal C\mathbf G,
   \qquad \mathbf G\leftarrow\mathcal P\mathbf G.

:math:`\mathcal G` is the VMEX flux-tube geometry, :math:`\mathcal F` is the
field solve, and :math:`\mathcal P` restores spectral Hermitian symmetry after
each Runge--Kutta step. The campaign is electrostatic. Its physical diagnostic
is GKX's spectral ion heat flux,

.. math::

   Q_i=2n_iT_i\sum_{k_x,k_y,z}w_k w_z\,
       \Re\!\left[\overline{i k_y\phi}\,\bar p_i\right],\qquad
   \bar p_i=\sum_\ell\left(
      \mathcal J^{fac}_\ell G_{i\ell0}
      +\frac{\mathcal J_\ell}{\sqrt2}G_{i\ell2}\right).

The public objective is the discrete post-saturation mean

.. math::

   J(x)=\frac1{N_w}\sum_{k=N-N_w+1}^{N}Q_i(G_k,x),\qquad
   G_{k+1}=\Phi_{\Delta t}(G_k;x),\qquad
   G_0=\operatorname{stop\_gradient}(G_{sat}).

The least-squares residual is

.. math::

   \mathbf r=\left(
      \sqrt{w_{QS}}\,\mathbf f_{QS},\;
      \sqrt{w_A}(A-6),\;
      \sqrt{w_\iota}(\bar\iota-0.42),\;
      \sqrt{w_T}\,J/Q_{seed}\right),
   \qquad \min_x\tfrac12\|\mathbf r\|_2^2,

with :math:`(w_{QS},w_A,w_\iota,w_T)=(10^3,10^3,10^5,20)`.

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

.. figure:: _static/qa_transport_equilibria.png
   :width: 900px
   :alt: initial and optimized QA boundaries and Boozer field strength on the LCFS

   Initial and optimized vacuum equilibria. The 3-D LCFS and Boozer maps use
   one :math:`|B|/\langle|B|\rangle` scale. Eight low-order boundary
   coefficients move; aspect and mean iota remain within 0.05%.

Forward GKX runs use :math:`s=0.64`, :math:`\alpha=0`,
``(Nx, Ny, Nz, Nl, Nm)=(16,16,24,4,8)``, ``Lx=Ly=62.8``, and
:math:`\Delta t=0.05`. Each pair shares one fresh random Hermitian multimode
state; seeds are independent between pairs. Nominal traces run to
:math:`t=1500`, are sampled every :math:`\Delta t_s=1`, and average
:math:`1100\leq t\leq1500`.

Uncertainty estimator
---------------------

For design :math:`d\in\{b,c\}`, seed :math:`i`, and :math:`n` samples,

.. math::

   \bar Q_{d,i}=\frac1n\sum_jQ_{d,i,j},\qquad
   \rho_{d,i}(k)=\frac{\operatorname{cov}(Q_j,Q_{j+k})}
                       {\operatorname{var}(Q_j)}.

GKX integrates :math:`\rho` with the trapezoid rule up to its first negative
lag [Sokal97]_. This positive-window estimator gives

.. math::

   \tau_{d,i}=\int_0^{t_{K_0}}\rho_{d,i}(t)\,dt,\qquad
   n_{eff,d,i}=\min\!\left(n,\frac{n\Delta t_s}{2\tau_{d,i}}\right),\qquad
   \sigma_{d,i}=\frac{s_{d,i}}{\sqrt{n_{eff,d,i}}}.

Matched seeds remove much of the initial-condition variance:

.. math::

   r_i=100\left(1-\frac{\bar Q_{c,i}}{\bar Q_{b,i}}\right),\qquad
   \bar r=\frac1M\sum_i r_i.

Two errors are evaluated. The first is the scatter of the :math:`M`
independent seed-pair reductions. The second propagates the two correlated
trace means:

.. math::

   \sigma_{pair}=\frac{s_r}{\sqrt M},\qquad
   \sigma_{IAT}=\frac1M\left[\sum_i100^2\left\{
      \left(\frac{\bar Q_{c,i}\sigma_{b,i}}{\bar Q_{b,i}^2}\right)^2
      +\left(\frac{\sigma_{c,i}}{\bar Q_{b,i}}\right)^2
      \right\}\right]^{1/2}.

The reported interval is deliberately conservative,

.. math::

   \sigma=\max(\sigma_{pair},\sigma_{IAT}),\qquad
   \mathrm{CI}_{95}=\bar r\pm t_{0.975,M-1}\sigma.

Stationarity is checked on every trace with the half-window shift and linear
trend,

.. math::

   H=100\frac{\bar Q_2-\bar Q_1}{\bar Q},\qquad
   S=100\frac{\hat\beta(t_{max}-t_{min})}{\bar Q},

where :math:`\hat\beta` is the least-squares slope. The final-drift test is
:math:`|S|\leq20\%`, following [Oberparleiter16]_. The gate is conjunctive over
traces; signed drifts may not cancel across seeds. Heat flux is in gyro-Bohm
units.

.. list-table:: Matched post-saturation transport
   :header-rows: 1

   * - check
     - pairs
     - :math:`\bar Q_b`
     - :math:`\bar Q_c`
     - reduction [95% CI]
     - stationary traces
   * - nominal
     - 24
     - 11.16
     - 9.78
     - 12.26% [10.64, 13.88]
     - 44/48
   * - :math:`\Delta t=0.04`
     - 8
     - 11.27
     - 9.58
     - 14.79% [10.52, 19.05]
     - 13/16
   * - :math:`\Delta t=0.025`
     - 4
     - 10.76
     - 9.40
     - 12.57% [6.51, 18.64]
     - 7/8
   * - :math:`N_x=N_y=12` (coarse)
     - 4
     - 18.73
     - 14.38
     - 23.13% [14.22, 32.04]
     - 7/8
   * - :math:`N_x=N_y=20`
     - 16
     - 10.05
     - 9.53
     - 4.95% [2.12, 7.78]
     - 27/32
   * - :math:`N_x=N_y=24` (short, rejected)
     - 4
     - 9.27
     - 9.24
     - 0.04% [-14.87, 14.96]
     - 6/8
   * - :math:`N_x=N_y=24` (long)
     - 16
     - 9.92
     - 9.07
     - 8.50% [6.34, 10.66]
     - 30/32
   * - :math:`N_z=16`
     - 4
     - 10.74
     - 9.76
     - 9.06% [2.37, 15.74]
     - 6/8
   * - :math:`N_z=32`
     - 4
     - 11.60
     - 10.15
     - 12.51% [5.76, 19.25]
     - 8/8
   * - :math:`(N_l,N_m)=(3,6)` (coarse)
     - 4
     - 13.78
     - 11.33
     - 17.12% [0.42, 33.81]
     - 6/8
   * - :math:`(N_l,N_m)=(6,12)`
     - 16
     - 10.93
     - 9.56
     - 12.32% [9.62, 15.03]
     - 30/32

The intervals above are conditional summaries, not validation intervals.
The old gate averaged signed half-window shifts across seeds; opposing drifts
cancelled. Applying the published final-drift test independently rejects 4 of
48 nominal traces. Every case except :math:`N_z=32` has at least one drift
failure, including 2 of 32 long-24x24 and 2 of 32
:math:`(N_l,N_m)=(6,12)` traces.

The compact raw files contain only :math:`Q_i(t)`, so neither the increasing
high-:math:`k_y` tail nor spectral convergence can be tested. Thus none of the
104 matched pairs is promotion-ready. The numbers remain useful preliminary
evidence and a reproducible cost baseline (15.32 measured GPU integration
hours on one RTX A4000).

.. figure:: _static/qa_transport_reduction.svg
   :width: 900px
   :alt: matched QA heat-flux traces and transport-reduction convergence

   Initial and optimized nominal ensemble means with seed SEM; the shaded
   interval is the measured window. Error bars are autocorrelation-corrected
   conditional summaries. Per-trace stationarity and spectral gates remain
   open.

Reproduce
---------

The checked-in workflow has three owners:

* :download:`QA_optimization.py <../examples/optimization/QA_optimization.py>`
  computes the VMEX--GKX design step;
* :download:`qa_transport_validation.py
  <../tools/campaigns/qa_transport_validation.py>` runs restartable matched
  saturation pairs;
* :download:`build_qa_transport_figures.py
  <../tools/artifacts/build_qa_transport_figures.py>` computes
  :math:`\tau_{int}`, effective sample counts, intervals, CSV tables, and plots.

The exact accepted boundaries are
:download:`initial <../examples/optimization/input.qa_transport_baseline>` and
:download:`optimized <../examples/optimization/input.qa_transport_candidate>`.
For example,

.. code-block:: bash

   python tools/campaigns/qa_transport_validation.py nominal \
     --seed-stop 24 --output-dir campaign/transport
   python tools/campaigns/qa_transport_validation.py perp24long \
     --seed-stop 16 --output-dir campaign/transport
   python tools/campaigns/qa_transport_validation.py v612 \
     --seed-stop 16 --output-dir campaign/transport
   python tools/artifacts/build_qa_transport_figures.py \
     --raw-dir campaign/transport --output-dir docs/_static

Use the case names and pair counts in the table for the timestep, perpendicular,
parallel, and velocity-space scans. Each ``.npz`` stores only ``time``,
``heat_flux``, and elapsed seconds. The 208 raw traces stay outside git; their
per-trace summary statistics and the plotted nominal mean/SEM are committed.
Future promotion runs must also retain content hashes plus :math:`W_\phi(t)`,
:math:`W_g(t)`, and resolved flux/field spectra.

Download the :download:`case summary <_static/qa_transport_summary.csv>`,
:download:`per-trace statistics <_static/qa_transport_traces.csv>`, or
:download:`nominal time series <_static/qa_transport_nominal_timeseries.csv>`,
or :download:`boundary displacement <_static/qa_transport_boundary_delta.csv>`.
The Hermite hypercollision exponent follows the resolution-aware
``min(20, Nm//2)`` policy.

The derivative mathematics, memory profile, and limitations are in
:doc:`nonlinear_autodiff`. Reduced linear and quasilinear objectives remain
screening diagnostics, not substitutes for the saturated nonlinear objective.
