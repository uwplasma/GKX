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
:download:`examples/10_vmex_optimization/run.py <../examples/10_vmex_optimization/run.py>` (formerly ``QA_optimization.py``).
It follows VMEX's QA mode ladder and adds one objective tuple:

.. code-block:: python

   objective_function_terms = [
       (qs, 0.0, QA_PRIORITY),
       (opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY),
       (opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY),
       (turbulent_transport, 0.0, transport_weight),
   ]

Here ``ASPECT_TARGET = 6.0`` and ``IOTA_TARGET = 0.42``. The VMEC equilibrium
is vacuum (``AM=0`` and ``PRES_SCALE=0``). GKX retains finite ITG drive with
:math:`a/L_T=3` and :math:`a/L_n=1`.

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
iota, and transport priorities are visible beside
the physics and resolution constants.

Run
---

.. code-block:: bash

   pip install -e /path/to/VMEX
   python examples/10_vmex_optimization/run.py

Use a GPU for the default saturation and 1024-step differentiated windows. ``WIRING_RUN = True`` in the script runs a short
CPU case that checks the complete solver/derivative path.

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

Independent transport validation
--------------------------------

The differentiated window selects a local boundary direction. Evaluate the
initial and candidate boundaries with matched equilibrium tolerances, flux
tubes, species, grids and timestep policies. Use independent saturated runs
and paired seeds for transport acceptance. Save energy and resolved
field/flux spectra as well as heat-flux traces so stationarity and spectral
convergence can be tested.

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

The reported interval is conservative,

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

Check every trace independently before reporting an ensemble interval.
Opposing signed drifts cannot cancel a stationarity failure. Repeat the
comparison at finer perpendicular, parallel and velocity-space resolutions
and smaller timesteps. A conditional confidence interval at one resolution
does not establish converged transport reduction.

VMEX turbulence examples
------------------------

VMEX ships three QA boundary optimizations that call GKX, in
``examples/optimization/`` of the VMEX repository (``pip install
'vmex[turbulence]'``). Each adds one GKX objective to VMEX's
``QA_optimization.py`` on flux tubes chosen by ``s`` and ``alpha``.

- ``QA_optimization_turbulence_linear.py``: growth rate of the dominant linear
  ITG eigenmode, with VMEX's implicit Jacobian through ``jnp.linalg.eigvals``.
  At ``s = 0.5`` the growth rate went 0.196 to 0.097 (:math:`v_{ti}/a`), the
  aspect ratio 11.5 to 6.4, and the quasisymmetry error rose 0.039 to 0.20.
  28 min, 6.5 GB peak.
- ``QA_optimization_turbulence_quasilinear.py``: mixing-length heat flux
  :math:`\gamma W_Q/\langle k_\perp^2\rangle`, which needs eigenvector
  derivatives. The flux went 2.57 to 0.94, aspect ratio 11.5 to 7.7,
  quasisymmetry error unchanged at 0.04; mean :math:`|\iota|` fell to 0.12,
  below the 0.42 floor these weights do not enforce. 29 min, 6.9 GB peak.
- ``QA_optimization_turbulence_nonlinear.py``: gated saturated heat flux over
  a finite window, differentiated with the checkpointed discrete adjoint from a
  fixed saturated start and passed to L-BFGS-B as one scalar. The flux went
  53.0 +- 2.6 to 36.3 +- 1.8 after stage 1 and 42.6 +- 2.1 after stage 2, which
  traded flux for quasisymmetry (error 2.29 to 0.75); aspect ratio 11.5 to 6.0.
  70 min, 5.5 GB peak.

Each number is a single run with default settings on a shared 36-core CPU host
(12 cores, JAX 0.10.2, GKX 2.4.0, other jobs running). The nonlinear result is
one tube and one seed with no held-out check: it shows the pipeline works, not
a certified transport reduction.

Running and assessing the workflow
----------------------------------

Run the checked-in optimization example from the repository root:

.. code-block:: bash

   python examples/10_vmex_optimization/run.py

Write generated equilibria, restart states and diagnostics to a separate output
directory. Record the input configuration and source revision with the result;
restart states and analysis windows must use the same operator conventions.

The derivative mathematics and limitations are in :doc:`nonlinear_autodiff`.
Reduced linear and quasilinear objectives provide screening diagnostics;
validate proposed improvements with independent saturated nonlinear runs.
