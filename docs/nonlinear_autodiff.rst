Nonlinear automatic differentiation
===================================

GKX uses one nonlinear derivative: a checkpointed discrete adjoint of a
post-saturation heat-flux window. It is the derivative of the code that runs,
not of a surrogate or a continuous-time approximation.

Objective
---------

Let :math:`G_{k+1}=\Phi_{\Delta t}(G_k;p)` be one projected Runge--Kutta step
and :math:`Q(G_k,p)` the gyro-Bohm heat flux. After a separate saturation run,
GKX differentiates

.. math::

   J(p)=\frac{1}{N_w}\sum_{k=N-N_w+1}^{N}
        Q(G_k,p), \qquad G_0=\operatorname{stop\_gradient}(G_{sat}).

``p`` may enter the species drives, field-line geometry, or both. Reverse mode
applies the discrete recursion

.. math::

   \lambda_k = \left(\frac{\partial\Phi_k}{\partial G_k}\right)^T
               \lambda_{k+1}
               +\frac{1}{N_w}\left(\frac{\partial Q_k}{\partial G_k}\right)^T,

and accumulates :math:`dJ/dp` during the same backward sweep.

Why this method
---------------

An implicit root adjoint is inappropriate: saturated turbulence is a chaotic
trajectory, not a steady root. Long initial-value tangents also grow with the
leading Lyapunov exponent. The finite window gives a useful local design
direction while leaving the long, replicated nonlinear run as an independent
holdout.

Shadowing prototypes were removed from the production API. On the tested GKX
trajectory, multiple-shooting changed the gradient sign and NILSAS inherited a
rapidly growing reduced-system condition number. Neither justified a second
user-facing method.

Memory
------

A plain reverse scan stores :math:`O(N)` distribution states. GKX rematerializes
steps in blocks of length :math:`B` and retains block boundaries:

.. math::

   M(B)=O(N/B+B),\qquad B=\lceil\sqrt N\rceil,
   \qquad M=O(\sqrt N).

At 2048 steps, measured XLA temporary memory fell from 759 MB to 12.6 MB on CPU
and from 11.88 GB to 168 MB on an RTX A4000. Runtime rose by 1.54x and 1.67x,
respectively. The blocked and plain values and gradients agree.

Spectral zero mode
------------------

The :math:`k_\perp=0` mode is differentiated analytically. Near :math:`b=0`,
GKX evaluates :math:`\mathcal J_{\ell+1}=-\mathcal J_\ell b/[2(\ell+1)]`, so
:math:`d\mathcal J_1/db=-1/2` is retained even though
:math:`\mathcal J_1(0)=0`. The collision correction contracts its two
:math:`\sqrt b` factors as :math:`b`; this removes a removable ``sqrt(0)``
tangent singularity without changing the operator.

.. figure:: _static/nonlinear_autodiff_validation.png
   :width: 760px
   :alt: nonlinear adjoint memory and finite-difference validation

   Measured checkpoint memory and physical heat-flux AD/FD agreement. The
   gradient knee between 1024 and 2048 steps bounds the useful initial-value
   window for this Cyclone case.

Python use
----------

Run to saturation once, then differentiate the physical window:

.. code-block:: python

   saturated = gkx.integrate_nonlinear(
       initial, grid, geometry(theta0), params, dt, saturation_steps,
       method="rk3", terms=terms, return_fields=False,
   )

   def loss(theta):
       return gkx.nonlinear_heat_flux_window(
           saturated, grid, geometry(theta), params, dt, window_steps,
           method="rk3", terms=terms,
       )

   heat_flux, gradient = jax.value_and_grad(loss)(theta0)

The saturation state is fixed inside ``loss`` by design. For continuation,
refresh it after an accepted geometry step. Choose a window of several measured
heat-flux autocorrelation times, check the AD direction with a local line
search, and verify the final design with independent saturated runs.

Evidence and scope
------------------

The physical Cyclone heat-flux gradient matches centered finite differences
through 2048 RK3 steps; the worst relative discrepancy is
:math:`2.5\times10^{-5}`. A VMEX--Boozer--GKX state-control test matches finite
differences to :math:`5.7\times10^{-6}` and lowers the local heat flux by 1.25%.

These results establish a finite-window derivative and local descent. They do
not establish an infinite-time turbulent derivative. Independent matched runs
of the accepted QA equilibrium establish a finite-time transport reduction:
the nominal 24-pair interval is 10.64--13.88%, and the 20x20 and stationary
24x24 refinement intervals overlap above zero. The protocol and scope are in
:doc:`stellarator_optimization`.

Tests
-----

* mathematics: blocked and plain discrete adjoints are identical;
* numerics: Runge--Kutta order and AD/centered-FD agreement;
* physics: the differentiated diagnostic is GKX's physical total heat flux;
* optimization: the QA example uses the analytic VMEX and GKX derivatives.

References
----------

* Griewank & Walther, `Revolve <https://doi.org/10.1145/347837.347846>`_,
  *ACM TOMS* **26**, 19--45 (2000).
* JAX, `gradient checkpointing
  <https://docs.jax.dev/en/latest/gradient-checkpointing.html>`_.
* Mandell, Dorland & Landreman, `Laguerre--Hermite gyrokinetics
  <https://arxiv.org/abs/1708.04029>`_, *JPP* **84** (2018).
* Mandell et al., `GX <https://arxiv.org/abs/2209.06731>`_ (2022).
* Wang, Hu & Blonigan, `least-squares shadowing
  <https://arxiv.org/abs/1204.0159>`_, *JCP* **267**, 210--224 (2014).
