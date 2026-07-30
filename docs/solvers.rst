Solvers
=======

Time integration
----------------

The linear solver supports explicit Euler, RK2, and RK4 updates inside a JAX
``scan`` loop.  RK4 is used in the Cyclone harness.

Two options address stiff Hermite--Laguerre systems:

* ``method="imex"`` treats collision and hyper-diffusion damping implicitly
  while keeping drift and streaming terms explicit.
* ``method="implicit"`` applies backward Euler through SOLVAX FGMRES.  Its
  physical preconditioners include diagonal damping and an FFT/Hermite-line
  streaming approximation.  Linked chains retain the twist-and-shift layout.

Complex FGMRES uses unitary Givens rotations and reports the physical residual
``||b - A x||``.  Users set its tolerance, restart length, iteration limit,
and preconditioner; a failed residual gate is never converted into a finite
success value.

Differentiable eigenmodes
-------------------------

``adaptive_propagator_eigenpair`` is the opt-in matrix-free path for linear and
quasilinear objectives.  It:

* estimates a stable RK4 step from broadband spectral probes;
* extracts several leading-growth candidates from a long-horizon propagator;
* certifies every pair against the original continuous operator;
* tracks a requested branch with right or biorthogonal overlap; and
* uses an implicit reverse rule for eigenvalue and eigenvector observables.

It avoids the ``O(n^2)`` dense matrix, but the explicit filter is still slow at
high Hermite resolution.  The qualification, QI full-frequency result, cold
timings, and remaining preconditioner gate are documented in
:doc:`differentiable_eigensolver`.

Optional damping
----------------

The linear operator supports:

* Lenard--Bernstein diagonal damping with rate ``nu``;
* velocity-space hyper-collisions controlled by ``nu_hyper_l``,
  ``nu_hyper_m``, ``nu_hyper_lm`` and their exponents; and
* smooth field-aligned end damping controlled by ``damp_ends_widthfrac`` and
  ``damp_ends_amp``.

All may be disabled or refined independently in resolution studies.

Performance caching
-------------------

``LinearCache`` stores geometry-dependent gyroaverages, drift coefficients,
linked-chain maps, and masks outside the time loop.  ``build_linear_cache``
constructs it, while the runtime integration entry points build and reuse it
automatically.

Growth-rate extraction
----------------------

For

.. math::

   \phi(t) \approx \exp[(\gamma - i\omega)t],

GKX fits :math:`\log|\phi|` and unwrapped phase against time.
``fit_growth_rate_auto`` scans for the most exponential window and is used by
the Cyclone harness when ``auto_window=True``.
