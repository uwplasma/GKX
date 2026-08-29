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
  inverse of the additive diagonal-plus-streaming symbol.  Linked chains
  retain the twist-and-shift layout.

The same line inverse is available to shift-invert Arnoldi.  The default
``shift_preconditioner="auto"`` uses it for electrostatic models, selects the
Woodbury field correction directly for electromagnetic models, and retries a
residual-rejected electrostatic pair with field correction.  Explicit
``"hermite-line"`` and ``"field-corrected"`` choices remain available.

Complex FGMRES uses unitary Givens rotations and reports the physical residual
``||b - A x||``.  Users set its tolerance, restart length, iteration limit,
and preconditioner; a failed residual gate is never converted into a finite
success value.

When the matrix-free inner solve still stalls at high velocity resolution,
``method="sparse_shift_invert"`` assembles the sparse operator 64 columns at a
time and factors the complete shifted operator through the optional SciPy
backend.  It requires ``shift`` from a coarse solve or continuation point and
certifies every candidate with the original JAX operator.  The SOLVAX bridge
can reuse the same factors for the adjoint inverse in implicit eigenpair AD.
This path is eager and CPU-factorized; it is not the default for small or
target-free solves.

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
high Hermite resolution.  The collisional QI full-frequency ladder is closed;
the separate collisionless stress case remains unresolved.  Qualification,
cold timings, and the remaining implementation boundary are documented in
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
