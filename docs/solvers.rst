Solvers
=======

Time integration
----------------

The linear solver supports explicit Euler, RK2, and RK4 updates inside a JAX
``scan`` loop, enabling JIT compilation and differentiability of the entire time
history. The time integrator lives in ``gkx.solvers.linear.integrators.integrate_linear`` and
is configured via the ``method`` argument. RK4 is used in the Cyclone harness.

To stabilize higher-order Hermite-Laguerre scans, two additional options are
available:

- ``method="imex"``: a semi-implicit update that treats Lenard-Bernstein and
  hyper-diffusion damping implicitly while keeping the drift/streaming terms
  explicit.
- ``method="implicit"``: a backward-Euler solve using a matrix-free GMRES
  iteration provided by SOLVAX. This is slower but robust for stiff linear runs.
  The same shared policy is used by nonlinear IMEX time steps. A few stabilized
  fixed-point sub-iterations provide an initial guess and a diagonal
  preconditioner based on the damping terms and drift/mirror diagonals
  (cv/gb/bgrad) accelerates convergence for higher-order scans. For
  streaming-dominated stiffness, ``implicit_preconditioner="hermite-line"``
  applies a Hermite streaming line solve (tridiagonal in ``m`` at fixed
  :math:`k_z`) and can substantially reduce GMRES iterations. The reusable
  tridiagonal algebra is provided by SOLVAX; GKX constructs the
  gyrokinetic coefficients and linked-chain layout. CPU execution uses a
  deterministic Thomas recurrence, while accelerator execution uses the fused
  backend selected by SOLVAX.

Implicit time stepping has one supported complex unitary-Givens FGMRES
algorithm. Users configure its tolerance, restart length, iteration limit,
and physical preconditioner rather than selecting equivalent backend aliases.
Shift-invert and block-rational eigenmode extraction now reuse SOLVAX's
right-preconditioned FGMRES. Its least-squares norm is the physical residual
``||b - (A - sigma I)x||``; this removes the former false-convergence path in
which a small left-preconditioned norm triggered a complete unpreconditioned
retry. Complex Ritz vectors use the mathematically correct ``V @ y``
reconstruction, small Arnoldi directions are treated as numerical breakdown,
and an outer original-operator eigenpair residual remains mandatory.

The ``field-corrected`` shifted preconditioner represents the three
low-moment field maps exactly as a Woodbury correction to the Hermite-line
inverse. On the driven electrostatic unit fixture this reduces the one-apply
shifted residual from ``3.01`` to ``0.143``. On the first circular-tokamak
validation rung (``n=768``), a one-candidate rational solve reaches
``6.7e-13`` original-operator residual and ``1.3e-14`` relative eigenvalue
error in 9.5 seconds. The next rung (``n=1536``) does not pass: with a
512-vector inner restart it takes 75.9 seconds and stops at ``2.1e-6``
eigenpair residual. The rational lane is therefore an explicit experimental
method, not the release default; validated dense eigensolves and time
integration remain the fail-closed paths.

Two bounded full-resolution refinements were also rejected. Re-projecting the
physical operator onto the shift-focused Arnoldi basis returned a non-finite
pair and increased peak resident memory from about ``0.87`` to ``1.18`` GB.
Starting the unpreconditioned inner retry from zero likewise returned a
non-finite pair without reducing the roughly ``133`` second CPU runtime. The
released implementation therefore retains the finite fail-closed baseline;
future work requires a branch-preserving thick restart or field-coupled complex
preconditioner with a lower physical residual, not another tolerance sweep.

A bounded full-operator defect-correction preconditioner lowered the same
full-resolution residual to ``0.330``, but a second explicit restart returned
``0.326`` and therefore added cost without canonical-grid convergence. The
candidate was removed. A future revisit should use retained-subspace
Krylov--Schur with harmonic extraction for this interior mode and must pass the
same physical residual, branch-identity, memory, and runtime gates before it
can replace time integration.

Three subsequent reduced-operator discriminators further narrow that work.
An exact projected Jacobi--Davidson step solved its correction equation to
relative residual ``0.035`` but worsened the physical eigenpair residual from
``0.742`` to ``0.876``. Ordered complex-Schur compression retaining four
vectors selected a damped branch, worsened ``0.755`` to ``0.956``, and cost
about ``61`` seconds per restart. Seeding shift-invert from a short propagator
lowered the residual to ``0.521`` but also selected the wrong damped branch.
No implementation was retained. A release candidate must instead demonstrate
a branch-preserving two-sided or field-coupled low-moment correction on the
physical operator, not only convergence of its inner projected equation.
A bounded A4000 run at ``Nl=8,Nm=24`` likewise reached residual ``0.429`` only
after moving to the wrong high-frequency branch, so the negative result is not
confined to the smallest CPU case.

A two-sided correction did not repair the problem: the left inverse iterations
remained unconverged and the first well-solved projected equation moved to a
different branch with residual ``0.870``. A matrix-free field-coupled
low-moment preconditioner was also rejected because it took ``240`` seconds
versus ``30`` seconds for the damping baseline and returned residual ``0.972``
on the wrong branch. Any future field-coupled candidate must use an explicitly
reduced Schur block; nesting a full-operator solve is neither accurate nor
competitive.

An explicitly assembled field/moment block confirmed that the remaining
problem is the complement, not block construction. The ``1536 x 1536`` complex
coarse matrix assembled in ``0.49`` seconds, factored in ``0.15`` seconds, and
used ``36`` MiB. Coupling it to either a diagonal or full Hermite-line
high-moment smoother nevertheless selected incorrect branches with residuals
``0.936`` and ``0.999``; both complete solves took about ``55`` seconds and
exceeded ``1.3`` GB resident memory. These variants were removed. The release
continues to use validated time integration for KBM, while a future direct
interior spectral transformation must establish branch identity before it can
be considered an acceleration.

The earlier SOLVAX 0.7.3 comparison used a left/right-preconditioning mismatch
and is superseded by the physical-residual FGMRES measurements above. Migration
of the inner algebra is complete; promotion of the eigensolver is still blocked
by high-resolution cost, residual, and branch-identity gates.

Acceptance also follows the requested spectral selection. Growth-selected
searches enforce ``fallback_real_floor``; nearest-shift, overlap, and explicit
shift searches may intentionally select a stable mode and therefore use
finiteness plus the physical outer residual instead. Rejection messages identify
which of the non-finite, growth-floor, or residual criteria failed.

Optional damping
----------------

To stabilize high-order Hermite-Laguerre moments, the linear operator supports
two damping models:

- A Lenard-Bernstein diagonal rate ``nu`` with coefficients
  :math:`\nu(\alpha m + \beta l)`.
- Hyper-collisions in velocity space with coefficients ``nu_hyper_l``,
  ``nu_hyper_m``, and ``nu_hyper_lm`` and exponents ``p_hyper_l``,
  ``p_hyper_m``, ``p_hyper_lm``. These damp the highest Hermite/Laguerre
  indices and are enabled by default for Cyclone-style scans.
- Smooth end-cap damping in the field-aligned direction controlled by
  ``damp_ends_widthfrac`` and ``damp_ends_amp`` to suppress reflections at
  linked boundaries.

Both can be disabled or tuned via ``LinearParams`` for resolution studies.

Performance caching
-------------------

To reduce repeated geometry work inside the time loop, we cache the gyroaverage
coefficients, drift frequency, and zero-mode mask in a ``LinearCache`` object.
The helper ``build_linear_cache`` constructs this cache, and
``integrate_linear`` will build and reuse it automatically.

Growth rate extraction
----------------------

Given a complex mode time series

.. math::

   \phi(t) \approx \exp[(\gamma - i \omega) t],

we estimate :math:`\gamma` and :math:`\omega` by least-squares fits of
:math:`\log|\phi|` and the unwrapped phase versus time. The helper
``fit_growth_rate_auto`` can automatically select a fitting window by scanning
for the most exponential-like segment of the time history, reducing sensitivity
to early transients in ky scans. This method is used in the Cyclone linear
benchmark harness when ``auto_window=True``.
