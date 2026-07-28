Harmonic Krylov-Schur eigensolver
=================================

Status: specification and validation plan. No solver is wired into the objective
until every gate in :ref:`hks-validation` passes against the dense reference on
real GKX operators.

.. _hks-motivation:

Why
---

The linear growth rate is currently obtained from a dense eigendecomposition of
the flux-tube operator: all ``n`` eigenvalues and eigenvectors are computed and
one is kept, with ``n = n_laguerre * n_hermite * ntheta``. Measured on a QA
boundary at ``ntheta = 32``, the dense solve scales as :math:`t \sim n^{2.19}`:

.. list-table::
   :header-rows: 1

   * - :math:`(N_\ell, N_m)`
     - n
     - matrix
     - dense eig
   * - (2, 3)
     - 192
     - 0.6 MB
     - 0.02 s
   * - (4, 6)
     - 768
     - 9.4 MB
     - 0.90 s
   * - (6, 8)
     - 1536
     - 37.7 MB
     - 5.32 s
   * - (8, 10)
     - 2560
     - 104.9 MB
     - 12.23 s

Projected to converged resolution: ``(12,16)`` with ``ntheta = 64`` is 7 min and
2.4 GB per evaluation; ``(32,16)`` -- roughly the published ITG guidance -- is
60 min and 17.2 GB, which exceeds the memory of the GPUs this runs on. A
convergence ladder needs several evaluations per configuration and a campaign
needs a ladder per configuration, so the dense path is what bounds how much
converged science is affordable.

.. _hks-difficulty:

Why it is not simply a matter of calling Arnoldi
------------------------------------------------

The wanted eigenvalue is **interior**. Measured on the same QA boundary:

.. code-block:: text

   rightmost eigenvalue    0.143 - 0.127i     (the ITG mode)
   spectral radius        80.15
   ratio                  ~560

Plain Arnoldi converges to extremal :math:`|\lambda|` and returns the large
:math:`|\mathrm{Im}\,\lambda|` modes. This was verified rather than assumed: the
error does **not** decrease with ``krylov_dim`` (30x to 300x too large at
``krylov_dim`` = 32, 64, 96), which distinguishes convergence to the wrong region
from under-convergence. The propagator variant maps max-Re to max-:math:`|\mu|`
via :math:`\mu = e^{\lambda \Delta t}`, but only while
:math:`|\mathrm{Im}\,\lambda|\,\Delta t \ll \pi`; at
:math:`|\mathrm{Im}\,\lambda| = 80` a step large enough to be useful aliases,
and a step small enough to avoid aliasing leaves per-step separation
:math:`e^{0.143\Delta t} \approx 1.0014`.

The matrix-free RHS and the dense matrix agree to 3e-16 on the same vector, so
none of this is an operator mismatch.

.. _hks-choice:

Why harmonic Krylov-Schur rather than shift-and-invert
------------------------------------------------------

Roman, Kammerer, Merz & Jenko [Roman2010]_ evaluated shift-and-invert, the
Cayley transform, and harmonic projection on GENE's linear gyrokinetic operator
-- the same structure and the same difficulty (they report the imaginary extent
exceeding the real by three orders of magnitude). Their conclusions:

* harmonic projection beats the spectral transformations "with a gain of one
  order of magnitude at least", and is "always at least five times faster";
* it converges with a small basis, about 10-12 vectors;
* it requires **no large linear solves**, which is decisive for a matrix-free
  code;
* shift-and-invert is a poor fit precisely because the operator is available
  only implicitly, so no good preconditioner exists for the inner solves, and
  its accuracy degrades as the shift moves away from the spectrum.

Harmonic extraction alone is **not** sufficient. Applied to a single Arnoldi
pass it does not recover the interior eigenvalue on GKX's operator. The method
in [Roman2010]_ is harmonic extraction *inside* Krylov-Schur with restarts: the
restart repeatedly filters the subspace toward the target, and the extraction
needs that filtered subspace to work from. The restart loop is the algorithm.

.. _hks-theory:

Theory
------

**Krylov decomposition.** Stewart's generalization [Stewart2001]_ of the Arnoldi
relation is

.. math::

   A V_m = V_m B_m + v_{m+1} b_m^{*},

where :math:`B_m` is not restricted to be Hessenberg and :math:`b_m` is
arbitrary. This is the key structural freedom: Arnoldi's Hessenberg form is what
makes deflation awkward in implicitly restarted Arnoldi [Sorensen1992]_, and
dropping it allows converged Ritz pairs to be decoupled cleanly.

**Krylov-Schur form.** Krylov decompositions are invariant under similarity
transformations of :math:`B_m`. Choosing a unitary :math:`Q` that brings
:math:`B_m` to (quasi-)triangular Schur form gives

.. math::

   A \tilde V_m = \tilde V_m T_m + v_{m+1} \tilde b_m^{*},

which can be truncated at any point by keeping a leading block of :math:`T_m`
and remains a Krylov decomposition. Restarting is then reordering the Schur form
so the wanted Ritz values occupy the leading block, and truncating -- no implicit
QR sweep, hence no forward instability.

**Harmonic extraction.** Standard Rayleigh-Ritz extracts eigenvalues of
:math:`B_m`, which approximate the *peripheral* spectrum. Harmonic Ritz values
about a target :math:`\kappa` approximate eigenvalues of :math:`A` closest to
:math:`\kappa` [Morgan1991]_, [Paige1995]_. With

.. math::

   g = (B_m - \kappa I)^{-*} b_m,

the harmonic Ritz values are the eigenvalues of :math:`B_m + g b_m^{*}`
([Roman2010]_, Eq. 21). Everything is :math:`m \times m` with :math:`m \sim 10`,
so the extraction costs nothing beside the matrix-vector products -- this is the
whole appeal relative to a spectral transformation, which pays a large linear
solve per iteration.

**Locking and purging.** [Stewart2001]_ defines both: locking decouples a
converged Ritz pair from the active subspace so later iterations do not rediscover
it, purging removes converged but unwanted pairs. Practical detail for the
harmonic variant is in [SLEPcSTR9]_; the standard variant is in [SLEPcSTR7]_.

.. _hks-numerics:

Numerics that must be got right
--------------------------------

These are the places this class of solver typically fails, each with the
mitigation this implementation will use.

**Orthogonality loss.** Classical Gram-Schmidt loses orthogonality
catastrophically in finite precision. Use CGS with one reorthogonalization pass
and a norm-based criterion (SLEPc's default; see [SLEPcSTR1]_). Gate: the basis
must satisfy :math:`\|V^{*}V - I\| < 10^{-10}` at every restart.

**Harmonic extraction breakdown.** :math:`(B_m - \kappa I)` is singular when
:math:`\kappa` is itself a Ritz value. Detect via condition estimate and perturb
:math:`\kappa`, rather than letting the solve return noise.

**Target selection.** Harmonic extraction needs a :math:`\kappa` near the wanted
eigenvalue. [Roman2010]_ report the error is *insensitive* to the target, unlike
shift-and-invert. GKX has a natural source: the dense eigenvalue at a small
:math:`(N_\ell, N_m)` costs milliseconds and lands within a few percent, so
continuation up the resolution ladder supplies :math:`\kappa` for free. Gate: the
converged eigenvalue must be independent of :math:`\kappa` over a stated range.

**Convergence criterion.** Judge on the residual of the *original* problem,
:math:`\|A x - \lambda x\| / \|\lambda\|`, not on the transformed one --
[Roman2010]_ show that the transformed criterion is exactly how shift-and-invert
returns confidently wrong answers.

**Complex arithmetic in JAX.** ``jax.numpy`` supports complex, but
``jnp.linalg.eig`` is CPU-only under JIT. The projected problem is
:math:`m \times m` with :math:`m \sim 10`, so it is cheap on host; the
matrix-vector products stay on device. Keep that split explicit.

**Determinism.** Restart decisions must not depend on non-deterministic
reductions, or the same input gives different eigenvalues on reruns. Gate:
bitwise-identical output across two runs with the same seed.

.. _hks-plan:

Implementation steps
--------------------

Each step is independently reviewable and leaves the tree green.

**S1 -- Matrix-free operator adapter.** A thin callable wrapping
``linear_rhs_cached`` in the shape the solver wants, built from
``_solver_geometry_context`` so it is the same operator the dense reference
uses. Validated by the 3e-16 agreement already measured.

**S2 -- Arnoldi with reorthogonalization.** Extend the existing ``_arnoldi`` to
return the Krylov decomposition in the form S3 needs, with the orthogonality
gate above. GKX already has this modulo the return signature.

**S3 -- Krylov-Schur restart.** Schur form of :math:`B_m`, reordering so wanted
Ritz values lead, truncation to the restart dimension. Gate against the dense
spectrum of the real operator at small :math:`n`.

**S4 -- Harmonic extraction.** :math:`g = (B_m - \kappa I)^{-*} b_m`, eigenvalues
of :math:`B_m + g b_m^{*}`, with the breakdown guard. Gate: harmonic Ritz values
approach eigenvalues near :math:`\kappa` faster than standard Ritz values do, on
the real operator.

**S5 -- Locking and purging.** Per [Stewart2001]_ and [SLEPcSTR9]_. Required for
computing the subdominant modes the transport work eventually needs, not only the
rightmost.

**S6 -- Continuation-based target.** Solve at the smallest ladder rung densely,
use that eigenvalue as :math:`\kappa` for the next rung, and so on. Removes the
target as a user-facing knob.

**S7 -- Custom JVP.** For a simple eigenvalue,
:math:`d\lambda = (w^{*} dA\, v)/(w^{*} v)` with :math:`w` the left eigenvector,
so the eigenpair carries an analytic derivative rather than being differentiated
through the iteration -- the same implicit-differentiation posture as the VMEC
adjoint. Only then can the differentiable objective switch over.

.. _hks-validation:

Validation gates
----------------

All against **real GKX operators**. No synthetic matrices: the dense solve is an
exact reference at the sizes where it is affordable, and it is the thing being
replaced, so it is the correct comparison.

**V1 Numerical agreement.** Eigenvalue matches dense to :math:`10^{-8}` relative
across :math:`(N_\ell, N_m)` from (2,3) to (10,14), on at least three devices
spanning QA, QH and QI.

**V2 Eigenvector agreement.** :math:`|\langle v_{\rm hks}, v_{\rm dense}\rangle|
> 1 - 10^{-8}` after phase alignment.

**V3 Residual.** :math:`\|A x - \lambda x\| / \|\lambda\| < 10^{-10}` on the
original problem.

**V4 Physics -- Landau damping.** Recovers the exact kinetic roots of
:math:`1 + T_i/T_e + \zeta Z(\zeta) = 0` to the accuracy the dense path achieves
(0.004% at :math:`T_e/T_i = 10`), reusing
``tests/validation/physics_gates/test_landau_damping.py``.

**V5 Physics -- benchmark parity.** Growth rates and frequencies on the tracked
linear benchmark cases stay within their existing tolerances, so the substitution
changes no shipped result.

**V6 Target insensitivity.** Converged eigenvalue independent of :math:`\kappa`
across a stated range, confirming [Roman2010]_'s claim in GKX's setting.

**V7 Determinism.** Bitwise-identical across reruns.

**V8 Performance.** Measured speedup and peak memory versus dense across the
ladder, with the crossover size reported. The claim to be established is the
order-of-magnitude gain [Roman2010]_ report; anything less is still worth having
but must be stated as measured.

**V9 Gradient.** Once S7 lands: custom JVP agrees with dense-path ``jax.grad``
and with finite differences to 1e-6 relative.

References
----------

.. [Stewart2001] G. W. Stewart, "A Krylov-Schur Algorithm for Large
   Eigenproblems", *SIAM J. Matrix Anal. Appl.* **23**\ (3), 601-614 (2001).
   https://doi.org/10.1137/S0895479800371529

.. [Sorensen1992] D. C. Sorensen, "Implicit Application of Polynomial Filters in
   a k-Step Arnoldi Method", *SIAM J. Matrix Anal. Appl.* **13**\ (1), 357-385
   (1992).

.. [Morgan1991] R. B. Morgan, "Computing Interior Eigenvalues of Large Matrices",
   *Linear Algebra Appl.* **154-156**, 289-309 (1991).

.. [Paige1995] C. C. Paige, B. N. Parlett & H. A. van der Vorst, "Approximate
   Solutions and Eigenvalue Bounds from Krylov Subspaces", *Numer. Linear Algebra
   Appl.* **2**\ (2), 115-133 (1995).

.. [Roman2010] J. E. Roman, M. Kammerer, F. Merz & F. Jenko, "Fast eigenvalue
   calculations in a massively parallel plasma turbulence code", *Parallel
   Computing* **36**\ (5-6), 339-358 (2010).
   https://doi.org/10.1016/j.parco.2009.12.001

.. [SLEPcSTR7] V. Hernandez, J. E. Roman, A. Tomas & V. Vidal, "Krylov-Schur
   Methods in SLEPc", SLEPc Technical Report STR-7.
   https://slepc.upv.es/documentation/reports/str7.pdf

.. [SLEPcSTR9] J. E. Roman, "Practical Implementation of Harmonic Krylov-Schur",
   SLEPc Technical Report STR-9.
   https://slepc.upv.es/documentation/reports/str9.pdf

.. [SLEPcSTR1] V. Hernandez, J. E. Roman, A. Tomas & V. Vidal, "Orthogonalization
   Routines in SLEPc", SLEPc Technical Report STR-1.
   https://slepc.upv.es/documentation/reports/str1.pdf
