Harmonic Krylov-Schur eigensolver
=================================

Status: **adaptive propagator qualified as the production low-memory
eigensolver; objective integration remains gated.** SOLVAX now estimates a
stable full-operator RK4 step from a small peripheral-spectrum sketch, advances
in fixed physical-horizon restart chunks, and stops on the original-operator
residual. It matches dense eigenpairs on four-rung QA, QH, and QI ladders and is
at timing parity on QA and 1.89x and 2.34x faster than dense on QH and QI at
``n=4480``. A refined seven-configuration matrix passes ITG, ETG, TEM, KBM,
Miller, QHS, and QI.
Certificate-only GPU ladders reach ``n=172032`` with linear memory and certify
growth-rate convergence; an individual ``n=199680`` solve also passes. Full
complex-eigenvalue convergence remains unresolved for the QI ladder because the
frequency continues to move despite candidate overlaps above 0.99. The dense
differentiable objective should therefore remain the default until candidate
continuation and the implicit derivative are wired through the objective API
and the QI frequency gate is either passed or made an explicit growth-only
exception. Measurements and remaining work are below.

.. _hks-motivation:

Why
---

The linear growth rate is currently obtained from a dense eigendecomposition of
the flux-tube operator: all ``n`` eigenvalues and eigenvectors are computed and
one is kept, with ``n = n_laguerre * n_hermite * ntheta``. Measured on a QA
boundary at ``ntheta = 32``, the median of three dense solves scales as
:math:`t \sim n^{2.32}`:

.. list-table::
   :header-rows: 1

   * - :math:`(N_\ell, N_m)`
     - n
     - matrix
     - dense eig
   * - (2, 3)
     - 192
     - 0.6 MB
     - 0.01 s
   * - (4, 6)
     - 768
     - 9.4 MB
     - 0.40 s
   * - (6, 8)
     - 1536
     - 37.7 MB
     - 1.87 s
   * - (8, 10)
     - 2560
     - 104.9 MB
     - 6.66 s

Projected to converged resolution: ``(12,16)`` with ``ntheta = 64`` is 4.1 min
and 2.4 GB per evaluation; ``(32,16)`` -- roughly the published ITG guidance --
is 40.2 min and 17.2 GB, which exceeds the memory of the GPUs this runs on. A
convergence ladder needs several evaluations per configuration and a campaign
needs a ladder per configuration, so the dense path is what bounds how much
converged science is affordable. The :download:`cost-model artifact
<_static/eigensolver_cost_model.json>` retains all timing samples and
projections.

.. _hks-difficulty:

Why it is not simply a matter of calling Arnoldi
------------------------------------------------

The wanted eigenvalue is **interior**. Measured on the same QA boundary:

.. code-block:: text

   rightmost eigenvalue    0.08694 - 0.91716i
   spectral radius        79.36
   magnitude ratio        86.1

Plain Arnoldi converges to extremal :math:`|\lambda|` and returns the large
:math:`|\mathrm{Im}\,\lambda|` modes. This was verified rather than assumed: the
error does **not** decrease with ``krylov_dim`` (30x to 300x too large at
``krylov_dim`` = 32, 64, 96), which distinguishes convergence to the wrong region
from under-convergence.

A long-horizon propagator changes the problem: for
:math:`P \simeq e^{T A}`, :math:`|\mu|=e^{T\operatorname{Re}\lambda}` makes the
rightmost mode extremal, following the timestepper stability-analysis posture
of [Tuckerman2000]_. Phase wrapping changes only
:math:`\operatorname{Im}(\log\mu)/T`, not the amplification ordering. GKX
therefore selects by :math:`|\mu|` and recovers the physical complex
eigenvalue from :math:`v^* A v/v^*v`, never from the wrapped logarithm. The
multi-step map is full-operator RK4, a polynomial in :math:`A`, so it shares
eigenvectors with :math:`A`; a split IMEX map does not provide that guarantee.
The remaining cost is the physical growth-gap horizon and the explicit RK4
stability limit, not phase aliasing.

The matrix-free RHS and the dense matrix agree to 3e-16 on the same vector, so
none of this is an operator mismatch.

.. _hks-choice:

Why test harmonic Krylov-Schur before shift-and-invert
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
* shift-and-invert requires accurate inner linear solves, so its performance is
  controlled by a physics-structured preconditioner rather than only the outer
  eigensolver.

Those measurements are evidence for the method, not a guarantee for every GKX
operator. GKX now has an FFT-in-``z``/tridiagonal-Hermite line inverse plus an
exact Woodbury representation of the field-coupled low moments. The correction
substantially improves the inner residual, but the kinetic complement still
requires a large FGMRES space at the second validation rung. Harmonic and
rational approaches therefore remain measured competitors rather than
assumptions.

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
([Roman2010]_, Eq. 21). Everything is :math:`m \times m` with :math:`m \ll n`
(``m=64``--``128`` in the current GKX measurements), so the extraction remains
small beside the matrix-vector products. This is the appeal relative to a
spectral transformation, which pays a large linear solve per iteration.

**Harmonic restart and recovery.** It is not valid to extract harmonic Ritz
vectors and then restart from Schur vectors of :math:`B_m`. STR-9 translates
the decomposition so its Rayleigh quotient is
:math:`\widetilde B_m = B_m + g b_m^*`, Schur-sorts
:math:`\widetilde B_m`, truncates the harmonic subspace, and applies a recovery
translation that restores an orthonormal Krylov decomposition of the original
operator. The implementation now checks the recovered relation directly after
restart.

**Refined extraction.** Harmonic Ritz values can identify the desired interior
mode while the associated vector converges erratically. For each selected
harmonic value :math:`\theta`, refined extraction minimizes
:math:`\|(A-\theta I)V_m y\|` by taking the smallest right singular vector of
:math:`[B_m-\theta I; b_m^*]` [Jia2002]_. The final eigenvalue is the
original-operator Rayleigh quotient of that vector.

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
:math:`\kappa` is itself a Ritz value. Detect this with a condition estimate and
fall back to the untranslated quotient once the projected solve has lost about
half the working digits.

**Target and branch selection.** Harmonic extraction needs a :math:`\kappa`
near the wanted eigenvalue. Resolution continuation is useful, but the dominant
branch moves enough on the tested ladder that a target solver can converge
accurately to the wrong physical mode. The numerical gate therefore uses each
rung's dense value as an oracle target, while a separate continuation audit
tests branch tracking. Production continuation must retain a small candidate
subspace and use overlap/conditioning diagnostics rather than one scalar seed.

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

.. _hks-result:

Measured result on real operators
----------------------------------

The corrected solver preserves the restarted Krylov relation to about
``3e-15`` and passes normal and triangular non-normal synthetic spectra. On the
shipped QA low-resolution VMEC input, using an oracle target and ``m=64``, it
passes the first two rungs but not the larger two:

.. list-table::
   :header-rows: 1

   * - :math:`(N_\ell, N_m)`
     - n
     - ratio
     - dense
     - hks
     - rel. error
     - converged
   * - (4, 6)
     - 768
     - 47
     - 0.39 s
     - 4.19 s
     - 6.62e-11
     - YES
   * - (6, 8)
     - 1536
     - 66
     - 1.66 s
     - 11.67 s
     - 4.42e-10
     - YES
   * - (8, 10)
     - 2560
     - 86
     - 6.04 s
     - 17.71 s
     - 3.83e-07
     - NO
   * - (10, 14)
     - 4480
     - 128
     - 25.19 s
     - 21.39 s
     - 9.46e-02
     - NO

Increasing the basis confirms that the largest rung is approachable: at
``m=128`` its eigenvalue error reaches ``1.4e-10``, but the vector residual is
still ``5.9e-9`` after 26,064 matrix-vector products. That is evidence of a
convergence-rate and subspace-size problem, not a reason to accept the current
cost. Locking is required for multiple/clustered modes but cannot explain the
present ``k=1`` failure.

The field-corrected rational path changes the accuracy result but not yet the
performance verdict. A one-candidate, ``m=8`` solve with a prepared shifted
inverse gives:

.. list-table::
   :header-rows: 1

   * - :math:`(N_\ell, N_m)`
     - n
     - compile
     - warm rational
     - dense
     - rel. error
     - residual
   * - (4, 6)
     - 768
     - 1.72 s
     - 2.90 s
     - 0.38 s
     - 7.45e-12
     - 4.10e-11
   * - (6, 8)
     - 1536
     - 4.36 s
     - 20.64 s
     - 1.72 s
     - 2.81e-11
     - 1.34e-10
   * - (8, 10)
     - 2560
     - 23.02 s
     - 117.09 s
     - 6.23 s
     - 1.57e-11
     - 7.37e-11

All three converge in one restart and 15 outer applications. That count does
not include FGMRES-internal operator applications. Strict inner spaces of 512,
1024, and 2048 vectors respectively are what make the residuals pass, so the
kinetic complement remains the scaling wall. On the office GPU, the first two
warm solves take 30.91 and 102.61 seconds; reduction latency makes CPU the
preferred backend at these sizes.

The long-horizon path resolves the accuracy failure without a shifted solve.
The original uniform configuration (RK4 ``dt=0.015``, horizon 60, ``m=16``)
passes all four QA rungs:

.. list-table::
   :header-rows: 1

   * - :math:`(N_\ell, N_m)`
     - n
     - compile
     - warm propagator
     - dense
     - rel. error
     - residual
   * - (4, 6)
     - 768
     - 14.71 s
     - 13.81 s
     - 0.37 s
     - 1.21e-14
     - 1.30e-14
   * - (6, 8)
     - 1536
     - 21.92 s
     - 21.09 s
     - 1.65 s
     - 1.42e-14
     - 1.70e-14
   * - (8, 10)
     - 2560
     - 26.95 s
     - 26.06 s
     - 5.86 s
     - 2.63e-14
     - 2.01e-14
   * - (10, 14)
     - 4480
     - 37.44 s
     - 37.26 s
     - 24.14 s
     - 1.00e-13
     - 8.04e-13

The :download:`long-horizon artifact
<_static/long_horizon_propagator_validation.json>` records the input hash,
versions, device, timings, horizon, and exact operator-evaluation count.
Regenerate it with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python tools/campaigns/validate_harmonic_krylov_schur.py \
     --input examples/vmec/input.LandremanPaul2021_QA_lowres \
     --solver long-horizon --krylov-dim 16 --max-restarts 1 \
     --tol 1e-9 --propagator-dt 1.5e-2 --propagator-steps 4000

The production candidate removes that uniform-setting penalty. Two 12-vector
Arnoldi sketches, one from the caller seed and one deterministic broadband
probe, estimate the peripheral spectrum and evaluate the actual complex RK4
stability polynomial to select ``dt`` with a 0.9 safety factor.
Each restart advances by horizon 30 and stops as soon as the original-operator
relative residual is below tolerance. A growth-defect guard rejects artificial
RK4 amplification. If either stability gate fails, the step is halved and
retried.

The adaptive result closes V1--V3 and establishes the measured V8 crossover on
three stellarator devices:

.. list-table::
   :header-rows: 1

   * - device
     - n
     - dense
     - warm adaptive
     - speedup
     - rel. error
     - residual
   * - QA
     - 4480
     - 26.89 s
     - 27.97 s
     - 0.96x
     - 9.96e-14
     - 6.49e-14
   * - QH
     - 4480
     - 26.06 s
     - 13.82 s
     - 1.89x
     - 1.91e-12
     - 1.14e-13
   * - QI
     - 4480
     - 26.65 s
     - 11.39 s
     - 2.34x
     - 4.32e-14
     - 3.78e-14

The robust :download:`QA artifact
<_static/adaptive_propagator_broadband_qa_validation.json>`,
:download:`QH crossover artifact
<_static/adaptive_propagator_broadband_qh_rung4_validation.json>`, and
:download:`QI crossover artifact
<_static/adaptive_propagator_broadband_qi_rung4_validation.json>` record
selected steps, restart counts, operator evaluations, versions, and timings.
The earlier single-probe artifacts remain as provenance for the estimator
ablation. Regenerate the production setting with
``--solver adaptive-propagator``; for example:

.. code-block:: bash

   JAX_ENABLE_X64=1 python tools/campaigns/validate_harmonic_krylov_schur.py \
     --input examples/vmec/input.LandremanPaul2021_QA_lowres \
     --solver adaptive-propagator --krylov-dim 16 --max-restarts 4 \
     --tol 1e-9 --propagator-chunk-horizon 30 \
     --stability-dimension 12 --stability-probe-count 2 \
     --stability-safety 0.9

A refined block-propagator restart remains available for competing branches.
The scalar path is preferred for a single isolated objective; continuation must
retain multiple candidates near crossings.

The TOML-driven adaptive physics matrix passes all seven shipped configurations:
ITG, ETG, TEM, KBM, Miller, QHS, and QI. It covers
electrostatic/electromagnetic, single-/multi-species, and periodic/linked
layouts. On the refined ``(N_l,N_m)=(4,6)`` matrix, the maximum relative
eigenvalue error is ``1.45e-10`` and maximum original-operator residual is
``1.09e-9``. This is a branch-selection and architecture gate only, not a claim
of velocity-space convergence. The :download:`broadband-probe artifact
<_static/adaptive_propagator_broadband_physics_matrix_refined.json>` records
input hashes, device, versions, timings, and the exact scope. The earlier
single-probe artifact remains as estimator-ablation provenance.

Regenerate it from the repository root with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python tools/campaigns/validate_rational_physics_matrix.py \
     --solver adaptive-propagator --spatial-points 16 \
     --n-laguerre 4 --n-hermite 6 --stability-probe-count 2

The rational and harmonic solvers remain useful fallbacks and research
comparators, but the adaptive propagator is the measured default for a
rightmost eigenpair. The next production work is candidate continuation through
crossings and objective integration with implicit sensitivities. The low-moment
field Schur/Woodbury correction remains available for rational extraction; its
measured bottleneck is convergence of the kinetic complement.

Recycling the eigenvector, zero-padded into the next Hermite--Laguerre
resolution, is much more effective than recycling only the scalar target. With
guarded recycling and ``m=80``, the first three rungs pass V1 in 85, 134, and
243 restarts. The largest rung improves to ``6.6e-8`` eigenvalue error and
``3.5e-7`` residual, but still fails after 16,440 matrix-vector products and
36.5 s versus 27.4 s dense. This is a useful continuation path, not yet a
production replacement.

Convergence beyond the dense-memory limit
-----------------------------------------

Dense-oracle agreement establishes solver accuracy, not velocity-space
convergence. The adaptive path was therefore run at ``ntheta=64`` without
forming a dense matrix. Every reported point passes the original-operator
residual, RK4 stability, and growth-defect gates. Growth and the full complex
eigenvalue are audited separately because a growth-only linear objective does
not require a converged frequency.

The fine QA ladder ``(16,24)`` through ``(28,36)`` reaches ``n=64512``. Its
growth changes are 0.76%, 0.66%, and 0.69%, while full-eigenvalue changes are
0.51%, 0.45%, and 0.49%; both quantities pass the 5% two-consecutive-rung gate.
The QI ultra-fine ladder ``(24,32)`` through ``(36,44)`` reaches ``n=101376``.
Growth changes are 2.45%, 2.63%, and 1.64%, so the growth rate passes. The
full-eigenvalue changes are 19.58%, 15.26%, and 11.53%, so the frequency does
not. Consecutive normalized eigenvector overlaps rise from 0.9878 to 0.9934,
which is evidence for slow convergence of one branch rather than accidental
branch switching, but it does not waive the numerical gate.

The production two-probe hyperfine ladder continues through ``n=172032``. Its
three rows all certify in one restart, with selected steps decreasing from
``0.0058651`` to ``0.00522193`` as the velocity resolution rises. The largest
certified point has ``lambda=0.09702353767 + 0.02707803460i``, relative
residual ``8.61e-13``, 367,706 original-operator evaluations, and 229.3 s warm
solve time. Its dense complex matrix would require about 474 GB, whereas the
GPU campaign remained near 1.24 GiB observed allocation. Growth changes of
0.54% and 1.32% pass the two-consecutive-rung gate; full-eigenvalue changes of
7.24% and 6.20% do not, so the QI frequency remains unresolved.

This is the principal scaling result: memory is linear in state size and the
solve is feasible well beyond the dense path. See the :download:`QA/QI fine
artifact <_static/adaptive_propagator_convergence_fine.json>`,
:download:`QI ultra-fine artifact
<_static/adaptive_propagator_convergence_qi_ultrafine.json>`, and
:download:`production broadband hyperfine artifact
<_static/adaptive_propagator_convergence_qi_broadband_validation.json>`.

A predecessor hyperfine extension records the first large-scale failure rather
than hiding it. Rungs at ``n=122880`` and ``n=146432`` pass with residuals
``9.05e-10`` and ``6.32e-13``. At ``n=172032`` the recycled-seed-only
12-vector stability sketch admits a step that returns essentially the prolonged
prior vector (overlap 1.0) with residual ``1.67e1`` after all four restarts. The
next rung, ``n=199680``, selects roughly half that step and passes with residual
``2.80e-12`` in 684.1 s. Its dense complex matrix would be about 638 GB while
observed GPU allocation remained about 1.24 GiB. The whole predecessor ladder
is correctly marked uncertified because every rung must pass; neither growth
nor eigenvalue convergence is claimed. The :download:`negative-evidence artifact
<_static/adaptive_propagator_convergence_qi_hyperfine.json>` preserves the raw
values and timings. This pattern motivates a bounded step-halving retry after
residual exhaustion, even when the selected scalar mode alone appears RK4
stable.

That retry is now implemented and closes the failed rung. Replaying the
``n=122880``, ``146432``, and ``172032`` continuation selects
``dt=0.0028082`` after the estimated ``dt=0.0056159`` exhausts its residual
budget. The recovered value is
``0.09702353767 + 0.02707803460i`` with residual ``1.39e-12`` and continuation
overlap 0.9954. Growth changes of 0.54% and 1.32% pass the two-consecutive-rung
gate. Full-eigenvalue changes of 7.24% and 6.20% remain outside 5%, so the
frequency is still not converged. The warm solve takes 1279.6 s and 2,051,286
operator evaluations including the failed first attempt; this is a correctness
fallback, not the desired steady-state cost. See the :download:`retry artifact
<_static/adaptive_propagator_convergence_qi_retry_validation.json>`.
The production broadband probe avoids that failed attempt, reducing both warm
time and operator work by 5.58x while retaining the retry as a fail-safe.

Regenerate a certificate-only ladder with:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
   JAX_ENABLE_X64=1 python \
     tools/campaigns/validate_adaptive_propagator_convergence.py \
     --device qi --ntheta 64 \
     --resolution 40,48 --resolution 44,52 \
     --resolution 48,56 \
     --krylov-dim 16 --max-restarts 4 --tol 1e-9 \
     --convergence-tol 0.05 --chunk-horizon 30 \
     --stability-dimension 12 --stability-probe-count 2 \
     --stability-safety 0.9

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

**S3 -- Harmonic Krylov-Schur restart.** Translate, Schur-sort the modified
quotient, truncate, and recover per STR-9. Gate the Krylov relation itself as
well as the dense spectrum. Implemented in SOLVAX.

**S4 -- Refined harmonic extraction.** Select through
:math:`B_m + g b_m^{*}`, minimize the original residual in the retained
subspace, and return its Rayleigh quotient. Implemented in SOLVAX.

**S5 -- Block candidates, locking, and recycling.** Implemented in SOLVAX,
including structured continuation seeds, duplicate rejection, exact-target
breakdown handling, and an optional rational subspace action. Required for
branch crossings and the subdominant modes transport eventually needs.

**S6 -- Candidate continuation.** Solve a small rung densely, retain several
rightmost candidates, and propagate them with biorthogonal overlap. Periodically
reseed from a broader spectral search so a newly dominant branch is not missed.

**S7 -- Implicit eigenpair derivative.** SOLVAX now provides a custom JVP. For
growth-only objectives it uses
:math:`d\lambda = (w^{*} dA\, v)/(w^{*} v)`. For eigenvector-dependent
quasilinear outputs, it solves Nelson's bordered system [Nelson1976]_ for the
right-eigenvector tangent and differentiates the matrix-free action
:math:`A(p)v`. A left/right condition-number guard fails explicitly near a
cluster or exceptional point. Replacing GKX's dense objective remains gated on
the real-operator solver qualification. On a 24-state real GKX operator, a
phase-invariant objective combining growth and a quadratic eigenvector weight
has implicit gradient ``0.22387295824979078`` versus
``0.2238729582497859`` from differentiating the dense eigensolve; centered
finite differences agree to ``8.1e-10`` relative. This validates the AD
mechanism, not the production branch selection.

**S8 -- Long-horizon polynomial propagator.** Implemented in GKX. Multi-step
basis generation uses full-operator RK4, selection uses amplification magnitude,
and the returned value is the continuous-operator Rayleigh quotient. SOLVAX's
block transformed-subspace path uses Rayleigh--Ritz for ``largest_real``;
harmonic extraction about an unrelated target would undo the transformation.
The scalar and block paths always certify the original-operator residual.

**S9 -- Adaptive stability, horizon, and certification.** Implemented in SOLVAX
and GKX. Caller-seeded and deterministic broadband peripheral-spectrum sketches
choose a stable RK4 step using the complex stability region,
residual-controlled restart chunks avoid unnecessary long horizons, and a
growth-defect guard detects numerically induced amplification. Step halving
after either instability or residual exhaustion is the guarded fallback. The
estimator, invariant-seed blindness, early stop, retry accounting, and
instability rejection have focused regression tests.

.. _hks-validation:

Validation gates
----------------

Release gates use **real GKX operators** against dense references. Synthetic
normal, non-normal, clustered, Grcar, and near-Jordan matrices remain mandatory
unit tests because they isolate restart invariants and conditioning failures.

**V1 Numerical agreement.** Passed by the adaptive propagator: eigenvalue
matches dense to :math:`10^{-8}` relative across four rungs through
``(10,14)`` on QA, QH, and QI.

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

**V8 Performance.** Passed on QH and QI at the first measured crossover: at
``n=4480`` the robust two-probe warm solve is 1.89x and 2.34x faster than dense.
QA is at timing parity (0.96x), so no QA crossover is claimed. The certified GPU
continuation reaches ``n=172032`` with observed allocation about 1.24 GiB; a
dense complex matrix there is about 474 GB. An individual ``n=199680`` row also
passes. This is not an order-of-magnitude timing claim, and the 172k residual
fallback is not yet a performance win.

**V9 Gradient.** SOLVAX's custom JVP agrees with dense-path ``jax.grad`` and
frozen-branch directional finite differences on the small real-operator gate.
GKX objective integration and end-to-end optimization gradients remain open.

References
----------

.. [Stewart2001] G. W. Stewart, "A Krylov-Schur Algorithm for Large
   Eigenproblems", *SIAM J. Matrix Anal. Appl.* **23**\ (3), 601-614 (2001).
   https://doi.org/10.1137/S0895479800371529

.. [Jia2002] Z. Jia, "The refined harmonic Arnoldi method and an implicitly
   restarted refined algorithm for computing interior eigenpairs of large
   matrices", *Applied Numerical Mathematics* **42**, 489-512 (2002).
   https://doi.org/10.1016/S0168-9274(01)00132-5

.. [Nelson1976] R. B. Nelson, "Simplified calculation of eigenvector
   derivatives", *AIAA Journal* **14**\ (9), 1201-1205 (1976).
   https://doi.org/10.2514/3.7211

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

.. [Tuckerman2000] L. S. Tuckerman & D. Barkley, "Bifurcation Analysis for
   Timesteppers", in *Numerical Methods for Bifurcation Problems and Large-Scale
   Dynamical Systems*, IMA Volumes in Mathematics and its Applications
   **119**, 453-466 (2000).
   https://doi.org/10.1007/978-1-4612-1208-9_20

.. [SLEPcSTR7] V. Hernandez, J. E. Roman, A. Tomas & V. Vidal, "Krylov-Schur
   Methods in SLEPc", SLEPc Technical Report STR-7.
   https://slepc.upv.es/documentation/reports/str7.pdf

.. [SLEPcSTR9] J. E. Roman, "Practical Implementation of Harmonic Krylov-Schur",
   SLEPc Technical Report STR-9.
   https://slepc.upv.es/documentation/reports/str9.pdf

.. [SLEPcSTR1] V. Hernandez, J. E. Roman, A. Tomas & V. Vidal, "Orthogonalization
   Routines in SLEPc", SLEPc Technical Report STR-1.
   https://slepc.upv.es/documentation/reports/str1.pdf
