Research-grade GKX: removing the proxies
========================================

Status: **plan**. Nothing here is promoted. Each step names what it removes or
proves, and the gate that decides it.

What the optimization examples actually compute
-----------------------------------------------

An earlier version of this page asserted that the three optimization examples
run on ``objectives/stellarator_reduced.py``'s analytic feature-map model. That
is **wrong**, and the correction matters because it inverts the plan's ordering.
Reading the call chain rather than the filenames:

.. list-table::
   :header-rows: 1

   * - example
     - calls
     - what it computes
   * - linear
     - ``turbulent_growth_rate``
     - GKX's **real** dense linear operator, then ``eigvals``
   * - quasilinear
     - ``quasilinear_flux_proxy``
     - **real** linear solve, then a named mixing-length saturation rule
   * - nonlinear
     - ``nonlinear_heat_flux_proxy``
     - **real** linear solve, then the algebraic closure
       :math:`c_{\rm sat} W_Q\, 2\gamma_+ / (1 + 2.2 k_{\perp,\rm eff}^2 + 0.15\gamma_+)`

All three already sit on the real linear gyrokinetic solve. The reduced
feature-map model is consumed by ``vmec_transport.py``, the
``reduced_stellarator_itg`` demo and one test -- not by these examples.

So the genuine gap is narrower and deeper than "migrate the examples". The
nonlinear objective is an **algebraic surrogate of nonlinear flux built from
linear quantities**: quasilinear-class physics wearing a nonlinear label. No file
move fixes that. It is fixed by having a differentiable nonlinear flux at all,
which is P4.

P1 -- Rename what the nonlinear surrogate claims
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until P4 lands, the honest change is naming. ``nonlinear_heat_flux_proxy``
returns a closure over linear quantities; calling it a nonlinear heat flux
invites exactly the reading this page made. Rename it to
``saturation_rule_flux_estimate`` and say in one line what closure it applies.

*Gate*: no public name asserts nonlinear gyrokinetics for a quantity computed
from a linear solve.

P2 -- Re-scope, do not delete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``stellarator_reduced.py`` and friends have three real consumers. Whether they
should exist is a separate question this page has not answered, and the previous
version of it proposed deleting 1709 lines on an inference that turned out to be
wrong. Deferred until someone establishes what the reduced model is uniquely for.

P3 -- Audit the remaining "proxy" names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``flux_proxy`` appears in the parallel-decomposition identity gates
(``spectral_identity_integrator.py``, ``domain_decomposition.py``,
``device_z.py``). Those are **legitimate**: they compare serial against
decomposed execution and need a cheap trace-level observable, not a transport
number, and they already say so. Keep them, rename to
``decomposition_trace_observable`` so the word "flux" stops implying transport.

The quasilinear *mixing-length* proxy in ``core.py`` is also legitimate and
stays -- it is a named model with a stated saturation rule, not a stand-in for
something GKX cannot do.

*Gate*: every remaining use of "proxy" in ``src/`` is a named physical model or
an identity-check observable, with the distinction stated at the definition.

P4 -- Nonlinear autodiff
~~~~~~~~~~~~~~~~~~~~~~~~

This is the enabling step for everything nonlinear, and the measurements needed
to scope it already exist (``docs/nonlinear_gradient_plan.rst``, on the nonlinear branch).

What is known: on a verified-saturated state the windowed adjoint grows as a
**power** of :math:`N` out to :math:`t \approx 4.5\,\tau_{\rm ac}` with no
exponential divergence, so a usable window exists and is *longer* than the
published flux-tube result. What is not known: where the knee is, because
reverse mode exhausts a 16 GB card at :math:`N = 2048` even with
rematerialization.

Steps, in order:

**P4.1** Replace the state-norm objective with the **production heat flux**. The
current tool differentiates mean-square amplitude, which shares the Lyapunov
behaviour but is not what anyone optimizes. This needs the field solve inside the
differentiated window.

**P4.2** Find the knee. Options that fit in memory: nested ``lax.scan`` with
two-level rematerialization, gradient accumulation over sub-windows, or sharding
the window across both A4000s. Report the knee in :math:`\tau_{\rm ac}`.

**P4.3** Measure the bias against finite differences inside the window. The
published comparison recovers 15--34% of the FD value and still optimizes;
reproducing that ratio would be a *reproduction*, not a failure.

**P4.4** Decide the production scheme on evidence: windowed adjoint if the bias
is tolerable and the variance is low, NILSS shadowing if not. Shadowing is
unbiased and expensive; it is the fallback, not the default.

*Gate*: a short descent on one boundary coefficient lowers the nonlinear flux,
measured on the correlation-time protocol, with the gradient's bias **and**
variance both reported.

P5 -- Fix the statistics the gradients rest on. P5.1 and P5.2 DONE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every nonlinear window in the tree reported :math:`\sigma/\sqrt{n}` over
correlated samples, which understates the uncertainty of the mean by
:math:`\sqrt{n/n_{\rm eff}}`.

**P5.1 -- done.** ``NonlinearWindowMetrics`` now carries ``heat_flux_tau_ac``,
``window_in_tau_ac``, ``heat_flux_n_eff`` and ``heat_flux_stderr``, computed in
``gkx.diagnostics.analysis`` where every window metric already flows. The
corrected standard error divides by :math:`\sqrt{n_{\rm eff}}`, and a test pins
the contract against an AR(1) process with a known correlation time.

**P5.2 -- done, and the finding holds.** Re-scoring the 16 tracked heat-flux
traces through the production path:

.. list-table::
   :header-rows: 1

   * - quantity
     - range
   * - :math:`\tau_{\rm ac}`
     - 3.98 -- 15.59
   * - window length in :math:`\tau_{\rm ac}`
     - 5.5 -- 26.9
   * - **independent samples**
     - **2.6 -- 11.8**
   * - error-bar understatement
     - **2.05x -- 3.67x**

**14 of 16 windows hold fewer than ten independent samples**, and the worst
averages 2.6. This is the one nonlinear finding in this program that has grown
rather than shrunk under scrutiny -- the recycling win, the movie blow-up and the
precision defect all became smaller when measured; this became larger.

**P5.3 -- done, with the design corrected by a test.** The plan said to gate a
minimum :math:`n_{\rm eff}`. That is **wrong**, and the existing convergence
test caught it: a very smooth trace is maximally autocorrelated and so has small
:math:`n_{\rm eff}`, while its mean is extremely well determined precisely
because it barely varies. Gating :math:`n_{\rm eff}` would fail exactly the
converged windows the gate exists to accept.

The gate is therefore on the **correlation-corrected relative standard error**,
:math:`\sigma/(\bar q\sqrt{n_{\rm eff}})`, which combines variance and
independence correctly. :math:`n_{\rm eff}` is an input to it, not a criterion.

``NonlinearHeatFluxConvergenceMetrics`` carries ``tau_ac`` and ``n_eff``, both
defaulting to the fail-closed values so a hand-constructed metrics object cannot
pass a statistical gate by omission -- two test fixtures had to state their
independent-sample count explicitly, which is the intended consequence.

P6 -- Precision. MEASURED, and it is a controllability bug, not an accuracy one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``integrate_nonlinear_scan`` sets ``state_dtype = result_type(G0, complex64)``.
Confirmed by running the production path with ``JAX_ENABLE_X64=True``: the
returned state is ``complex64``. **Every tracked nonlinear number in the tree was
computed in single precision.**

Whether that matters was then measured rather than assumed. Same case, same seed,
integrated to ``t = 60`` in both precisions:

.. list-table::
   :header-rows: 1

   * - precision
     - final fluctuation energy
   * - ``complex64``
     - ``2.4002927603e-05``
   * - ``complex128``
     - ``2.4003281458e-05``
   * - relative difference
     - **1.47e-05**

That is three orders of magnitude below the error-bar understatement P5
addresses and far below the window-to-window scatter, so single precision is
**not** a threat to the tracked results. An earlier version of this page claimed
it "would affect every nonlinear number in the repository"; that claim is
withdrawn.

The caveat the measurement does not cover: ``t = 60`` is still the growth phase,
where two precisions track each other. A chaotic saturated trajectory separates
exponentially, so this bounds the *integrator's* precision sensitivity, not the
long-time divergence of two trajectories. A saturated-regime comparison should
report ensemble statistics, not a pointwise difference, because pointwise
divergence there is expected physics rather than a defect.

The defect that remains is real but narrower: **double precision cannot be
requested**. ``result_type(G0, complex64)`` silently ignores ``JAX_ENABLE_X64``
unless the caller happens to pass a complex128 seed. That is a controllability
bug.

*Gate*: a complex128 seed, or an explicit dtype argument, produces a complex128
trajectory, and a test asserts it. The default may stay single precision -- it is
faster and, measured, accurate enough.

P7 -- The physics matrix
~~~~~~~~~~~~~~~~~~~~~~~~

Research-grade means each cell is *gated*, not merely runnable. Current state
from the verification matrix and benchmark parity:

.. list-table::
   :header-rows: 1

   * - axis
     - status
   * - electrostatic / electromagnetic
     - both run; KBM parity is the known 20% outlier
   * - adiabatic / kinetic electrons
     - both run; TEM finite-mass gated
   * - tokamak / stellarator
     - both; VMEC and Boozer paths differentiable
   * - linear / quasilinear
     - gated, dense and matrix-free agree to 6.4e-11
   * - nonlinear
     - runs; **gradients blocked**, statistics understated
   * - collisions
     - five operators, conservation and H-theorem at machine precision;
       species-coupled Coulomb open
   * - recurrence
     - reflectionless closure lands; see P8

The gaps that stop this being research-grade are **nonlinear gradients (P4)**,
**window statistics (P5)**, **precision (P6)**, and **KBM parity**. Nothing else
on the matrix is missing -- it is the evidence quality that is uneven.

P8 -- Recurrence
~~~~~~~~~~~~~~~~

Free streaming in a truncated Hermite basis recurs at
:math:`t_{\rm rec} \approx 2\sqrt{M}/(k_\parallel v_{\rm th})`. GKX ships the
reflectionless closure :math:`R_{M+1} = 1 - 1/(4M)` asymptotically, which is the
Hammett--Perkins value at :math:`M=3`.

*Gate*: for each production resolution, report :math:`t_{\rm rec}` against the
run length and the correlation time. A run longer than :math:`t_{\rm rec}`
without an adequate closure is reporting recurrence, not physics -- this belongs
in the same artifact as :math:`n_{\rm eff}`, since both answer "is this window
meaningful".

P9 -- Performance
~~~~~~~~~~~~~~~~~

Measured facts to build on: the dense eigensolve is bounded by **memory, not
speed** (:math:`O(n^2)`; 3.6 TiB at :math:`n = 494{,}592`), the matrix-free path
is :math:`O(nm)`, and the shift-invert inner solver was tested and the incumbent
kept (:doc:`solvax_defaults`).

Untested candidates, in the order their expected payoff justifies:

1. ``p_multigrid`` over the ``(Nl, Nm)`` hierarchy -- the coarse space is exactly
   a subspace of the fine one, so restriction is truncation and prolongation is
   zero-padding, making the transfers exact by construction.
2. ``mixed_precision`` + ``iterative_refinement`` -- fp32 preconditioner, fp64
   residuals. Needs GPU measurement, and interacts with P6.
3. ``chunked_jacrev`` -- GKX chunks forward-mode Jacobians only; optimization is
   the reverse-mode case, and that is where peak memory binds.

*Gate*: each is adopted only against a measurement on the production operator
with a control that could have failed.

P10 -- README
~~~~~~~~~~~~~

Lead with what is impactful and true, in this order: the turbulence movie, the
collision hierarchy no other JAX code has, end-to-end differentiability, and the
resolutions the matrix-free path reaches. Scans and landscapes are diagnostics
and belong in the docs.

*Gate*: every README figure answers "why would I use this" rather than "what did
we measure last week", and no figure whose own status line reports a failure
appears as a result.

Sequencing
----------

**P5 and P6 come first.** They are independent of everything, they gate the
credibility of every nonlinear number already in the tree, and neither depends on
research going well. P1 and P3 are naming work and can run alongside. P4 is the
long pole; the nonlinear example is rewritten as part of it, not before it. P2 is
deferred.
P7 is a reporting exercise once P4--P6 land. P8 and P9 are independent. P10
follows whatever P1--P4 produce, because the README should showcase working
features rather than promise them.

What this deliberately does not do
----------------------------------

It does not add a surrogate, a scaffold, or a testbed anywhere. Where a
measurement is not affordable yet, the plan says so and states the gate that
would decide it, rather than substituting a cheaper stand-in and reporting the
stand-in's number.
