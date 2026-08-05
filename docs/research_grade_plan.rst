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

P5 -- Fix the statistics the gradients rest on
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Independently of the gradient method, every nonlinear window in the tree reports
:math:`\sigma/\sqrt{n}` over correlated samples. Measured: windows hold **2.6 to
11.8 independent samples**, so error bars are understated **2.0x to 3.7x**, and
the blocked production gate (``gradient_uncertainty_rel = 1.806`` against 0.5)
inherits that.

**P5.1** Make ``n_eff`` a required field of every window artifact, computed from
:math:`\tau_{\rm ac}`, and have the window gate assert a minimum.

**P5.2** Re-score the existing tracked windows. Some will fail; that is the point.

**P5.3** State every window length in :math:`\tau_{\rm ac}`, not code time.

*Gate*: no artifact reports a flux uncertainty computed as if samples were
independent.

P6 -- Precision
~~~~~~~~~~~~~~~

``integrate_nonlinear_scan`` sets ``state_dtype = result_type(G0, complex64)``,
so **a single-precision seed pins the whole trajectory to complex64 even with
JAX_ENABLE_X64 set**. This surfaced only because ``lax.scan`` rejected a
mismatched carry.

*Gate*: determine whether the tracked nonlinear results are single precision, and
either justify that (with a measured x64-vs-x32 comparison on one case) or fix
the seed dtype and re-run. This affects every nonlinear number in the repository,
so it comes before new production campaigns, not after.

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
