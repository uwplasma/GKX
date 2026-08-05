Research-grade GKX: removing the proxies
========================================

Status: **plan**. Nothing here is promoted. Each step names what it removes or
proves, and the gate that decides it.

The one problem underneath all of this
--------------------------------------

GKX carries **two parallel objective stacks**, and the user-facing optimization
examples use the wrong one.

``src/gkx/objectives/stellarator_reduced.py`` and its facade, contracts and
tables are 1709 lines of an *analytic feature-map model* -- growth rate,
:math:`k_\perp^2` and flux weights from fitted expressions in the boundary
parameters, with the nonlinear "trace" an ODE envelope,

.. math::

   \frac{dE}{dt} = 2\gamma E - \alpha E^2, \qquad Q_{\rm env} = W_i E .

That is not gyrokinetics. The docstring of
``examples/optimization/QA_optimization_nonlinear_ITG.py`` says so plainly: it
optimizes "a reduced nonlinear-window ITG proxy" whose "smooth saturation-rule
proxy uses a dominant eigenvector and therefore uses finite-difference
optimization".

Meanwhile the *real* stack exists and works: ``objectives/core.py`` builds the
production linear RHS on a solver-ready flux tube and returns growth, frequency,
mode scale and quasilinear transport, differentiably, and the ``vmec_boozer_*``
modules carry the VMEC chain. Dense and matrix-free agree to **6.42e-11 across
all six objectives**, including the eigenvector-dependent ones.

So the work is not to build a research-grade path. It is to **delete the proxy
path and move the examples onto the real one**, then close the gaps that made
the proxy attractive in the first place.

Ordered steps
-------------

P1 -- Move the three optimization examples onto the real objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``QA_optimization_{linear,quasilinear,nonlinear}_ITG.py`` should call
``solver_objective_vector_from_geometry`` on VMEX geometry, not the reduced
model. Linear and quasilinear work today: the objective is differentiable and
the dense/matrix-free parity is measured. Nonlinear is blocked on P4.

*Gate*: each example reports a growth rate and a quasilinear flux that match a
direct call on the same boundary to round-off, and the reduced-model import
disappears from ``examples/``.

P2 -- Delete the reduced stellarator model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once P1 lands, ``stellarator_reduced.py``, ``stellarator.py``,
``stellarator_contracts.py`` and ``stellarator_tables.py`` have no production
consumer. Delete them and the tests that only exercise them.

*Gate*: 1709 lines out, coverage unchanged on the real path, and no test asserts
a property of a model nobody runs. Anything the reduced model was uniquely
testing gets re-pointed at the real objective or deleted with its rationale
recorded.

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

P1 and P3 are independent and can start now. P2 follows P1. P5 and P6 are
independent of everything and gate the credibility of all nonlinear numbers, so
they should not wait behind P4. P4 is the long pole and P4.1 is its first step.
P7 is a reporting exercise once P4--P6 land. P8 and P9 are independent. P10
follows whatever P1--P4 produce, because the README should showcase working
features rather than promise them.

What this deliberately does not do
----------------------------------

It does not add a surrogate, a scaffold, or a testbed anywhere. Where a
measurement is not affordable yet, the plan says so and states the gate that
would decide it, rather than substituting a cheaper stand-in and reporting the
stand-in's number.
