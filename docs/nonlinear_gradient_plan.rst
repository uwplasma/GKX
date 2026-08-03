Nonlinear turbulence gradients
==============================

Status: **plan, with one measurement that has to come first.** Nothing here is
promoted; the point of the first step is to find out whether the current
approach can work at all.

Why the current approach is stuck
---------------------------------

GKX gets nonlinear turbulence gradients by central finite differences over
matched long post-transient windows. The production gate records why that is
blocked (``docs/_static/nonlinear_turbulence_gradient_evidence_gap_report.json``):

.. code-block:: text

   gradient_uncertainty_rel = 1.806     gate maximum = 0.5

The gradient's own uncertainty is 3.6x the tolerance -- the measured gradient is
not statistically distinguishable from noise. Because a finite-difference error
on a chaotic signal falls as :math:`1/\sqrt{T/\tau_{\rm ac}}`, closing that gap by
averaging alone needs

.. math::

   \left(\frac{1.806}{0.5}\right)^2 \approx 13\times

longer windows, per perturbed point, per design coefficient. The
time-horizon audit already reports 0 of 36 records production-ready with 8
"long but failed convergence". Thirteen times longer is not a scheduling
problem, it is a different method.

This is not a GKX defect. It is the generic obstruction: **derivatives of
long-time averages of chaotic systems are ill-conditioned as initial-value
problems** [Wang2014]_. Naive adjoints diverge with the leading Lyapunov
exponent, and finite differences convert that divergence into variance.

What the literature does instead
--------------------------------

Two families, with opposite trade-offs.

**Windowed adjoint (biased, cheap).** Backpropagate only the last :math:`N` steps
from a state already in the saturated regime. [iGENE2026]_ does exactly this for
flux-tube gyrokinetics in a differentiable framework, and reports the number that
matters: gradients diverge for :math:`N \gtrsim 512` steps, while the heat-flux
autocorrelation time is 500-1000 steps. **The usable window is set by the
dynamical memory of the turbulence, not by a solver tolerance.** Inside the
window their gradients recovered only 15--34% of the finite-difference values --
biased low -- and still drove a successful seven-point optimization. Their
practical recipe is :math:`N = 16` backpropagated steps taken every 1000
simulation steps.

**Shadowing (unbiased, expensive).** Least-squares shadowing replaces the
ill-conditioned initial-value problem with a well-conditioned least-squares
problem along a shadow trajectory, and converges to the derivative of the
infinite-time average [Wang2014]_. Non-intrusive LSS [Ni2017]_ needs only tangent
or adjoint solves that a code already has; multiple-shooting shadowing
[Blonigan2018]_ reduces its memory and runtime.

The honest reading for GKX: a windowed adjoint is the cheap thing to try, its
bias is real and must be reported, and shadowing is the fallback if a biased
descent direction turns out not to be good enough.

What GKX must measure first
---------------------------

GKX has **no autocorrelation tooling at all** -- the quantity that sets the
window is not currently computed anywhere in the repository. Every downstream
choice depends on it, so it comes first.

**N1 -- Heat-flux autocorrelation time. DONE, and the result is worse than
expected.** ``tools/campaigns/heat_flux_autocorrelation.py`` post-processes the
16 committed heat-flux traces (no new simulation). Measured
:math:`\tau_{\rm ac}` is 4.0 to 15.6 code time units. Every tracked window spans
only 5.5 to 27 :math:`\tau_{\rm ac}`, so it holds

.. list-table::
   :header-rows: 1

   * - quantity
     - range across the 16 traces
   * - :math:`\tau_{\rm ac}`
     - 4.0 -- 15.6
   * - window length in :math:`\tau_{\rm ac}`
     - 5.5 -- 26.9
   * - **independent samples** :math:`n_{\rm eff}`
     - **2.6 -- 11.8**
   * - error-bar understatement :math:`\sqrt{n/n_{\rm eff}}`
     - **2.0x -- 3.7x**

Four of sixteen windows are shorter than ten correlation times. The consequence
is not subtle: every one of these windows reports a standard error computed as
:math:`\sigma/\sqrt{n}` over *correlated* samples, so **the published flux error
bars are understated by 2 to 3.7 times**, and the worst case averages fewer than
three independent samples.

That reframes the gradient blocker. ``gradient_uncertainty_rel = 1.806`` was
propagated from those understated error bars; corrected for correlation it is
larger still. The gap to the 0.5 gate is therefore wider than 13x of extra
sampling -- which strengthens rather than weakens the case for changing method.

**N2 -- Gradient-divergence curve. BLOCKED on the integrator, with the cause
diagnosed.** ``tools/campaigns/nonlinear_gradient_window.py`` is written, runs on
GPU, and refuses to report a result -- correctly.

The measurement needs a *saturated* trajectory. The tool integrates with a
fixed-step RK2, and on the reduced Cyclone case that never saturates: over
``t = 600`` the fluctuation energy grew by :math:`3.3\times 10^{49}`, drift over
the final quarters was 100%, and the saturation gate returned **NOT SATURATED**.
The gradient ladder over :math:`N = 1 \dots 1024` came out *linear* in :math:`N`
(each doubling roughly doubles the gradient) rather than exponential -- the
signature of unbounded linear growth, not of a chaotic adjoint.

The cause is the time step, not the physics. The shipped nonlinear TOML sets
``fixed_dt = false`` with ``dt_max = 0.05``: the :math:`E\times B` nonlinearity
imposes a CFL condition that tightens as the amplitude grows, so a fixed-step
scheme that is stable during the linear phase goes unstable once the
nonlinearity would otherwise bite. Forcing a fixed step removed exactly the
mechanism that saturates the run.

The fix is to separate the two phases, which is also what the windowed-adjoint
literature does: reach saturation with the production adaptive integrator, then
unroll a *short* fixed-step window from that state for differentiation. Adaptive
stepping is awkward to differentiate because the step count varies with the
parameter; a fixed-step window avoids that, and only needs to be stable over the
window rather than over the whole trajectory.

A second constraint is memory. Reverse mode through the window OOM'd a 16 GB
card at ``N = 2048`` even with ``jax.checkpoint``; XLA reported it could not get
below 5.3 GiB. ``N = 1024`` fits at a ``16^3`` grid. Any production scheme has to
budget for this rather than discover it.

**N3 -- Bias against finite differences, inside the window.** At
:math:`N` below the knee, compare the windowed adjoint with the existing central
FD artifact on the same coefficient. Report the ratio, not just the error:
[iGENE2026]_ found 15--34%, and a similar ratio for GKX would be a
*reproduction*, not a failure.

**N4 -- Does a biased gradient descend?** The decisive test. Run a short descent
on one coefficient with the windowed adjoint and check the nonlinear flux
actually falls, measured with the N1 protocol. A gradient that is 30% of truth
but correctly signed and low-variance beats an unbiased one with 180%
uncertainty.

**N5 -- Shadowing, only if N4 fails.** NILSS on the smallest tracked case.
Deferred deliberately: it is a large build, and N4 may make it unnecessary.

Validation gates
----------------

**G1** :math:`\tau_{\rm ac}` is computed per case with the definition and the
window length in units of :math:`\tau_{\rm ac}` recorded in the artifact.

**G2** The divergence knee from N2 is reported with the growth rate of the
adjoint norm beyond it, and compared against the leading Lyapunov exponent
estimated from the same traces.

**G3** Any adopted gradient reports its bias against FD *and* its variance.
A bias-free claim requires shadowing, not a windowed adjoint.

**G4** No promotion without N4: a descent that moves the flux, on the same
averaging protocol used to certify the flux.

**G5** Every window used anywhere in the campaign is at least a stated multiple
of :math:`\tau_{\rm ac}`, replacing the current fixed-time convention.

What this changes about the existing plan
-----------------------------------------

The current campaign is trying to make finite differences converge. The
literature says that costs 13x more sampling than GKX has, and that the field's
working alternative accepts a biased gradient instead. The plan therefore stops
buying longer FD windows and starts measuring the window the physics allows.

The one thing worth keeping unchanged: the matched-perturbation, post-transient
averaging protocol behind the existing gates. It is what makes N3 and N4
interpretable.

References
----------

.. [iGENE2026] "iGENE: A Differentiable Flux-Tube Gyrokinetic Code in
   TensorFlow", arXiv:2605.03086. https://arxiv.org/abs/2605.03086

.. [Wang2014] Q. Wang, R. Hu & P. Blonigan, "Least Squares Shadowing
   sensitivity analysis of chaotic limit cycle oscillations", *J. Comput. Phys.*
   **267**, 210-224 (2014). https://arxiv.org/abs/1204.0159

.. [Ni2017] A. Ni & Q. Wang, "Sensitivity analysis on chaotic dynamical systems
   by Non-Intrusive Least Squares Shadowing (NILSS)", *J. Comput. Phys.* **347**,
   56-77 (2017).

.. [Blonigan2018] P. Blonigan & Q. Wang, "Multiple shooting shadowing for
   sensitivity analysis of chaotic dynamical systems", *J. Comput. Phys.* **354**,
   447-475 (2018).
