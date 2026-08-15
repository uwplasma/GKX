Nonlinear turbulence gradients
==============================

Status: **bounded-memory windowed adjoint implemented; promotion tests still
pending.** The memory wall and physical divergence window are now measured, but
the finite-difference bias and descent gates below still prevent calling this a
production stellarator gradient.

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

Algorithm selection
-------------------

The nonlinear run is a chaotic trajectory, not a converged residual
:math:`F(x,p)=0`. An implicit root adjoint would differentiate a steady root that
this problem does not have. A continuous backsolve adjoint would also change the
gradient: stellarator optimization needs the derivative of GKX's actual discrete
RK map, including its projection and field solves. The implementation therefore
uses a **checkpointed discrete adjoint** [JAXCheckpointing]_ [Revolve2000]_.

Checkpointing only the scan step rematerializes its local intermediates but
still leaves all :math:`N` scan carries on the reverse tape. GKX now splits the
trajectory into blocks of length :math:`B`, rematerializes each step and each
block, and retains only the block boundaries plus one block's reverse state:

.. math::

   M(B) = O(N/B + B), \qquad B = \lceil\sqrt{N}\rceil,
   \qquad M = O(\sqrt{N}).

This is reverse-tape state storage; requested time-series outputs retain their
inherent :math:`O(N)` result storage. The schedule does not alter a time step or
the returned derivative. A recursive binomial schedule can reduce storage to
:math:`O(\log N)` [DiffraxAdjoints]_, but incurs :math:`O(N\log N)`
recomputation. The measured physical knee below is only
:math:`N=1024`--2048, where the square-root schedule already fits comfortably
and has the smaller recomputation burden.

``tools/profiling/profile_nonlinear_adjoint_checkpointing.py`` compiles the same
production nonlinear RHS and checks primal/adjoint parity before reporting XLA
memory. At :math:`N=2048` in complex64:

.. list-table:: Reverse-mode checkpoint profile
   :header-rows: 1

   * - device / state
     - policy
     - XLA temporary memory
     - warmed runtime
   * - CPU, 98 kB state
     - per-step
     - 759 MB
     - 4.57 s
   * - CPU, 98 kB state
     - blocked
     - **12.6 MB (60.1x lower)**
     - 7.03 s (1.54x)
   * - RTX A4000, 1.57 MB state
     - per-step
     - 11.88 GB
     - 3.42 s
   * - RTX A4000, 1.57 MB state
     - blocked
     - **168 MB (70.5x lower)**
     - 5.71 s (1.67x)

The raw CPU and GPU artifacts are
``docs/_static/nonlinear_adjoint_checkpointing_cpu32.json`` and
``nonlinear_adjoint_checkpointing_gpu32.json``. The production saturated-state
ladder below runs in complex128 and now completes at :math:`N=2048` on the same
16 GB GPU.

**N2 -- Gradient-divergence curve. MEASURED, including the knee.**
``tools/campaigns/nonlinear_saturated_state.py`` reaches saturation with the
production CFL-adaptive stepper (all five checks pass:
:math:`\tau_{\rm ac} = 8.71`, window 23.0 :math:`\tau_{\rm ac}`, late drift
1.9%), and ``nonlinear_gradient_window.py`` differentiates a fixed-step window
from that state. The requested ``Nx=Ny=Nz=16`` linked-grid override retains
``ntheta=24``, so the actual spectral state is ``16 x 16 x 24``:

.. list-table:: :math:`|dE/d(\text{drive scale})|` from a verified-saturated state
   :header-rows: 1

   * - :math:`N`
     - :math:`t`
     - :math:`t/\tau_{\rm ac}`
     - gradient
     - ratio to previous
   * - 64
     - 2.49
     - 0.29
     - 1.854e-01
     - 2.190
   * - 128
     - 4.98
     - 0.57
     - 4.040e-01
     - 2.179
   * - 256
     - 9.96
     - 1.14
     - 8.353e-01
     - 2.068
   * - 512
     - 19.92
     - 2.29
     - 1.673e+00
     - 2.003
   * - 1024
     - 39.84
     - 4.58
     - 2.297e+00
     - 1.373
   * - 2048
     - 79.68
     - 9.15
     - **4.904e+01**
     - **21.349**

The gradient grows nearly linearly through :math:`N=512`, then decorrelates at
:math:`N=1024`. At :math:`N=2048` it jumps by 21.3x. An exponential fit is
better than a power law on the fitted tail (residual 0.408 versus 0.642), with
rate :math:`0.0639` per code-time unit. Thus the usable initial-value adjoint
window is now bracketed between **4.58 and 9.15 correlation times** for this
case. This is longer than [iGENE2026]_'s roughly one-correlation-time window, but
it is not an unlimited long-time derivative.

Two caveats remain:

* The objective is a state-norm proxy, not the production heat flux. It shares
  the trajectory's Lyapunov behaviour but is not the quantity being optimized.
* One case, one resolution, one drive parameter.

The implementation also has a direct mathematics/physics gate: on a nonlinear
Cyclone trajectory, reverse mode through the blocked scan differentiates the
production heat-flux kernel and agrees with centered finite differences on both
CPU and CUDA. The lower-level test covers a non-square step count, so its tail
block and both primal and reverse results are checked independently.

Hermite--Laguerre structure and SOLVAX
---------------------------------------

The Hermite--Laguerre recurrences are valuable, but they do not make the full
nonlinear Jacobian block tridiagonal. Streaming, magnetic drifts, and collisions
couple nearby :math:`m` and :math:`\ell` orders [Mandell2018]_; GX implements
that neighborhood as a local stencil. The nonlinear :math:`E\times B` bracket,
however, transforms Laguerre coefficients to a velocity grid and performs a
Fourier convolution [GX2022]_. Its Jacobian is therefore global in perpendicular
wavenumber.

SOLVAX's block-Thomas and transposed block solves are consequently not an exact
inverse for this trajectory adjoint. They remain the right building blocks for
a **preconditioner** if N3/N4 require NILSAS or multiple-shooting shadowing:
retain the local Hermite--Laguerre linear block in the preconditioner and apply
the nonlinear convolution matrix-free. SOLVAX already exposes the needed
transposed block solve and Newton--Krylov interfaces, so no SOLVAX change is
justified at this stage.

**Precision note, found here and applicable well beyond this measurement.** The
production stepper sets its state dtype with ``result_type(G0, complex64)``, so a
single-precision seed pins the whole trajectory to complex64 **even with
JAX_ENABLE_X64 set**. The saved saturated state came back single precision and
only surfaced because ``lax.scan`` rejected the mismatched carry against an x64
RHS. Whether the tracked nonlinear production results are themselves single
precision is worth checking on its own; it would affect every nonlinear number in
the repository, not just this one.

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

**N5 -- Shadowing, only if N4 fails.** NILSAS on the smallest tracked case.
Its adjoint cost is independent of the number of stellarator design parameters
[NILSAS2019]_. Deferred deliberately: it is a large build, and N4 may make it
unnecessary.

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

.. [NILSAS2019] A. Ni, "Sensitivity analysis on chaotic dynamical systems by
   Non-Intrusive Least Squares Adjoint Shadowing (NILSAS)", arXiv:1801.08674.
   https://arxiv.org/abs/1801.08674

.. [Revolve2000] A. Griewank & A. Walther, "Algorithm 799: Revolve: an
   implementation of checkpointing for the reverse or adjoint mode of
   computational differentiation", *ACM TOMS* **26**, 19-45 (2000).
   https://doi.org/10.1145/347837.347846

.. [JAXCheckpointing] JAX documentation, "Gradient checkpointing".
   https://docs.jax.dev/en/latest/gradient-checkpointing.html

.. [DiffraxAdjoints] Diffrax documentation, "Adjoints".
   https://docs.kidger.site/diffrax/api/adjoints/

.. [Mandell2018] N. R. Mandell, W. Dorland & M. Landreman,
   "Laguerre-Hermite pseudo-spectral velocity formulation of gyrokinetics",
   *J. Plasma Phys.* **84** (2018). https://arxiv.org/abs/1708.04029

.. [GX2022] N. R. Mandell et al., "GX: a GPU-native gyrokinetic turbulence
   code for tokamak and stellarator design", arXiv:2209.06731.
   https://arxiv.org/abs/2209.06731
