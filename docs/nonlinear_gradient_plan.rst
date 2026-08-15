Nonlinear turbulence gradients
==============================

Status: **physical heat-flux windowed adjoint and a local VMEC-state descent are
implemented; a stationary equilibrium-boundary gradient is not promoted.** The
discrete RK3 derivative is FD-verified through 2048 steps, the divergence knee
is measured, and a QH state coefficient descends over one autocorrelation time.
Three independent long-window finite-difference anchors still disagree in sign,
so they cannot certify the corresponding infinite-time derivative.

Why the current approach is stuck
---------------------------------

GKX's earlier turbulence-gradient evidence used central finite differences over
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

Before this work GKX had no autocorrelation tooling, even though that quantity
sets the usable window. Every downstream choice depends on it, so it came first.

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

That reframes the original gradient blocker. ``gradient_uncertainty_rel = 1.806`` was
propagated from those understated error bars; corrected for correlation it is
larger still. The gap to the 0.5 gate is therefore wider than 13x of extra
sampling -- which strengthens rather than weakens the case for changing method.

The new production continuation experiment makes the point directly. Three
independent saturated anchors, each evaluated at drive scales 0.98 and 1.02 for
200 time units, give

.. math::

   -526.8\pm162.4,\qquad +263.9\pm155.0,\qquad +525.6\pm181.7.

Their equally weighted ensemble is :math:`+87.6\pm316.3` (:math:`z=0.28`), two
of six component windows fail convergence, and the signs are inconsistent.
``nonlinear_stationary_heat_flux_fd_ensemble.json`` therefore fails closed.
Long-window FD is useful as a holdout, but it is not a usable optimizer gradient
at this cost.

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

**N2 -- Physical heat-flux gradient-divergence curve. DONE, including the
knee.** ``tools/campaigns/nonlinear_saturated_state.py`` reaches saturation with
the production CFL-adaptive stepper (all five checks pass:
:math:`\tau_{\rm ac}=8.705`, window 23.0 :math:`\tau_{\rm ac}`, late drift
1.9%). ``nonlinear_gradient_window.py`` then differentiates the production
heat-flux average through exactly the RK3 map recorded in that state. The
``Nx=Ny=16`` override retains ``Nz=24`` and the production four-Laguerre,
eight-Hermite basis:

.. list-table:: Physical :math:`d\langle Q\rangle/d(\text{drive scale})`
   :header-rows: 1

   * - :math:`N`
     - :math:`t/\tau_{\rm ac}`
     - :math:`\langle Q\rangle`
     - gradient
     - ratio to previous
     - AD/FD relative error
   * - 64
     - 0.286
     - 98.928
     - 40.432
     - --
     - 2.5e-11
   * - 128
     - 0.572
     - 92.980
     - 68.483
     - 1.69
     - 1.1e-11
   * - 256
     - 1.144
     - 86.296
     - 102.590
     - 1.50
     - 9.1e-12
   * - 512
     - 2.288
     - 88.958
     - 153.505
     - 1.50
     - 3.1e-11
   * - 1024
     - 4.577
     - 95.713
     - 183.253
     - 1.19
     - 2.6e-9
   * - 2048
     - 9.153
     - 117.616
     - **3934.660**
     - **21.47**
     - 2.5e-5

The tangent stays bounded enough for optimization through :math:`N=1024`, then
jumps by 21.5x at :math:`N=2048`. An exponential fits the tail better than a
power law (residual 0.419 versus 0.681), with rate :math:`0.0575` per code-time
unit. The usable initial-value adjoint window is therefore bracketed between
**4.58 and 9.15 correlation times** for this case. This is an actual heat-flux
derivative, not the earlier state-energy proxy. Its remaining scope limitation
is one case, resolution, and drive parameter. The machine-readable result is
``docs/_static/nonlinear_heat_flux_gradient_window_rk3.json``.

The mathematics/physics gate checks the exact discrete derivative three ways:
blocked and plain scans agree on a non-square step count, centered FD agrees at
every production ladder rung, and the same tests run on CPU and CUDA. The
recorded integration-method check refuses to differentiate an RK3 saturation
state with an RK2 window.

The VMEC/Boozer path exposed one additional AD singularity at the zero
perpendicular-wavenumber mode. Directly evaluating
:math:`J_0(\sqrt{x})` and :math:`J_1(\sqrt{x})/\sqrt{x}` makes the primal finite
at :math:`x=0` but leaves a ``sqrt``/division tangent of ``NaN``. GKX now uses
the analytic small-:math:`x` series

.. math::

   J_0(\sqrt{x}) = 1-\frac{x}{4}+\frac{x^2}{64}-\frac{x^3}{2304},
   \qquad
   \frac{J_1(\sqrt{x})}{\sqrt{x}} =
   \frac12-\frac{x}{16}+\frac{x^2}{384}-\frac{x^3}{18432}.

The unit gate checks both limits and their exact derivatives, :math:`-1/4` and
:math:`-1/16`, plus a full cache geometry JVP containing the zero mode.

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
a **preconditioner** in future production NILSAS or multiple-shooting work:
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

**N3 -- Bias against stationary finite differences. UNRESOLVED, with negative
evidence preserved.** Inside any fixed window, the discrete adjoint and centered
FD are the same derivative and agree to at worst :math:`2.5\times10^{-5}`. That
is a correctness test, not a bias measurement. Bias means comparison with the
stationary response. The three-anchor result above has inconsistent signs and
:math:`z=0.28`, so no meaningful adjoint/FD ratio exists yet. Reporting a ratio
against any single anchor would select noise after seeing it. For completeness,
the raw :math:`N=1024` magnitude ratios are 0.348, 0.694, and 0.349; the first
anchor has the opposite sign from the adjoint and from the other two anchors.

**N4 -- Does a physical stellarator gradient descend? PARTIAL PASS.** The
QH ``nfp4_QH_warm_start`` chain now differentiates

``VMEX state coefficient -> Boozer geometry -> GKX cache -> projected RK2 map
-> physical heat flux``.

For ``Rcos_mid_surface_m1``, the descent direction persists from 16 steps
(:math:`1.096\tau_{\rm ac}`) to 32 steps
(:math:`2.192\tau_{\rm ac}`). At 32 steps reverse AD gives
:math:`9.6980581\times10^5` and centered FD gives
:math:`9.6980032\times10^5` (relative error :math:`5.7\times10^{-6}`). A local
negative-gradient line search lowers the window heat flux from 26.0283 to
25.7021 (1.253%) while limiting the maximum normalized sampled-geometry change
to 4.58%. The artifacts ``vmec_boozer_nonlinear_window_descent_n16.json`` and
``vmec_boozer_nonlinear_window_descent_n32.json`` pass all three gates.

This is deliberately a partial claim: it perturbs one internal equilibrium
state coefficient without re-solving the fixed-boundary equilibrium, starts all
candidates from the detached base turbulent state, and measures a finite window.
It proves the physical solver chain and a local descent direction, not a
stationary boundary-shape gradient. Independently, GKX's existing replicated
QA optimized-equilibrium holdout records an 18.44% nonlinear heat-flux reduction
at 7.82 standard errors, while the QA quasisymmetry/QL weight scan contains
several points that improve both metrics. Those results support the optimization
strategy [Jorge2023]_ but were not produced by this new derivative.

**N5 -- Matrix-free shadowing. PROTOTYPE IMPLEMENTED, not promoted.** GKX now
has two real-state discrete-map pilots:

* discrete NILSAS follows Appendix B of [NILSAS2019]_, recomputes one trajectory
  segment at a time, uses VJPs only, and solves a reduced constrained problem;
* multiple-shooting shadowing applies both the constraint and its transpose by
  segment-map JVPs/VJPs and solves the regularized Schur system by matrix-free
  CG. Its adjoint form returns all design-parameter derivatives from one solve.

Neither assembles a state Jacobian. Complex spectral states are packed as real
and imaginary parts so QR and least-squares coefficients remain real. A
matrix-free Benettin QR iteration [Benettin1980]_ measures the leading Lyapunov
exponents and therefore the minimum NILSAS homogeneous-adjoint count instead of
guessing it. Continuous-time neutral projection/time dilation is not yet
implemented, so neither pilot is an infinite-time gyrokinetic-gradient claim.

On the verified-saturated ``8 x 8 x 24`` RK3 case, the four-vector spectrum is
:math:`[+0.0839,-0.0278,-0.0296,-0.0019]` per time unit: one resolved positive
exponent and one nearly neutral direction. At the short 32-step horizon,
NILSAS with 1, 2, and 4 homogeneous adjoints gives 2.35761, 2.35743, and
2.35727, versus 2.35766 from finite-window AD, with constraint residuals below
:math:`3\times10^{-16}`. The one-adjoint horizon ladder is

.. list-table:: Reduced RK3 NILSAS pilot
   :header-rows: 1

   * - :math:`N`
     - :math:`t/\tau_{\rm ac}`
     - finite-window AD
     - NILSAS
     - KKT condition
   * - 32
     - 0.129
     - 2.35766
     - 2.35761
     - 1.32e2
   * - 512
     - 2.067
     - 32.8397
     - 32.8449
     - 2.85e4
   * - 1024
     - 4.134
     - 44.6532
     - 44.7934
     - 8.59e4
   * - 2048
     - 8.268
     - 69.4630
     - 69.4384
     - 2.86e5

The corresponding artifacts are ``nonlinear_shadowing_rk3_n32_pilot.json`` and
``nonlinear_nilsas_rk3_n{512,1024,2048}_pilot.json`` in ``docs/_static``.

At :math:`N=2048`, one-adjoint NILSAS takes 55.2 s versus 57.1 s for the
blocked finite-window adjoint and retains a constraint residual of
:math:`8.0\times10^{-14}`. This reduced case does not exhibit the production
case's sharp gradient knee, so agreement is encouraging but not shadowing
validation. The regularized MSS pilot converges to a :math:`9.99\times10^{-6}`
normal residual in 97 CG iterations and 152 s, but gives :math:`-0.296` rather
than :math:`+2.358` at the short horizon. Until regularization convergence,
neutral-time treatment, and the partial-SVD block preconditioner of
[Shawki2019]_ are added,
that sign disagreement is negative evidence.

The resulting method choice is deliberate: use the physical windowed adjoint
with an explicit local line-search gate now; advance one-adjoint NILSAS on the
production state as the long-time research path; do not put the present MSS
estimate in an optimizer.

Validation gates
----------------

**G1** :math:`\tau_{\rm ac}` is computed per case with the definition and the
window length in units of :math:`\tau_{\rm ac}` recorded in the artifact.

**G2** The divergence knee from N2 is reported with the growth rate beyond it.
Finite-time Lyapunov exponents are reported with their own grid, state, and
horizon; numerical agreement is claimed only when those match the ladder.

**G3** Any adopted gradient reports its bias against FD *and* its variance.
A bias-free claim requires shadowing, not a windowed adjoint.

**G4** A finite-window descent is labelled as such. No stationary
boundary-gradient promotion without a re-equilibrated candidate and a matched
long-window holdout.

**G5** Every window used anywhere in the campaign is at least a stated multiple
of :math:`\tau_{\rm ac}`, replacing the current fixed-time convention.

What this changes about the existing plan
-----------------------------------------

The campaign no longer tries to make one central difference do three jobs.
Exact windowed AD supplies inexpensive design directions; replicated long
continuations remain independent holdouts; NILSAS is the next long-time method
to qualify. Regularized MSS is retained as an algorithm comparison, but the
unpreconditioned solve is presently slower and its gradient is not robust to the
missing neutral treatment.

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
   447-475 (2018). https://arxiv.org/abs/1704.02047

.. [Shawki2019] K. Shawki & G. Papadakis, "A preconditioned multiple shooting
   shadowing algorithm for the sensitivity analysis of chaotic systems",
   *J. Comput. Phys.* **398**, 108861 (2019).
   https://arxiv.org/abs/1810.12222

.. [NILSAS2019] A. Ni, "Sensitivity analysis on chaotic dynamical systems by
   Non-Intrusive Least Squares Adjoint Shadowing (NILSAS)", arXiv:1801.08674.
   https://arxiv.org/abs/1801.08674

.. [Benettin1980] G. Benettin, L. Galgani, A. Giorgilli & J.-M. Strelcyn,
   "Lyapunov characteristic exponents for smooth dynamical systems and for
   Hamiltonian systems; a method for computing all of them. Part 1: Theory",
   *Meccanica* **15**, 9-20 (1980).

.. [Jorge2023] R. Jorge et al., "Direct Microstability Optimization of
   Stellarator Devices", *J. Plasma Phys.* **89** (2023).
   https://arxiv.org/abs/2301.09356

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
