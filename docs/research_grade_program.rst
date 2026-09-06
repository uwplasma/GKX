Research decisions and their evidence
=====================================

Planning review, 2026-09-06. The repository's
`plan.md <https://github.com/uwplasma/GKX/blob/plan/focused-research-20260906/plan.md>`_
owns the execution queue. This page explains choices, not new validation
results. Current claims are in :doc:`research_grade_plan`.

Optimization: different derivatives
-----------------------------------

For design parameters :math:`p`, projected RK step :math:`\Phi_{\Delta t,p}`,
and detached turbulent state :math:`G_*`,

.. math::

   G_0=\operatorname{stopgrad}(G_*),\qquad
   G_{n+1}=\Phi_{\Delta t,p}(G_n),\qquad
   J_N(p;G_*)=\frac{1}{N-N_b}\sum_{n=N_b+1}^{N}Q(G_n,p).

The discrete adjoint computes :math:`\partial_p J_N`. It does not include
the parameter dependence of the turbulent measure:

.. math::

   \bar Q(p)=\int Q(G,p)\,d\mu_p(G),\qquad
   \frac{d\bar Q}{dp}
   =\int \partial_p Q\,d\mu_p+\int Q\,d(\partial_p\mu_p).

Long tangent-map products can grow despite a bounded observable.
AD/finite-difference agreement at fixed initial state verifies implementation,
not the derivative of the statistical objective.
**Retain checkpointed reverse mode**, choosing the physical horizon from
directional usefulness and gradient variance. Checkpointing solves memory,
not chaotic sensitivity. Implicit eigenvalue/VMEX derivatives remain appropriate
for their different equations.

`iGENE (2026) <https://arxiv.org/html/2605.03086v1>`_ reports useful truncated
gradients from saturated states, but also divergence and imperfect agreement with
long-run finite differences. This preprint motivates descent tests, not an
unbiased-gradient claim.
`Acton et al. (2024) <https://arxiv.org/abs/2403.12621>`_ verifies a linear
microstability adjoint; do not generalize its eigenvalue argument to turbulence.

Warm in-memory continuation
---------------------------

**Already present:** ``SaturationWarmStart`` stores distribution and geometry
signature; QA refreshes between VMEX stages and sets ``max_reuse=0``.
NumPy conversions can move its state to the host.
The prepared object's generic ``value_and_grad`` is only a JAX wrapper,
not a general equilibrium-to-transport parameter-update contract.

.. math::

   G_*^{k+1}=\Phi_{p_{k+1}}^{\,N_{\rm relax}}
      (\mathcal T_{k\rightarrow k+1}G_*^k),\qquad
   \phi^{k+1}=\mathcal F(G_*^{k+1},p_{k+1}).

:math:`\mathcal T` is identity only for compatible basis, coordinates,
normalization and species. Old potential does not satisfy new quasineutrality.

1. Keep distribution on-device; host cache/acceptance decisions stay outside AD.
2. Freeze seed ensemble and numerical topology across trial evaluations.
   Rejected trials must not mutate accepted state.
3. Preserve nonlinear amplitude; never apply linear-mode renormalization.
4. Cold restart incompatible topology/species/basis. Add conservative projection
   only if its cost and accuracy justify it.
5. AD/FD checks use the same detached state; re-equilibration and seed bias
   are distinct from derivative implementation error.
6. Validation seeds are independent of optimization. Timestep repeats of one
   seed are numerical controls, not independent physical samples.

**Experiment:** cold/warm, two relaxation budgets, small/larger shape changes,
three seeds. Report bias/uncertainty, directional agreement and total time.
The existing 5% geometry-distance and quarter-spin-up policy is heuristic.
Adopt only if equal-accuracy cost decreases.

`Kim et al. (2024) <https://arxiv.org/html/2310.18842v2>`_ optimized nonlinear
stellarator turbulence with stochastic finite-difference information.
Their radial/field-line checks and imperfect linear-proxy ranking motivate
held-out geometry validation and an SPSA control, not a claim that GKX's warm
policy is validated.

How much saturation is enough?
------------------------------

``Wphi`` is not heat flux. A quiet five-time-unit segment is insufficient if it
samples less than a decorrelation cycle. Later levels in the supplied trace
move again; the excerpt has no heat-flux history for transport uncertainty.

For uniform samples of stationary heat flux,

.. math::

   \tau_{\rm int}=\Delta t_s\left(\frac12+\sum_{k=1}^{K}\rho_Q(k)\right),
   \quad N_{\rm eff}\simeq\frac{N\Delta t_s}{2\tau_{\rm int}},
   \quad {\rm SE}(\bar Q)\simeq\frac{s_Q}{\sqrt{N_{\rm eff}}}.

Main estimates IAT and tests flux plus optional ``Wphi/Wg`` guards.
Its first-median-crossing window, median spacing and unweighted mean are not
general burn-in or nonuniform-time estimators. Near-zero/white-noise rejection
guards dead runs but can overrun stable or rapidly decorrelating cases.

- Separate transient, insufficient-data, stationary-turbulence and
  negligible-transport outcomes.
- Use physical-time sampling or a tested weighted estimator; audit restart
  timestamps and chunk boundaries.
- Require resolved batches, minimum horizon and per-trace drift checks;
  no selection of one quiet interval.
- Predeclare absolute/relative precision tolerances. Near zero needs an
  absolute criterion.
- Calibrate sequential false stops on correlated/drifting synthetic traces
  and archived physical traces replayed at chunk boundaries.
- Select burn-in on pilot data; confirm on an independent fixed window.
  Repeated peeking does not preserve a nominal fixed-time 95% interval.
- Do not count timestep/resolution repeats as independent seeds.

`Vats, Flegal & Jones (2019) <https://arxiv.org/abs/1512.07713>`_ supplies
covariance/ESS and sequential-stopping theory, under CLT/stationarity assumptions.
It is not automatically a theorem for deterministic GK turbulence.
Start with scalar batch means and energy guards; multivariate diagnostics are
a comparison, not another required framework.

Resolution and coordinate choices
----------------------------------

A rising flux spectrum at the cutoff is a warning even with smooth potential:
flux is a cross-moment correlation. Larger :math:`N_y` at fixed :math:`L_y`
raises the cutoff. Smaller :math:`L_y` at fixed :math:`N_y` also does, but
removes long wavelengths. Record box, dealiased cutoffs, bin widths and units.

.. list-table:: Coordinate decisions
   :header-rows: 1
   :widths: 25 35 40

   * - Option
     - Potential benefit
     - Decision
   * - Field-aligned, Hermite–Laguerre
     - Sparse recurrences, local FFTs, existing AD
     - Retain; compare resolution-to-error, not point counts.
   * - Equal arclength
     - Uniform parallel resolution; potentially lower streaming CFL
     - Compare for uneven grids; preserve weights, transformed metrics,
       boundaries and geometry derivatives.
   * - Generalized twist-and-shift
     - Fewer perpendicular box constraints at low shear
     - Audit existing selection, symmetry and domain assumptions first.
   * - Non-twisting flux tube
     - Less shear distortion on long tubes
     - Reconsider only with measured high-shear/domain cost.
   * - Energy–pitch / bounce-adapted grid
     - Resolve weakly collisional trapped/passing structure
     - Deferred; compare moments with GS2 first. Changes complicate collisions,
       quadrature and differentiable topology.
   * - FCI / global model
     - Islands, separatrices, nonlocal profiles
     - Separate scope, not the next nested-surface QA calculation.

Sources: `Martin et al. (2018) <https://arxiv.org/abs/1803.09049>`_,
`Ball & Brunner (2021) <https://arxiv.org/abs/2012.04785>`_,
`GS2 <https://gyrokinetics.gitlab.io/gs2/>`_,
`GENE-X <https://genex.ipp.mpg.de/about>`_ and its
`2026 stellarator extension <https://www.sciencedirect.com/science/article/pii/S0010465526001207>`_.
None implies a universally better coordinate system.

ESSOS Biot–Savart fields/field-line derivatives are not automatically a valid
local equilibrium: require metric, divergence, periodicity and locality checks.
Islands invalidate a general surface label; a field-line plot is not island
transport. Finite-beta VMEX equilibrium and electromagnetic fluctuations are
different switches, with pressure-gradient consistency still required.

Boundary damping: adjudicate, do not overfit
--------------------------------------------

`Published GX, section 4.2 and appendix C <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gx-a-gpunative-gyrokinetic-turbulence-code-for-tokamak-and-stellarator-design/2C4BB81955E7E749B95B8B8141E997FA>`_
uses :math:`A=0.1/\Delta t`; its kinetic-electron CBC scan finds flux insensitive
above approximately :math:`A\Delta t=0.05`.
The `v3 preprint <https://arxiv.org/html/2209.06731v3>`_ instead describes a
velocity/domain-scaled rate. Cite the version actually reproduced.

A numerical absorber need not be a physical collision rate.
Changing timestep with :math:`A\Delta t` fixed changes its continuous rate.
Eigenproblems need an explicit operator convention, not a hidden adaptive step.

**Experiment:** reproduce the affected scan with legacy per-step damping,
then hold a matched rate fixed at half timestep. Repeat a width/domain
refinement at one problematic mode. Keep fields, linked chains and closure fixed.
Compare frequency, eigenfunction localization and interior flux.
Only then spend on nonlinear references; do not tune damping to one eigenvalue.

Local GX clamps a linked-map launch dimension. Prior scratch repairs/outputs
need checking before high-resolution GX runs become truth.
A reference-code defect does not prove GKX correct.

Preconditioners: qualify existing structure
-------------------------------------------

Main already has Hermite-line/field-corrected shifted preconditioners and
SOLVAX GMRES. For a frozen distribution/field step,

.. math::

   \begin{pmatrix}D&U\\V&F\end{pmatrix}
   \begin{pmatrix}G\\\varphi\end{pmatrix}
   =\begin{pmatrix}r_G\\r_\varphi\end{pmatrix},\qquad
   S=F-VD^{-1}U.

Approximate the distribution inverse and field Schur complement;
evaluate residuals with the full operator. Streaming couples :math:`m\pm1`,
drifts :math:`m\pm2`, mirror Laguerre neighbors. Grouped moments yield useful
bands, but the nonlinear FFT/field/collision system is **not** block-tridiagonal.

`stella's algorithm <https://arxiv.org/html/1806.02162v1>`_ demonstrates implicit
streaming/acceleration and field response for kinetic electrons.
Compare when streaming limits GKX, not as a reason to replace efficient
explicit adiabatic-electron runs.

Measure setup/application, iterations, true and transpose residuals, memory,
useful timestep and geometry rebuilds. Preconditioners cannot accelerate an
explicit ExB-limited step directly. Move general primitives to SOLVAX only after
a GKX benchmark proves the need; defer mixed-precision refinement pending
conditioning/residual evidence.

Coulomb: remaining shipping contract
------------------------------------

``coulomb`` on main is a low-moment drift-kinetic projection.
``coulomb_finite_kperp`` selects like-species 8/18-moment tables, not general
multispecies/arbitrary-order Landau. Tests of one do not validate the other.

.. math::

   C^L_{ab}=C_{ab}(\delta f_a,F_{Mb})+C_{ab}(F_{Ma},\delta f_b).

Both test- and field-particle responses are required.
`Frei, Ball, Hoffmann, Jorge, Ricci & Stenger (2021) <https://arxiv.org/html/2104.11480v1>`_
distinguishes particle-coordinate conservation from finite-wavelength
gyrocenter diffusion. Zero density/momentum/energy rows at every finite
:math:`k_\perp` would not be a valid general repair.

.. list-table:: Gates before promotion
   :header-rows: 1
   :widths: 25 75

   * - Coefficients
     - Independent high-accuracy test/field blocks, signs and normalization.
       Reproduce branch corrections separately.
   * - Basis/range
     - Ordered :math:`(l,m)` list, not only the product. Refuse unsupported
       shapes, species, temperatures and wavelengths; no silent clamping.
   * - Invariants
     - Drift-kinetic limits, particle conservation, pair momentum/energy
       exchange, weighted symmetry/entropy for the specified equilibrium.
   * - Interpolation/AD
     - Values/JVPs off-grid, at zero/endpoints in f32/f64. Definiteness alone
       does not validate coupled field/source derivatives.
   * - Dynamics
     - Relaxation, collisional ITG and zonal response at several resolutions
       and frequencies; include stable/damped cases.
   * - Execution
     - Same model in spin-up, objective, diagnostics; unsupported routes refuse;
       report memory and timestep cost.

For an appropriate like-species/equal-temperature metric :math:`W`, test the
symmetric part of :math:`WC` against a norm-scaled roundoff bound.
Do not impose the same quadratic H-theorem on arbitrary unequal-temperature
linearized Landau equilibria without its assumptions.

**GENE is not one collision model.**
`Crandall et al. (2020) <https://doi.org/10.1016/j.cpc.2020.107360>`_
describes multispecies Sugama.
`Pan, Ernst & Crandall (2020) <https://www.osti.gov/servlets/purl/1799694>`_
describes exact linearized Landau and model comparisons.
Compare Coulomb to the latter or explicitly label different models;
do not tune Sugama zonal damping to match Coulomb.
Grid/moment representations require independent velocity convergence.

`von Boetticher et al. (2024) <https://www.osti.gov/pages/servlets/purl/2447402>`_
provides a Fokker–Planck model in stella.
`GS2 collisions <https://gyrokinetics.gitlab.io/gs2/lists/modules.html>`_
uses Barnes–Abel pitch scattering, energy diffusion and conservation corrections.
These are cross-checks, not interchangeable definitions of full Coulomb.

A narrow validated like-species envelope is acceptable when named as such.
Unlike-species/arbitrary-order support needs pair coefficients, mass/temperature
ratios and field tests; more wavelength nodes cannot provide it.
Defer table compression until independent correctness gates pass.

Benchmark design and current-code lessons
-----------------------------------------

`Hoffmann, Frei & Ricci's Dimits study <https://arxiv.org/abs/2308.01016>`_
provides a moment-based nonlinear threshold/convergence target.
Begin with turbulent CBC and a small threshold bracket.
Coarse nonlinear flux convergence does not establish convergence of a weakly
collisional linear high-wavenumber mode.

Manufactured forcing must be independent of the production RHS:
test RK order, spectral truncation decay and weighted invariants separately.
The old blanket "order within 10%" gate is inappropriate for all spectral axes.
`Abel et al. (2013) <https://arxiv.org/abs/1209.4782>`_ supplies ordering/free-energy
context; discrete tests must use GKX's weights and boundary fluxes.

- `stella at 2b8e269f <https://github.com/stellaGK/stella/tree/2b8e269f2addd0baa5991057eafa022135e04498>`_
  includes EM, full-flux-surface and restart tests beyond its older ES description.
  Learn term-isolation/restart tests without promising all regimes now.
- `gyaradax at 8d9dc2d2 <https://github.com/gerkone/gyaradax/tree/8d9dc2d205e8993ae9e43e6e1e82ec1ea2875234>`_
  separates no-I/O initialization/evolution and JAX/CUDA operations.
  Its `2026 preprint <https://arxiv.org/html/2604.06085v1>`_ motivates GKW
  comparisons. Many tests need separately supplied reference data;
  missing-data skips are not successful validation.
- GX is the nearest moment-method comparator; GS2/stella/GENE provide independent
  discretizations. Raw grid counts and unmatched hardware times are misleading.
- CUDA acceleration requires explicit JVP/VJP support. Fast forward execution
  alone does not establish a differentiable backend.

Fresh checks and limitations:
`review evidence <https://github.com/uwplasma/GKX/blob/plan/focused-research-20260906/plan/baseline/review_2026_09_06.md>`_.
No fresh GPU parity or saturated cross-code speed result is claimed here.
