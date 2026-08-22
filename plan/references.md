# Literature and code survey

Only primary papers and upstream project documentation are used for technical
decisions. Access was available for the sources below; no inaccessible paper is
currently blocking the audit.

## Gyrokinetics and moment representation

- Mandell, Dorland & Landreman, Laguerre--Hermite pseudo-spectral gyrokinetics,
  JPP 84 (2018), https://arxiv.org/abs/1708.04029
- Mandell et al., GX: a GPU-native gyrokinetic turbulence code,
  https://arxiv.org/abs/2209.06731
- GX upstream documentation and nonlinear example,
  https://gx.readthedocs.io/en/latest/Nonlinear.html
- stella upstream source, https://github.com/stellaGK/stella
- W7-X stella/GENE benchmark, https://arxiv.org/abs/2107.06060

Leverage: sparse Hermite/Laguerre recurrences; GX-compatible layout and GPU
comparisons; independent stellarator linear/nonlinear benchmarks. Do not infer
agreement from shared normalization or geometry files—compare transformed
operators and converged observables.

## Flux-tube boundary and perpendicular resolution

- Martin et al., generalized twist-and-shift boundary condition,
  https://doi.org/10.1088/1361-6587/aad38a
- Ball & Brunner, non-twisting flux tubes,
  https://arxiv.org/abs/2012.04785
- Sánchez et al., stellarator gyrokinetics across flux-tube, full-surface, and
  global domains, https://arxiv.org/abs/2106.02828
- Schekochihin et al., gyrokinetic phase-space cascade,
  https://arxiv.org/abs/0806.1069
- Morel et al., dynamic gyrokinetic large-eddy procedure,
  https://arxiv.org/abs/1110.0747
- Merlo et al., multiscale turbulence in stellarators,
  https://arxiv.org/abs/2508.06116

Leverage: boundary-condition tests must use real stellarator geometry;
perpendicular pile-up can be numerical and must be resolved before transport is
trusted. Sánchez et al. show that required flux-tube length is
configuration-dependent and that short W7-X tubes can disagree even between
field-line labels; scan both `alpha` and `npol` rather than treating `Nz` on one
tube as a complete parallel convergence test. LES-style closures are research
options only after direct-resolution baselines exist. Merlo et al. resolve
separate ion/electron scales to `ky*rho_s < 46` and find cross-scale changes to
zonal flows and transport. The present adiabatic-electron QA/QHS campaign is
therefore an ion-scale validation only; extending `Ny` through `ky*rho_i ~ 2`
tests its rising ion-scale tail, not kinetic-electron or multiscale convergence.

## Stochastic averages and stopping

- Oberparleiter et al., uncertainty estimation and a stopping rule in nonlinear
  gyrokinetic simulations,
  https://publications.lib.chalmers.se/records/fulltext/247070/local_247070.pdf
- Vaezi & Holland, quantifying temporal uncertainties of nonlinear turbulence
  simulations, https://arxiv.org/abs/1902.10879
- Rezaeiravesh et al., in-situ estimation of time-averaging uncertainties,
  https://arxiv.org/abs/2310.08676
- Papadopoulos et al., statistical analysis of stellarator gyrokinetic
  turbulence, https://arxiv.org/abs/2212.14219
- Jones et al., fixed-width output analysis for correlated Monte Carlo output,
  https://arxiv.org/abs/math/0601446
- Flegal & Jones, consistent batch-means and spectral-variance estimators,
  https://arxiv.org/abs/0811.1729
- Vats, Flegal & Jones, multivariate effective sample size and fixed-volume
  stopping, https://arxiv.org/abs/1512.07713
- Killick, Fearnhead & Eckley, exact penalized change-point detection with
  expected linear cost, https://arxiv.org/abs/1101.1438
- Yu et al., online change-point detection with false-alarm control,
  https://arxiv.org/abs/2006.03283

Leverage: remove burn-in by stationarity testing; account for autocorrelation;
use batches of several correlation times; guard late drift; quote uncertainty,
not raw output count. GKX should compare its IAT estimator against the paper's
five-correlation-time batch means.

The current executable is narrower: discard samples before the first crossing
of the full-prefix median, estimate first-zero Sokal IAT, require a window of at
least `10*tau_ac`, target 5% corrected relative SEM, and require half-window
stationarity in Q, Wphi, and Wg. Held-out replay rejects this burn-in selector,
so it is not yet a validated contract. Fixed-width theory supports sequential
termination only after a consistent long-run variance estimate; multivariate
fixed-volume theory motivates treating the three correlated diagnostics
together. A change-point detector is only a candidate burn-in estimator:
offline PELT has look-ahead, while online CUSUM assumptions and false-alarm
calibration must be tested on causal prefixes. Require persistence for at least
one additional independent batch and score every candidate against held-out
future data before changing the default.
Vaezi--Holland specifically warns that gyrokinetic flux uncertainty becomes
harder near the critical gradient, so SAT-1 must be checked across drive, not
only on the present `tprim=3` case. The low-memory in-situ ACF update is a
future bounded-memory option; adopt it only if it reproduces offline batch/IAT
uncertainties on GKX traces. Singular-spectrum/cluster analysis can flag
avalanches and regime changes in difficult held-out traces, but it is a review
diagnostic rather than the default stop rule until its added cost and false-stop
rate beat the stationary-suffix gate.

## Nonlinear convergence and validation hierarchy

- Frei et al., gyrokinetic Z-pinch turbulence with gyromoments and advanced
  collisions, JPP 89 (2023),
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gyrokinetic-simulations-of-plasma-turbulence-in-a-zpinch-using-a-momentbased-approach-and-advanced-collision-operators/037F92D5250416CA7E7EDAC5A0480FB0
- White, validation of nonlinear gyrokinetic transport models using turbulence
  measurements, JPP 85 (2019),
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/validation-of-nonlinear-gyrokinetic-transport-models-using-turbulence-measurements/D659391275AB4B71D37BEF8BB2241D45
- Mandell et al., GX GPU-native gyrokinetics, including nonlinear filter and
  parallel-resolution convergence studies,
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gx-a-gpunative-gyrokinetic-turbulence-code-for-tokamak-and-stellarator-design/2C4BB81955E7E749B95B8B8141E997FA
- Faber et al., stellarator microinstabilities and turbulence at low magnetic
  shear, https://doi.org/10.1017/S0022377818001022
- Chen et al., geometry effects on zonal-flow dynamics in optimized
  stellarators, https://arxiv.org/abs/2505.21886
- Tiwari et al., zonal-flow suppression in W7-X and QSTK,
  https://arxiv.org/abs/2501.12722

Leverage: a converged mean alone does not establish moment convergence or
dynamics; retain spectra and refine the Hermite--Laguerre tail around the
transport-carrying scales. Organize validation hierarchically: operator
identities, manufactured/numerical convergence, standard physics benchmarks,
code-to-code comparison, then stellarator transport with sensitivity and
synthetic diagnostics. GKX's promotion gate therefore requires both integrated
transport and resolved spatial/velocity spectra.
GX also makes the end-damping amplitude/width part of the convergence surface,
not an invisible fixed constant. Low-shear stellarators can require multiple
field-line turns to resolve saturation structures, so `npol=1`/`nperiod=1`
must be scanned with `Nz`; a one-turn `Nz` refinement alone is not a parallel
convergence proof.
The 2025 global studies independently find configuration-dependent nonlinear
zonal-flow suppression, especially in QH/QI cases. The QA/QHS/QI campaign must
therefore retain zonal/nonzonal `Phi2`, zonal-flow frequency, and residual
alongside total heat flux. Similar linear growth rates or stationary `Q` alone
cannot validate cross-configuration transport.

## Discrete adjoints and chaotic sensitivity

- Griewank & Walther, Revolve checkpointing,
  https://doi.org/10.1145/347837.347846
- JAX gradient checkpointing documentation,
  https://docs.jax.dev/en/latest/gradient-checkpointing.html
- Wang, Hu & Blonigan, least-squares shadowing,
  https://arxiv.org/abs/1204.0159
- Ni & Talnikar, non-intrusive least-squares adjoint shadowing,
  https://arxiv.org/abs/1801.08674
- Thakur & Nadarajah, stabilized-march adjoint shadowing,
  https://arxiv.org/abs/2505.00838
- Ni, fast adjoint algorithm for linear response of hyperbolic chaos,
  https://doi.org/10.1137/22M1522383
- Hickling et al., online gradient flow for statistical steady-state turbulent
  objectives, https://doi.org/10.1016/j.jcp.2025.114610
- Wang & Zaki, mitigating adjoint chaos in wall turbulence,
  https://arxiv.org/abs/2606.25399
- Acton et al., adjoint optimization of linear gyrokinetic microstability,
  https://doi.org/10.1017/S0022377824000709
- Artigues, Merlo & Jenko, iGENE: differentiable flux-tube gyrokinetics,
  https://arxiv.org/abs/2605.03086

Leverage: retain the finite discrete adjoint with rematerialization as GKX's one
production derivative of a declared finite window, not yet as a derivative of
the invariant-measure transport. Validate its useful horizon and ensemble
variance for each physics class. Ordinary tangent/adjoint sensitivities grow
exponentially beyond the Lyapunov time even when the long-time statistical
response is finite. NILSAS removes that divergence and is independent of the
number of design parameters, but its work scales with the number of positive
Lyapunov exponents; the 2026 wall-turbulence study shows why direct ensemble
adjoints can also require prohibitive sample counts. Measure GKX's unstable
dimension on reduced resolved grids before implementing shadowing. Online
gradient flow is a current scalable comparator, but it uses finite-difference
updates and therefore is not the selected autodiff API.

Stabilized march replaces the NILSAS least-squares problem by segmented QR
propagation and triangular substitutions. Its paper proves convergence under
uniform hyperbolicity and demonstrates only Lorenz-63 and
Kuramoto--Sivashinsky. It still evolves one homogeneous adjoint per retained
unstable direction, requires the unstable dimension/Lyapunov spectrum, and
stores or checkpoints the primal path. For GKX it is therefore a worthwhile
reduced-grid experiment only after measuring the positive Lyapunov count; it
does not displace the finite-window discrete adjoint without a gyrokinetic
cost, bias, and matched-transport result.

Acton et al. provide an independent linear-adjoint benchmark and warn that
nearly degenerate dominant modes can make a growth-rate gradient ambiguous;
GKX's eigenpair residual/overlap/conditioning gates remain required. That
linear result does not justify an implicit fixed-point adjoint for chaotic
saturated turbulence. No surveyed source demonstrates shadowing or a stationary
adjoint for nonlinear gyrokinetics, so promotion requires GKX-specific sign,
conditioning, cost, and held-out transport gates.

iGENE is the closest nonlinear-AD comparator found. It also backpropagates only
from a separately saturated state. In its Cyclone case, nonlinear gradients
diverge beyond about 512 RK4 steps while the heat-flux correlation time is
500--1000 steps; at the apparent best horizon their magnitudes reach only
15--50% of finite-difference estimates. Its successful profile workflow uses
16-step gradients, clips them, averages six evaluations, and validates the
final state independently. This supports GKX's finite-window choice but rejects
an “exact turbulent-transport gradient” claim. Compare sign/descent probability,
variance, wall time, and final held-out transport against FD/SPSA; do not infer a
useful horizon from step count rather than measured correlation/Lyapunov time.

## Differentiable-code and FFT performance comparators

- gyaradax paper, https://arxiv.org/abs/2604.06085
- gyaradax upstream source, https://github.com/gerkone/gyaradax

gyaradax's pure-JAX nonlinear bracket packs the x/y derivatives of each operand
into the real/imaginary parts of one complex field, reducing four inverse real
FFTs to two inverse complex FFTs. It also runs the bracket FFTs in float32 while
retaining float64 linear terms and field solves. GKX already batches derivative
transforms and has a compressed-real path, but not this two-for-one packing.
Benchmark a pure-JAX packed prototype against GKX's current bracket with exact
forward, JVP/VJP, conservation, dealiased-spectrum, CPU/GPU wall, and peak-memory
gates. Do not adopt gyaradax's optional cuFFT FFI path without an explicit custom
derivative: its public implementation exposes FFI calls but no custom VJP/JVP.
The paper's 400-step inverse problem is linear, not evidence for nonlinear
transport gradients.

A first local feasibility screen on an Apple M3 Max, JAX 0.10.2, complex64,
`(Nl,Nm,Nx,Ny,Nz)=(2,4,32,48,8)` gives a 0.764 packed/current warm-wall ratio.
Forward relative L2 error is `3.16e-7`, and the gradient with respect to a real
physical-space state agrees to `3.49e-7`. The unconstrained full-complex-state
VJP differs by 26.8%, however: explicit Hermitian completion changes the
off-physical-manifold derivative even when forward values agree. Treat this as
a blocker, not a speedup claim. A checked-in experiment must show projected
full-step VJP/FD parity before GPU or transport benchmarking.

The paper describes roughly 3,000 lines for its core integrator/field solver;
the audited checkout contains about 12,100 Python lines plus CUDA. Its physics
scope is much narrower than GKX's collision, geometry, eigensolver, artifact,
and validation surface. Use its functional boundaries and measured kernels as
slimming comparators, not its line count as an accuracy-independent target.

## Stellarator optimization chain

- Hirshman & Whitson, steepest-descent moment method for VMEC,
  https://doi.org/10.1063/1.864116
- Kim et al., nonlinear stellarator turbulence optimization with GX/DESC,
  https://arxiv.org/abs/2310.18842
- Wei et al., low-dimensional geometry learning for turbulence prediction in
  optimized stellarators, https://arxiv.org/abs/2603.17366
- Paischer et al., GyroSwin nonlinear gyrokinetic surrogate,
  https://arxiv.org/abs/2510.07314; code and weights:
  https://github.com/ml-jku/neural-gyrokinetics
- Galletti et al., physics-informed neural compression of gyrokinetic data,
  https://arxiv.org/abs/2602.04758
- VMEC/VMEX and GKX must expose the chain
  `boundary -> equilibrium -> Boozer/field line -> gyrokinetic window -> Q` with
  every local derivative checked against finite differences before composition.
  Kim et al. are the direct nonlinear comparator: their noisy objective used
  two-evaluation SPSA. GKX's finite-window adjoint should be compared against
  matched SPSA/finite-difference cost and final independent saturated transport,
  not against a noiseless optimization trace. Their post-processing raises the
  perpendicular grid to `128x128`, velocity resolution to `(Nl,Nm)=(8,16)`,
  and tube length to two poloidal turns; their field-line scan still finds about
  50% heat-flux variation. This supports independent `alpha`, `npol`, spatial,
  and velocity gates rather than validating GKX's current single-tube result.

Wei et al. identify a low-dimensional latent space for optimized QH geometry
and a correlation between linear zonal residue and axis excursion. This is a
useful later design-of-experiments/surrogate coordinate, not a replacement for
nonlinear labels: train it only on source-pinned, resolution-qualified GKX/GX
transport and reserve complete geometries as held-out tests.

GyroSwin is a useful learned-rollout comparator, not a transport or derivative
oracle. Its present adiabatic-electron GKW data cover 241 training simulations;
the paper reports long rollouts but also accumulated error, smoothed zonal
profiles, and weaker high-`ky` spectra. Admission to GKX therefore requires
held-out geometry, `alpha`, `npol`, resolution, timestep, seed, invariant, and
long-window transport gates. Neural compression is likewise optional storage
research: lossless compact traces and exact hashes remain the regression source
of truth; no lossy artifact may support acceptance until it preserves Q,
`Wphi`, `Wg`, spectra, autocorrelation/SEM, restart, and derivative quantities.

## SOLVAX relevance

- Upstream code: https://github.com/uwplasma/SOLVAX

SOLVAX's banded/block solves are useful where Hermite/Laguerre recurrences
produce a banded implicit approximation. GKX already adopts that capability as
preconditioning inside matrix-free GMRES; it does not remove the nonlinear FFT
cost. Changing the default requires an operator-structure audit, a residual
gate, and CPU/GPU wall/memory comparison against the present diagonal route.

The present GKX operator is not globally block tridiagonal. Streaming couples
Hermite neighbors only after a spectral or twist-linked parallel derivative;
the mirror term is a sparse two-dimensional Hermite--Laguerre stencil, and
curvature/grad-B reaches `m+/-2` and `l+/-1`. Finite-wavelength Coulomb tables
couple about half of the flattened moment pairs, while the nonlinear
pseudospectral bracket couples perpendicular Fourier modes after the Laguerre
grid transform.

GKX already uses SOLVAX's backend-aware `tridiagonal_solve` in its opt-in
`hermite-line` and linked/coarse implicit preconditioners, batched over the
remaining modes; `auto` still selects a diagonal factor. Current direct tests
mostly establish shape and finite output, not iteration or device advantage.
Treat the existing line solve as a frozen-linear preconditioner, never an exact
nonlinear solve. Compare `auto`, `pas`, `hermite-line`, and
`hermite-line-coarse` with matched residuals: Krylov iterations, compile/warm
wall time, peak memory, and transpose/VJP parity on CPU and GPU. Change the
default only if resolved linear and nonlinear IMEX cases win without changing
the solution. PR #101 corrects the performance guide: the `diag` factor contains
damping plus the curvature/grad-B diagonal, not the off-diagonal mirror
stencil.
