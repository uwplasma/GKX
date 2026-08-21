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

Leverage: boundary-condition tests must use real stellarator geometry;
perpendicular pile-up can be numerical and must be resolved before transport is
trusted. Sánchez et al. show that required flux-tube length is
configuration-dependent and that short W7-X tubes can disagree even between
field-line labels; scan both `alpha` and `npol` rather than treating `Nz` on one
tube as a complete parallel convergence test. LES-style closures are research
options only after direct-resolution baselines exist.

## Stochastic averages and stopping

- Oberparleiter et al., uncertainty estimation and a stopping rule in nonlinear
  gyrokinetic simulations,
  https://publications.lib.chalmers.se/records/fulltext/247070/local_247070.pdf
- Vaezi & Holland, quantifying temporal uncertainties of nonlinear turbulence
  simulations, https://arxiv.org/abs/1902.10879
- Rezaeiravesh et al., in-situ estimation of time-averaging uncertainties,
  https://arxiv.org/abs/2310.08676

Leverage: remove burn-in by stationarity testing; account for autocorrelation;
use batches of several correlation times; guard late drift; quote uncertainty,
not raw output count. GKX should compare its IAT estimator against the paper's
five-correlation-time batch means. The executable contract is: regression
stationarity first, estimate the 1/e correlation time, use non-overlapping
batches of length `5*tau_c`, target 5--10% corrected relative SEM, and require
the final-window drift to remain within 20% of the mean. Manual review remains
necessary when stationarity fails.
Vaezi--Holland specifically warns that gyrokinetic flux uncertainty becomes
harder near the critical gradient, so SAT-1 must be checked across drive, not
only on the present `tprim=3` case. The low-memory in-situ ACF update is a
future bounded-memory option; adopt it only if it reproduces offline batch/IAT
uncertainties on GKX traces.

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

## Discrete adjoints and chaotic sensitivity

- Griewank & Walther, Revolve checkpointing,
  https://doi.org/10.1145/347837.347846
- JAX gradient checkpointing documentation,
  https://docs.jax.dev/en/latest/gradient-checkpointing.html
- Wang, Hu & Blonigan, least-squares shadowing,
  https://arxiv.org/abs/1204.0159
- Acton et al., adjoint optimization of linear gyrokinetic microstability,
  https://doi.org/10.1017/S0022377824000709

Leverage: retain the finite discrete adjoint with rematerialization as GKX's one
production nonlinear derivative. Validate its useful window for each physics
class. Acton et al. provide an independent linear-adjoint benchmark and warn
that nearly degenerate dominant modes can make a growth-rate gradient
ambiguous; GKX's eigenpair residual/overlap/conditioning gates remain required.
That linear result does not justify an implicit fixed-point adjoint for chaotic
saturated turbulence. Shadowing remains research-only until it passes sign,
conditioning, and cost gates on GKX trajectories.

## Stellarator optimization chain

- Hirshman & Whitson, steepest-descent moment method for VMEC,
  https://doi.org/10.1063/1.864116
- Kim et al., nonlinear stellarator turbulence optimization with GX/DESC,
  https://arxiv.org/abs/2310.18842
- VMEC/VMEX and GKX must expose the chain
  `boundary -> equilibrium -> Boozer/field line -> gyrokinetic window -> Q` with
  every local derivative checked against finite differences before composition.
  Kim et al. are the direct nonlinear comparator: their noisy objective used
  two-evaluation SPSA. GKX's finite-window adjoint should be compared against
  matched SPSA/finite-difference cost and final independent saturated transport,
  not against a noiseless optimization trace.

## SOLVAX relevance

- Upstream code: https://github.com/uwplasma/SOLVAX

SOLVAX's banded/block solves are promising where Hermite/Laguerre recurrences
produce a true banded implicit operator. They do not remove the nonlinear FFT
cost. A GKX adoption requires an operator-structure derivation, a residual gate,
and CPU/GPU wall/memory comparison against the present matrix-free route.
