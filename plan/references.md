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
- Schekochihin et al., gyrokinetic phase-space cascade,
  https://arxiv.org/abs/0806.1069
- Morel et al., dynamic gyrokinetic large-eddy procedure,
  https://arxiv.org/abs/1110.0747

Leverage: boundary-condition tests must use real stellarator geometry;
perpendicular pile-up can be numerical and must be resolved before transport is
trusted. LES-style closures are research options only after direct-resolution
baselines exist.

## Stochastic averages and stopping

- Oberparleiter et al., uncertainty estimation and a stopping rule in nonlinear
  gyrokinetic simulations,
  https://publications.lib.chalmers.se/records/fulltext/247070/local_247070.pdf

Leverage: remove burn-in by stationarity testing; account for autocorrelation;
use batches of several correlation times; guard late drift; quote uncertainty,
not raw output count. GKX should compare its IAT estimator against the paper's
five-correlation-time batch means.

## Discrete adjoints and chaotic sensitivity

- Griewank & Walther, Revolve checkpointing,
  https://doi.org/10.1145/347837.347846
- JAX gradient checkpointing documentation,
  https://docs.jax.dev/en/latest/gradient-checkpointing.html
- Wang, Hu & Blonigan, least-squares shadowing,
  https://arxiv.org/abs/1204.0159

Leverage: retain the finite discrete adjoint with rematerialization as GKX's one
production nonlinear derivative. Validate its useful window for each physics
class. Shadowing remains research-only until it passes sign, conditioning, and
cost gates on GKX trajectories.

## Stellarator optimization chain

- Hirshman & Whitson, steepest-descent moment method for VMEC,
  https://doi.org/10.1063/1.864116
- VMEC/VMEX and GKX must expose the chain
  `boundary -> equilibrium -> Boozer/field line -> gyrokinetic window -> Q` with
  every local derivative checked against finite differences before composition.

## SOLVAX relevance

- Upstream code: https://github.com/uwplasma/SOLVAX

SOLVAX's banded/block solves are promising where Hermite/Laguerre recurrences
produce a true banded implicit operator. They do not remove the nonlinear FFT
cost. A GKX adoption requires an operator-structure derivation, a residual gate,
and CPU/GPU wall/memory comparison against the present matrix-free route.
