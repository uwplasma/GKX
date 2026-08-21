# Pull-request audit

GitHub reports 73 merged PRs through #80, one open pre-audit branch (#74), and
the CI repair PR #81. None of the merged PRs has an approving review. This file
separates inventory from independent source verification; a merged state is not
an audit verdict.

Legend: `I` = metadata/diff inventory complete, source re-audit pending; `R` =
prior review exists but current-tree regression audit remains; `F` = concrete
follow-up defect found in the 2026-08-21 audit.

| PR | Change | Audit |
| ---: | --- | :---: |
| 1 | internal EIK geometry generator | I |
| 3 | VMEC/geometry/init CLI overrides | I |
| 7 | differentiable-architecture plan | F |
| 8 | GKX 2.0 refactor | I |
| 10 | release prep and old plan removal | I |
| 11 | version 1.7.1/PyPI check | I |
| 12 | wide-coverage timeout | I |
| 13 | runtime collision-operator selection | I |
| 14 | drift/gyrokinetic Coulomb operators | I |
| 15 | collision verification and Laguerre conditioning | I |
| 16 | non-symmetric eigenvector derivatives | I |
| 17 | Codecov and Spitzer--Harm gate | I |
| 18 | capability docs and strict Codecov | I |
| 19 | finite-Larmor collision cost | I |
| 20 | collision data generator resolution | I |
| 21 | geometry/field physics fixes | I |
| 22 | reflectionless Hermite closure | I |
| 23 | differentiable matrix-free eigensolver | I |
| 24 | host-resident adaptive diagnostics | I |
| 26 | nonlinear gradients and saturated window | R |
| 27 | autocorrelation-corrected window statistics | R |
| 28 | Cyclone linear threshold | I |
| 29 | gradient parameter names | I |
| 30 | linear-objective velocity closure | I |
| 31 | shift-invert preconditioner validation | I |
| 32 | copied strided diagnostic chunks | R |
| 33 | timestep/CFL cost report | I |
| 34 | two-GPU device-z measurement | I |
| 35 | parallel policy routing | I |
| 36 | local fused device-z bracket | I |
| 37 | sharded Laguerre diamagnetic closure | I |
| 38 | local fused device-z observables | I |
| 39 | removal of staged pencil transforms | I |
| 40 | device-z transport reductions | I |
| 41 | Python floor | R |
| 42 | README turbulence movie | F |
| 43 | conserved-contraction precision | R |
| 44 | TF32 completion | R |
| 45 | six-case GX parity/scaling | F |
| 46 | automatic fit-signal selection | R |
| 47 | precision-aware residual gates | R |
| 48 | nonlinear autodiff and QA optimization | F |
| 49 | research plan/log | F |
| 50 | overflow, scan plots, dependency floors | R |
| 51 | direct `gkx wout.nc` UX | R |
| 52 | certified Krylov route | R |
| 53 | publication plot library | F |
| 54 | saturation stopping | F |
| 55 | transient Cyclone probe label | R |
| 56 | resolved growth-rate reporting | R |
| 57 | differentiable curvature drift | R |
| 58 | auto plots and compile cache | F |
| 59 | sheared-tube derivative | R |
| 60 | thermal-velocity convention | R |
| 61 | compressed-real-FFT derivative | R |
| 62 | species/Hermite sharding | F |
| 63 | equilibrium-shorthand coverage | R |
| 64 | generic comparison reader | R |
| 65 | plan log wave | F |
| 66 | converged Cyclone horizon | R |
| 67 | measured zonal baseline | R |
| 68 | reject a never-excited flux trace | F |
| 69 | plan log wave | F |
| 70 | traced host-read fixes | R |
| 71 | optional warm state | R |
| 72 | plan log wave | F |
| 73 | deck/stop-policy audit | F |
| 75 | independent CI signals | R |
| 76 | autodiff coverage/evidence | F |
| 77 | bracket fusion change | R |
| 78 | plan log wave | F |
| 79 | binary-input error | R |
| 80 | run summary and final output | F |

Missing PR numbers were never merged; #74 remains open.

## Concrete findings

| PR(s) | Finding | Required action |
| --- | --- | --- |
| 7 | AI label remains in merge history | relabel during coordinated history rewrite |
| 42, 53, 58, 80 | movie/3-D plot renders a synthetic circular torus, not VMEC geometry | physical artifact-only coordinates and non-axisymmetric tests |
| 45 | some tracked parity evidence is transient or not resolution-matched | regenerate converged, like-for-like CPU/GPU/GX matrix |
| 48 | merged despite the stated no-merge workflow; finite-window evidence was promoted beyond replicated validation | keep method, narrow claims, rerun AD/FD and independent QA campaign |
| 49 | plan was merged, making the log stale source-tree payload | remove from main and keep replacement PR open |
| 54, 68, 73 | stopping logic checks too coarsely and burn-in selection can include nonlinear overshoot | RUN-1 and SAT-1 |
| 62 | later CI work documents seven genuine `NameError` failures that merged | re-audit every sharding branch and require CI |
| 65, 69, 72, 78 | plan-only changes were repeatedly merged | consolidate history; branch-only governance |
| 76 | evidence includes generated static assets and claims that need held-out repetition | migrate artifacts and distinguish regenerated from independent evidence |
| 80 | mypy failure; impossible saturation interval; final state/time mismatch; misleading tube | #81 plus RUN-1/GEO-1 |

## Independent review protocol

For each row, record in this file:

1. parent-to-merge diff and public API/config/schema changes;
2. mathematical/physics contract and dimensional/normalization check;
3. edge cases, precision, JIT/tracing, CPU/GPU, memory, and parallel semantics;
4. tests added versus behavior changed; tests that only assert an artifact exists
   do not validate its physics;
5. CI status at merge and whether a later PR proves the change was defective;
6. current-tree reproduction or a reason it cannot be reproduced; and
7. disposition: keep, fix, narrow claim, superseded, or remove.

The next pass begins with the high-risk physics and infrastructure sequence
`#8, #14--16, #21--27, #34--40, #45, #48, #54, #57--62, #68, #73, #76--80`,
then closes the remaining inventory rows.
