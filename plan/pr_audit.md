# Pull-request audit

GitHub reports 74 merged PRs through #81. None has an approving review. The
2026-08-21 pass inspected each surviving parent-to-merge diff, its tests, its
head checks, later fixes to the same contract, and the current code. A green
head is evidence that tests ran, not evidence that the physics is right.

Legend: `K` = keep; `D` = keep with named debt/follow-up; `S` = superseded or
remove; `U` = the old diff is no longer reconstructible from surviving Git
objects. `red` describes the PR head at merge, not today's main branch.

| PR | Change | Verdict | Independent audit |
| ---: | --- | :---: | --- |
| 1 | internal EIK generator | U | GitHub now reports a zero-file diff and its merge object is unreachable; current `gkx` replaced the legacy package. Preserve the PR record, not a correctness claim. |
| 3 | VMEC/geometry/init CLI overrides | S | Four-file legacy `spectraxgk` change with CLI tests; removed by #8. Current equivalent is #51/#79. |
| 7 | differentiable-architecture plan | D | 885 files, +124,733/-63,246, no approval: refactor, data, plots, release and physics were inseparable. AI-labelled merge metadata and generated evidence must be rewritten/relocated; current contracts need independent gates. |
| 8 | GKX 2.0 refactor | D | 1,141 files, +24,454/-94,990, no approval. It established today's package and removed much legacy bulk, but is too large for a credible line review. Treat current validation matrix as its acceptance test. |
| 10 | release prep | K | Nine packaging/docs files; the release checks are reproducible. Its own head had red coverage shards, so later green release checks are the evidence. |
| 11 | v1.7.1/PyPI check | K | Focused version/publish fix; later release workflows supersede it. Head had red test coverage. |
| 12 | wide-coverage timeout | S | CI-only timeout adjustment; #75 replaced the dependency topology. Head coverage remained red. |
| 13 | collision selection | D | Runtime selector and shape tests are sound, but this head was red and operator physics is accepted only through #15 conservation/limit tests. |
| 14 | Coulomb operators | D | Like-species-only contracts fail closed and tables are checksummed. Conservation, nullspace, finite-`k_perp` limit and literature normalization remain the acceptance basis; no multispecies claim. Head was red. |
| 15 | collision verification/Laguerre conditioning | K | Stable scaled recurrence, round-trip tripwire and conservation gates repair the unsafe transform. Retain compact NPZ plus minimal provenance; do not treat bundled JSON/plots as validation. Head was red. |
| 16 | nonsymmetric eigenvector AD | D | Enables JAX eigenvector derivatives, but the dense dominant-branch path itself has no spectral-gap/branch-continuation gate. Valid only away from crossings; matrix-free continuation is preferred. Head was red. |
| 17 | Codecov/Spitzer--Harm gate | K | CI upload semantics plus a literature-normalized physics gate; green head. |
| 18 | capability docs/strict Codecov | K | Documentation and strict upload restoration; green head. Comparison claims still require the referenced executable gates. |
| 19 | finite-Larmor collision cost | K | Measurement/test-only bound; useful as a regression budget, not a performance claim on other hardware. |
| 20 | collision generator resolution | K | Two-line generator fix makes the `b -> 0` check use requested resolution. |
| 21 | geometry/field fixes | K | Corrects `gds22`, imported `gradpar`, sharded drift sign and adiabatic `ky=0` response; dedicated physics contracts cover each. These are evidence of silent defects in the #7/#8 baseline. |
| 22 | reflectionless Hermite closure | D | Recurrence closure and analytic reflection test are appropriate; nonlinear/turbulent resolution still requires moment-tail convergence, so it is not a universal no-reflection claim. |
| 23 | differentiable matrix-free eigensolver | D | Residual, dense-parity and branch-continuation tests exist; a 37-file change and dependency on SOLVAX make precision/device/near-degeneracy validation ongoing. Keep as linear method, not the nonlinear AD method. |
| 24 | host-resident adaptive diagnostics | K | Copies strided diagnostics to host and tests bounded device residency. #32 fixes aliasing of those host views. |
| 26 | nonlinear gradient/window campaign | S | Production source was untouched; it added scripts and static evidence. #48/#76 supersede the method and contracts. Remove large tracked media during slimming. |
| 27 | autocorrelation-corrected statistics | D | Replaces output count by estimated `n_eff`; AR(1) unit test is useful. First-zero integrated autocorrelation is noisy on short/nonstationary traces, so #54 stopping cannot rely on this alone. |
| 28 | Cyclone linear threshold | D | Adds a measured threshold and tests, but it is one geometry/resolution; retain as a regression point, not broad validation. |
| 29 | gradient names | K | Separates ion/electron `a/L_T` and `a/L_n` through config/cache/tests. |
| 30 | linear-objective closure | D | Fixes missing velocity closure; acceptance still depends on eigenbranch locality from #16/#23. |
| 31 | preconditioner validation | K | Unknown shift-invert names now fail closed; focused tests. |
| 32 | copied diagnostic chunks | K | `.copy()` prevents chunk views retaining/aliasing larger buffers; focused regression. |
| 33 | timestep/CFL report | D | Attribution is diagnostic and tested, but subtractive `ExB` attribution is a heuristic; do not read it as an exact cost decomposition. |
| 34 | two-GPU device-z measurement | D | Reproducible profiler scripts, but tracked GPU plots are machine-specific and show no production speedup. Move results out of Git history. |
| 35 | parallel-policy routing | D | Routes configuration and fails closed; the path remains diagnostic because multi-device identity/speedup is not generally closed. |
| 36 | local fused device-z bracket | D | Focused kernel change without a test in the PR; later #38--#40 and current identity gates carry acceptance. |
| 37 | sharded Laguerre drive closure | D | Local moment-edge closure and tests are present; only the covered electrostatic slice is validated, not the full nonlinear sharded solver. |
| 38 | device-z observable transform | D | Focused optimization, but no test in the PR; #40/current serial identity tests are required. |
| 39 | remove staged pencil transforms | K | Deletes dead transforms after call-site search and adds nonlinear test coverage. |
| 40 | device-z compensated reductions | D | Compensated summation and identity tests reduce reduction-order error; still no demonstrated two-GPU production win. |
| 41 | Python floor | K | Packaging/typing compatibility sweep; current Python-floor check is green. Its parent object is absent locally after earlier history surgery, so the original diff is not reconstructible there. |
| 42 | README movie | S | Documentation pointed to a pre-rendered synthetic circular-torus movie. PR #96 makes snapshots cut-only and physical after #86; PR #97 adds a provenance-labelled production-state continuation. |
| 43 | contraction precision | K | Pins precision on conserved contractions with focused tests; #44 completes shape guards. |
| 44 | TF32 completion | K | Adds the remaining precision pins and shape guard tests; keep. |
| 45 | six-case GX parity/scaling | D | Scripts are useful, but several tracked rows are transient or resolution-mismatched and generated assets dominate the diff. Regenerate matched, converged CPU/GPU/GX evidence. |
| 46 | automatic fit signal | K | Correctly makes `auto` select the channel while respecting an explicit fit window; test coverage survives. |
| 47 | precision-aware residuals | D | Avoids unreachable float32 gates, but `1000 eps` is empirical; retain explicit reported tolerance and cross-device residual tests. |
| 48 | nonlinear AD/QA optimization | D | Consolidates to checkpointed finite-window reverse mode and removes much obsolete material. It was merged despite the no-merge request and promotes single-campaign evidence too far. The shipped JSON passes its AD/FD gate through 1024 steps but fails at 2048, while the docs claimed agreement through 2048 and gave a memory-regeneration command inconsistent with the 1024-step artifacts. The QA example also substitutes fixed-step `8x8x16` spin-up for the production Q/Wphi/Wg saturation contract, applies a Cyclone-derived 1024-step horizon without a QA-specific gate, and refreshes its detached state only between VMEX stages; its CI smoke is two `4x4x8` steps. Keep the adjoint implementation; PR #100 narrows the claim, and source-pinned AD/FD, saturation, horizon, and matched transport campaigns must pass before promoting the optimizer. |
| 49 | research plan/log | S | Accidentally merged mutable plan plus patches/notes. #83 removes it from main; #82 is the do-not-merge living replacement. Head mypy was red. |
| 50 | overflow/scan/dependency fixes | K | Fit overflow fails loudly, scan plotting is bounded and dependencies are pinned. |
| 51 | direct `gkx wout.nc` | D | Useful CLI path with tests, but its head release-artifact check was red and #79 later repairs input classification. |
| 52 | certified Krylov route | D | Adds residual/stability failure paths; keep, with #47 precision floors and dense parity as mandatory gates. |
| 53 | publication figure library | D | Reusable transforms are tested, but the flux-tube renderer invented a circular torus. #86 supplies physical VMEC coordinates; old images must not support physics claims. |
| 54 | saturation stopping | D | Sensible initial `tau`, SEM and stationarity gates, but median-crossing burn-in and coarse chunk checks admit overshoot and impossible windows. #84 fixes time/check cadence; SAT-1 must replace burn-in validation before changing defaults. Head release check was red. |
| 55 | transient probe label | K | Correctly demotes an unmatched Cyclone GX result; docs-only, despite inherited red release check. |
| 56 | resolved growth reporting | K | Adds fit uncertainty/resolution status and incidentally repairs #62's missing imports before tests ran. Head inherited red release check; current tests pass. |
| 57 | differentiable curvature drift | D | Adds missing VMEC pressure/current drift and finite-beta parity tests. Broader equilibria and radial derivative convergence remain required. |
| 58 | auto plots/cache | D | Compile-cache path is appropriate and plotting failures are nonfatal, but broad exception handling hides renderer defects and the tube was synthetic. #86 fixes geometry; add structured warnings. |
| 59 | sheared-tube AD | D | Removes traced host conversion and tests linked-boundary gradients. Valid only for the tested topology; generalized twist-and-shift remains separate. |
| 60 | thermal velocity | K | Documentation-only normalization clarification; its head mypy was red for unrelated source. |
| 61 | compressed real FFT AD | K | Explicit Hermitian projector repairs derivative semantics with unit parity/gradient tests. Head inherited a red release check. |
| 62 | species/Hermite sharding | D | Merged with red mypy while quick tests were skipped, leaving seven `NameError` uses until #56. Current route must stay experimental until full RHS identity and real GPU scaling pass. |
| 63 | shorthand coverage owner | K | Manifest-only ownership correction; no physics change. |
| 64 | generic comparison reader | K | Removes code-specific naming from artifact reader and restores release terminology checks. |
| 65 | plan-only wave | S | Mutable log merged to main; consolidate into #82 and remove from source history. |
| 66 | Cyclone horizon | K | Extends example/benchmark horizon to measured convergence; case-specific only. |
| 67 | zonal baseline | D | Makes a measured zonal artifact explicit and adds validation gates; tracked binary outputs belong outside core history. |
| 68 | zero-signal stop guard | D | Correctly prevents a zero trace from passing, but the gate key stores the positive condition under the negative name `flux_indistinguishable_from_zero`. Rename in a schema-versioned cleanup; head checks were cancelled. |
| 69 | plan-only wave | S | Move findings to #82; remove plan payload from main. |
| 70 | traced host reads | K | Replaces JAX round-trips at host-only boundaries and adds release/static tests. |
| 71 | optional warm state | D | Off-by-default state reuse has amplitude and step guards; parallel scans disable it. Head docs failed, and nonlinear statistical independence must never reuse warm states silently. |
| 72 | plan-only wave | S | Move warning to #82; head docs failed. |
| 73 | deck/stop audit | D | Adds fail-closed deck checks and zero-signal rules, but does not repair median-crossing burn-in; head docs failed. |
| 75 | independent CI signals | K | Removes false-green `needs: mypy` topology and adds `ci-required`; this explains #62. Branch protection still needs a human approval requirement. |
| 76 | AD coverage/evidence | D | Adds method/device/stepper contracts and regenerators; bundled JSON/PNG are not independent validation and held-out seeds/resolutions remain open. |
| 77 | bracket fusion | K | Removes an adverse fusion boundary with focused parallel regression tests. |
| 78 | plan-only wave | S | Move audit log to #82; remove from main. |
| 79 | binary input error | K | Focused TOML-vs-binary classification and regression test. |
| 80 | run summary/output | D | Useful summary/NetCDF integration, merged with red mypy/`ci-required`; emitted an impossible saturation interval, lost the exact final horizon/diagnostic and rendered a synthetic tube. #81/#84/#86 are the required repairs. |
| 81 | main CI repair | K | One-line run-summary typing repair plus a nonlinear-only 20-minute wrapper budget. All 41 PR checks and the post-merge main run passed; no test was weakened. It was merged externally with no approving review. |

Missing PR numbers were never merged; #74 remains open.

## Merge-control findings

- Approving reviews: **0/74**.
- Red heads merged: #10--#16 (coverage), #49/#60/#62 (mypy),
  #51/#54--#59/#61/#63 (release artifact), #71--#73 (docs), and #80
  (mypy plus aggregate CI).
- #68 merged with every head check cancelled.
- #7 and #8 are mega-merges whose combined scope defeats independent review.
  No future PR should mix physics, generated evidence, refactoring and release
  changes at that scale.

## Concrete follow-ups

| PR(s) | Finding | Required action |
| --- | --- | --- |
| 1 | original diff cannot be reconstructed | preserve GitHub record; do not invent an audit verdict |
| 7, 8 | foundational mega-merges cannot be accepted by line review | close the current validation matrix contract-by-contract |
| 7 | AI label remains in merge history | relabel during coordinated history rewrite |
| 42, 53, 58, 80 | movie/3-D plot renders a synthetic circular torus, not VMEC geometry | physical artifact-only coordinates and non-axisymmetric tests |
| 45 | some tracked parity evidence is transient or not resolution-matched | regenerate converged, like-for-like CPU/GPU/GX matrix |
| 48 | merged despite the stated no-merge workflow; finite-window evidence was promoted beyond replicated validation, the 2048-step prose contradicted its failed JSON gate, and the QA optimizer bypasses the production saturation contract | PR #100 narrows the claim; refactor the QA script toward VMEX, remove inert warm-start code, and require source-pinned case-specific saturation/horizon/AD-FD plus independent matched QA campaigns |
| 49 | plan was merged, making the log stale source-tree payload | remove from main and keep replacement PR open |
| 54, 68, 73 | stopping logic checks too coarsely and burn-in selection can include nonlinear overshoot | RUN-1 and SAT-1 |
| 62 | seven `NameError` uses merged because tests were skipped; #56 supplied the missing imports | keep sharding experimental and require #75 aggregate CI |
| 65, 69, 72, 78 | plan-only changes were repeatedly merged | consolidate history; branch-only governance |
| 76 | evidence includes generated static assets and claims that need held-out repetition | migrate artifacts and distinguish regenerated from independent evidence |
| 68 | gate key names the failure while storing the passing boolean | schema-versioned rename |
| 80 | mypy failure; impossible saturation interval; final state/time mismatch; misleading tube | #81, #84 and #86 |

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

This pass closes the diff audit. It does **not** close the validation debt named
in the `D` rows: those rows move to the public validation matrix and remain
open until their stated CPU/GPU, literature, resolution or statistical gates
pass. Re-audit the affected row whenever a follow-up changes its contract.
