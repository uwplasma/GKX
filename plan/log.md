# Work log

## 2026-08-21 — repository and runtime audit

- Audited `main` at `5f3ab32e`, GitHub PR metadata, branch protection,
  authorship, repository objects, the QA output bundle, saturation logic, plot
  geometry, and movie generator.
- Reproduced the main CI mypy failure and opened PR #81 with a one-line,
  source-line-neutral fix. Local mypy, architecture gate, and 44 plotting tests
  pass.
- Demonstrated that the QA artifact has an impossible saturation window ending
  after the saved trace and that the final state can be later than its label.
- Rejected the proposed `t=50--55` average: only two saved samples, about 24%
  below the later mean, with `Wg` still evolving.
- Identified the 3-D stellarator plot as a synthetic circular torus and the
  terminal heat-flux spectrum as unresolved.
- Rehearsed a source-complete history rewrite. Keeping every original commit and
  all `src/` history while snapshotting auxiliary paths produces a clean
  7.72-MiB pack; current tracked/generated assets prevent a sub-10-MiB result
  without migration.
- Replaced the stale append-only roadmap with current evidence. Old prose and
  patch snapshots remain recoverable in Git history and the pre-rewrite bundle.
- Opened PR #83 to remove the accidentally merged plan from `main`; this PR #82
  remains the branch-only living roadmap and must not be merged.
- Opened PR #84 to make the state, fields, restart, and terminal diagnostic share
  the exact physical-time horizon and to check saturation every 128 steps. Its
  broad nonlinear/runtime/release suite passes; the stopping statistic itself
  is deliberately unchanged pending the replicated validation campaign.
- Recorded an independent pre-existing validation defect: a float32 eager/JIT
  trajectory test demands `rtol=1e-12` although its observed relative difference
  is about `8e-8` on both the PR branch and untouched `main`.
- Opened PR #85 with a two-line nonlinear startup glossary: field model,
  kinetic/adiabatic response, `tprim=a/LT`, `fprim=a/Ln`, and definitions of
  `gamma`, `omega`, `Wphi`, `Wg`, and `Q`.
- Audited all 123 tracked nonlinear heat-flux traces: 54 end near `t=150`, 6 at
  `t=250`, 12 at `t=400`, 12 at `t=700`, and 39 at `t=900`; none reaches the
  `t=1500` advertised in its filename. The current terminal rule accepts 87.
  Of those, 81 also pass a loose late-tail drift gate. Retrospective sequential
  regression/batch-mean variants still make several early estimates more than
  10% from the late reference. Therefore no new default is promoted from these
  heat-only traces; Wphi/Wg histories and held-out runs remain mandatory.
- Opened PR #86 to carry VMEC `Rplot`, `Zplot`, and
  `zeta=zeta_center+q*theta_PEST` as host-only output metadata and render the
  actual field line. The JAX pytree is unchanged; the QA rerender uses `nfp=2`
  and 48 aligned samples instead of a synthetic circular torus.
- Completed a per-PR audit of all 73 merged PRs through #80. No PR had an
  approving review. #10--#16, #49, #51, #54--#63, #71--#73 and #80 include
  red head checks; #68 has only cancelled checks. The ledger separates 29 keep
  decisions, 34 named-debt decisions, 9 superseded/remove decisions and 1 diff
  whose old objects are no longer reconstructible. Validation debt in a `D` row
  remains open even though the diff audit is complete.
- Opened PR #87 with a high-`ky` cutoff warning on both transport and potential
  spectra. For the supplied QA run over `t=[100,200]`, the top-10%-tail/peak
  ratios are 0.65 for `Q(ky)` and 0.36 for `Phi2(ky)`, both above the 0.10
  warning threshold. The check is labelled necessary, not sufficient; the
  paired `Nx`/`Ny` convergence campaign remains open.
- Rebased PR #74 onto the CI-fix branch, removed its Claude co-author trailer
  and generated-by footer, and preserved Rogerio as both author and committer.
- Created and verified a complete 285,415,042-byte recovery bundle containing
  178 refs. Its SHA-256 is
  `d08b073f34886914512d9c6d459c08d3404a49f26b4b0df5388e73acfa837205`.
- Repeated the source-complete rewrite with aggressive garbage collection: one
  5.93-MiB pack, 15,925 objects, 3,359 commits, 29 refs, strict `fsck` clean,
  and a 2,269,659-byte current-tree archive. The candidate is not publishable:
  the release suite still reads a removed `_static` quasilinear JSON artifact.
  `plan/history_rewrite.md` records the lossless recovery and cutover protocol.
- Compared the documentation footprint with MHX and VMEX. MHX keeps 31 concise
  pages and about 1.1 MB of documentation; VMEX keeps a tutorial/how-to/
  reference/explanation hierarchy and about 3.3 MB of tracked documentation.
  GKX's 35 MB tree and test-coupled result archive should adopt the same small,
  generated-or-fetched evidence model rather than store campaign outputs.
- Fixed the new PR #84 x64 CI fixture to honor its declared float32 scan
  contract; the focused x64 horizon gate and all 15 explicit-step tests pass.
- Fixed PR #86's mypy failure by narrowing imported `R`, `Z`, and toroidal-angle
  profiles to explicit NumPy arrays; mypy, Ruff, and 122 relevant tests pass.
- Opened draft PR #88 after a literal-consumer audit of all 1,201 tracked
  `docs/_static` files. CI exposed that the first audit followed documentation
  links but missed a tool/test consumer. A repository-wide basename audit
  restored 44 consumed artifacts (339,046 bytes). The corrected cut removes
  153 assets, 7,696,596 bytes, and 5,882 text lines; 88 affected tests pass.
- Re-reduced all 208 raw QA campaign NPZ files (104 matched pairs). Signed
  ensemble drift had hidden individual failures. Under the published 20%
  final-drift gate the nominal set is 44/48 stationary; `dt=0.04` is 13/16,
  perpendicular 12 is 7/8, perpendicular 20 is 27/32, perpendicular 24 is 6/8,
  long perpendicular 24 is 30/32, `Nz=16` is 6/8, `Nz=32` is 8/8, velocity
  `(3,6)` is 6/8, and velocity `(6,12)` is 30/32. No resolved spectra were
  saved. Draft PR #89 changes the public 12.26% result to preliminary and makes
  the audit fail closed. Its release-test phrase mismatch was fixed in
  `330e07a9` after GitHub CI exposed it.
- The separately advertised strict-QA `t=1500` raw NetCDF directory is absent.
  Retained generated trace CSVs end near `t=400`, so those summaries cannot be
  reconstructed or promoted. Their recorded matched results are negative:
  growth 0.60%, quasilinear -0.49%, and nonlinear-window -0.25%.
- Opened draft PR #90, stacked on #89, to require auditable promotion evidence:
  unique raw paths and SHA-256 hashes, observed time/window bounds, individual
  stationarity, autocorrelation-corrected uncertainty, timestep, perpendicular,
  parallel, velocity, and spectral convergence. The production guard remains
  safe but unpromoted: zero qualifying optimized ensembles and matched audits.
  Focused tests (154), Ruff, mypy, strict Sphinx, and all release manifests pass;
  installable source decreases from 96,465 to 96,463 lines.
- GitHub CI found PR #86 nine lines over the no-regression source budget. Commit
  `17312709` condensed redundant snapshot prose without losing its FFT or
  normalization contract; the branch now has 96,464 source lines. Mypy, Ruff,
  the architecture gate, and 122 geometry/plot/runtime tests pass.
- Retrospective held-out stopping tests show why the current rule cannot be
  promoted unchanged. Heat-flux-only stopping accepted 54/208 traces with
  median 5.30% and 90th-percentile 10.50% error against their last 400 time
  units; 29 accepted means missed by more than 5% and 7 by more than 10%. The
  Oberparleiter 100-time burn-in plus five-correlation-time batches stopped
  earlier but was less safe on these traces. Q-only rules cannot replace fresh
  `Q`, `Wphi`, `Wg`, and spectral evidence.
- Opened draft PR #91 with compact fixed-horizon saturation-audit output: time,
  adaptive step, `Q`, `Wphi`, `Wg`,
  `kx`, `ky`, and resolved spectra. A local `16^2 x 48`, `t=100` CPU pilot took
  66.3 s and failed saturation with `Wg` still rising. A shared RTX A4000
  `32^2 x 48`, `t=250` pilot took 469.8 s and was also pre-saturation: mean `Q`
  rose from 0.147 over `t=100--150` to 2.45 over `150--200` and 7.25 over
  `200--250`; the highest retained positive-`ky` mode was the flux-spectrum
  maximum. These are diagnostic pilots, not timing baselines or converged
  transport results.
- The `48^2 x 48`, `t=250` resolution rung also failed the production
  stop rule: its selected `t=80.3--250` window has 17.7% autocorrelation-corrected
  relative SEM. `Wphi` and `Wg` pass the half-window drift check, showing why an
  energy plateau cannot replace flux uncertainty. The final four positive
  `ky` modes carry 49.7% of late heat flux and the cutoff mode carries 13.1%,
  so the grid fails perpendicular convergence even if the trace is extended.
- The `64^2 x 48`, `t=250` rung fails for the same reason. Recomputed with the
  exact production policy, its `t=50.3--250` window has mean `Q=11.48` and
  9.68% relative SEM. More importantly, the last retained positive-`ky` mode
  is at the spectral maximum and carries 19.5% of windowed flux; the last
  three carry 51.1%. Extending this state would reduce sampling error without
  resolving the perpendicular cascade, so the next rung is `96^2 x 48`.
- Re-read the user's original `96^2 x 48` NetCDF rather than inferring from its
  plot. The proposed `t=50--55` window contains only two saved points: mean
  `Q=8.18`, `Wphi=1.30`, and `Wg=180.2`, versus `Q=10.92`, `Wphi=1.64`, and
  `Wg=246.1` over `t=150--200`. It is shorter than one measured correlation
  time and precedes continued free-energy evolution. The rendered averaging
  window starts at `t=29.83` inside the `Q≈42` nonlinear overshoot and ends at
  `t=238`, beyond the saved trace at `t=200.62`; the spectrum panel inherits
  that invalid window. The late heat-flux
  spectrum peaks at `ky*rho=1.19`; its `ky*rho=1.48` cutoff remains 68% of the
  peak and the last three bins carry 14.6% of positive flux. This run is
  neither statistically nor perpendicularly resolved. Increasing `Ny` extends
  the retained `ky` range, but `Nx` and `Ny` must be converged together because
  the nonlinear bracket transfers both coordinates. The NetCDF records
  `nfp=2`; the apparently axisymmetric 3-D tube is therefore a renderer bug,
  not an equilibrium property.
- Audited the movie generator independently. It reruns a fixed-step RK4
  trajectory instead of sampling the production solve, stores full 3-D fields,
  and reconstructs a circular torus without VMEC coordinates. MOV-1 must record
  only decimated `x-y` and physical tube-skin cuts during the source run.
- A restart smoke exposed that PR #91's first `--nz` override changed `Nz` but
  left a deck's explicit `ntheta` in control. The `32^2` and `48^2` pilots
  therefore used 48 parallel points, not 32. Commit `177587ea` makes the tool
  report the built grid, updates `ntheta` consistently, verifies restart shape,
  keeps continuation time absolute, and removes its duplicate saturation rule
  in favour of the exact production policy.
- PR #91 then consolidated the runtime and post-hoc first-zero Sokal estimators.
  The exact production decision is now the only stopping method used by the
  audit utility, 76 focused analysis/saturation tests and all 117 release tests
  pass, and installed source decreases by 21 lines.
- PR #91 exposes random seed, `dt`, `dt_max`, and CFL controls for matched
  ensembles without deck copies. A real `8^3` CPU smoke confirmed seed 31,
  `dt_max=0.04`, `cfl=0.8`, the built state shape, and persisted metadata.
- The same audit tool now exposes stellarator field-line label `alpha` and tube
  length `npol`. A real VMEC `8^3` smoke at `alpha=0.25`, `npol=1.5` preserved
  both values and the exact `t=0.05` state/trace horizon.
- Opened draft PR #92 on #88 to decouple generated renders from release
  evidence. Numeric JSON/CSV/TOML remains mandatory; absent rendered media is
  counted but does not fail. In the rewrite rehearsal, restoring only current
  JSON/CSV evidence gives 121/121 release tests and a 7.36 MiB packed repo
  (3.5 MiB archive), still below the 10 MiB target.
- Audited authorship over the complete selected rewrite history. Seven objects
  carry AI markers: four duplicated Claude co-author trailers and three Codex
  branch/message labels. The rewrite rehearsal removes those markers and maps
  164 Rogerio Wisc-address commits to the IST identity; its reachable history
  has 3,355 Rogerio commits and preserves the four non-Rogerio human commits.
- Opened draft PR #93 as the first source-slimming cut. The public
  differentiable-geometry facade now delegates to the canonical VMEC/Boozer
  core while preserving its monkeypatch hooks and signature metadata: 28
  installed Python lines removed, 263 focused tests pass under JAX 0.10.2, and
  mypy passes on both touched modules. A GitHub API scan of all fifteen open
  PR heads finds only Rogerio's IST author/committer identity and no AI
  attribution markers.
- Opened draft PR #94 on #87 after visually reproducing the supplied QA plots.
  Automatic plotting now propagates `saturated=false`, removes the rejected
  `t=29.83--238` window from standalone averages, labels the result diagnostic,
  and computes the plotted spectrum and cutoff warning over the same late-half
  interval. The summary retains the attempted window with its explicit
  `NOT saturated` verdict. All 116 plotting/CLI tests pass and installed source
  decreases by five lines relative to the architecture baseline.
- The asset-free full-suite rehearsal found three unit-level PNG existence
  assertions outside PR #92's release gates. Commits `09b1adc1` and `adc9d497`
  put the parallel-artifact tests under the same numeric-required,
  render-optional suffix policy; all 25 parallel and 121 release tests pass.
  A repository-wide search finds no other existence assertion for a tracked
  render; remaining PNG checks generate their own temporary outputs.
- A fresh `96 x 96 x 48`, seed-22, `t=250` QA solve on one RTX A4000 took
  2,959.5 s and retained 540 samples. The unchanged production rule still
  rejects it: its selected `t=29.9--250` window includes the nonlinear
  overshoot, has mean `Q=12.03`, 9.64% autocorrelation-corrected relative SEM,
  and first/second-half means 12.97/11.09. Both energy guards pass, but late
  50-time means remain 10.76, 11.35, 11.14, and 10.58. This confirms that
  `Wphi`/`Wg` stationarity alone cannot define the averaging window.
- On the physical `t=150--250` spectrum, heat flux peaks at
  `ky*rho=1.333`; the `ky*rho=1.476` cutoff is 54.2% of that peak and the last
  three retained bins carry 14.0% of summed absolute positive-`ky` flux. By
  contrast, the outer three `kx` shells carry about 0.1%. The controlled next
  rung therefore holds `Nx=96`, `Nz=48`, seed, and horizon fixed and raises
  only `Ny` to 128 before a joint `Nx`/`Ny` confirmation.
- That analysis exposed a campaign-artifact defect: PR #91 initially wrote all
  FFT slots, including structural dealiased zeros, beside full `kx`/`ky` axes.
  A generic tail check could therefore report false convergence. Commit
  `48043360` now uses the canonical NetCDF dealiased layout and tests the
  physical `(63, 32)` axes for a `96 x 96` grid. The completed raw trace is
  recoverable by the same canonical indices; no solver result changed.
- The asset-free full suite then found the remaining source of PR #92's PNG
  coupling: the aggregate parallel validator still classified renders as
  required sidecars. Commit `5ddfce88` restricts required sidecars to JSON and
  CSV (24 current artifacts); PNGs remain registered and reported by the
  manifest but are reproducible, optional outputs.
- The refreshed sensitivity survey adds NILSAS, fast adjoint linear response,
  2026 online gradient flow, and the July 2026 wall-turbulence adjoint study.
  They sharpen the public claim: checkpointed reverse mode is the one
  production derivative of a declared finite GKX window, not yet a derivative
  of invariant-measure heat flux. Shadowing is mathematically better matched
  to the latter but scales with the unstable dimension; online gradient flow
  avoids a long adjoint but is finite-difference based. No source demonstrates
  either method for nonlinear gyrokinetics, and all cited papers were
  accessible for this audit.
- Opened draft PR #95 on #92 for the final render budget. It keeps the 12
  README/core-physics visuals requested for public use (2,354,028 bytes),
  including the lightweight turbulence loop, QA equilibrium/Boozer panel,
  transport trace, and nonlinear-adjoint validation. It removes 226 generated
  PNG/PDF/SVG outputs (18,170,355 bytes) and 121 optional RST image directives;
  useful captions remain explicit generated-figure notes. Strict Sphinx and
  all 146 release/parallel tests pass without suppressing missing-image
  warnings.
- A fresh single-branch clone of the combined rewrite rehearsal plus PR #95
  initially packed to 9.49 MB; its source archive was 5.6 MB. The asset-free
  full suite subsequently found one generic benchmark-manifest existence loop
  at 84%.
  PR #92 commit `e33813cf` now requires every numeric table while allowing a
  missing render only when its regeneration contract is complete. The full
  suite was restarted; this is validation progress, not rewrite authorization.
- That restarted suite found the final stale PNG requirement at 95%: the
  quasilinear promotion guard treated an absent optional plot as failed physics
  despite a valid numerical sidecar. PR #95 commit `7c68dba2` adds the explicit
  `regenerate_on_demand` release-manifest state, keeps hash checking for any
  present render, and makes quasilinear render presence informational. All 137
  focused release/quasilinear tests and strict Sphinx pass.
- Replaying the reviewed render and numeric-evidence commits into a fresh
  source-complete single-branch rehearsal produces one 9,464,119-byte pack and
  a 5,883,089-byte source archive. Repository hygiene, strict Sphinx, strict
  `fsck`, sdist/wheel build, installed-wheel import, CLI startup, and all 2,554
  collected x64 tests pass. No rewrite has been pushed.
- The Nuhrenberg--Zille QHS `64x96x48`, seed-22 sizing run completed exactly at
  `t=250` with 528 samples in 1,617.0 s on one RTX A4000. The production window
  fails at 9.64% corrected relative SEM. Its physical `t=150--250` heat-flux
  cutoff/peak is 15.6% and last-three-bin/summed-absolute-tail fraction is
  3.18%; Phi2 gives 10.4% and 2.32%. The trace SHA-256 is
  `da0a87a17b31fc762de3c89fde033d8f2464c622028953807d09ec81b4021ba4`.
  The `64x128x48` QHS and `96x128x48` QA refinements were then launched on
  separate GPUs. Neither launch by itself is a resolution or transport claim;
  the completed QA result is recorded below.
- A line-by-line stop-policy audit found that the standalone autocorrelation
  campaign still reported `n_eff=n/(1+2 tau/dt)` while the runtime and validated
  window statistics use `min(n,n dt/(2 tau))`. PR #91 commit `f1ad010e` removes
  that 41% independent-sample-floor discrepancy and adds a correlated-trace
  regression; it changes post-processing uncertainty only, not the solver or
  stop thresholds.
- Required CI is green on both PR #91 and PR #95, including all 24 wide shards,
  aggregate `ci-required`, docs/packaging, mypy, Python-floor, and Codecov.
- The matched QA `96x128x48`, seed-22 run completed exactly at `t=250` with 658
  samples in 3,648.2 s on one RTX A4000. On `t=150--250`, `Q=11.00584` with
  2.21% corrected relative SEM and stationary Q/Wphi/Wg, versus `Q=10.85954`
  at `Ny=96` (1.35% difference). The new cutoff remains 18.2% of the spectral
  peak and 16.3% of summed absolute heat-flux magnitude lies beyond the old
  cutoff, so `Ny=160` is running. Trace SHA-256:
  `d24c88a8c58f1614a082887bb5b85b63f6887cce6b17d7374b64677041ec37a9`.
- Opened draft PR #96 on #86. The movie snapshot schema now stores only the xy
  and yz cuts consumed by rendering, plus physical VMEC coordinates and box
  extent; full-volume legacy snapshots still replay. The README hero import
  failure is fixed and 60 lines of duplicate surface construction are removed.
  All 47 plotting tests, Ruff, the source-line budget, two CLI import smokes,
  and a physical two-frame render pass. Installable source decreases by two
  lines. Production-trajectory sampling remains explicitly unresolved.
- The remote-ref audit found 63 advertised branch heads and 17 open PR heads.
  A sub-10-MB single-branch rehearsal is therefore not a cutover proof. The
  final frozen rewrite must replay all open heads onto the reviewed slim base
  and remove closed heads only after their tips are in the verified bundle and
  published ref map; no force-push has occurred.
- The public-ref rehearsal is now complete locally. All 17 open PR heads were
  mapped onto the source-complete slim history: five reviewed slimming heads
  alias candidate `main`, and twelve were replayed from their actual PR bases
  with matching aggregate patch IDs and no old-`main` ancestry. PR #90's sole
  excluded path is the generated promotion-guard PNG deliberately removed by
  PR #95; its remaining patch is identical. A fresh no-alternates clone has 18
  remote heads, 23 tags, 3,169 commits, 16,873 objects, a 9,517,768-byte pack,
  473,516-byte index, and 5,883,633-byte source archive; strict `fsck` passes.
  No public ref was changed.
- The QHS `64x128x48`, seed-22 rung completed to `t=250` with 587 samples in
  2,192.3 s. On the fixed `t=150--250` suffix, `Q=6.47593` with 2.96% corrected
  relative SEM, but Q, Wphi, and Wg all fail half-window stationarity; the mean
  is 22.7% below the matched `Ny=96` result. The heat-flux cutoff/positive-peak
  ratio improves to 9.91% and the final three bins carry 1.69% of positive-ky
  magnitude. The spectral screen is necessary but cannot rescue a moving mean,
  so the exact state is continuing to absolute `t=500` on GPU 0. Trace SHA-256:
  `a0550f46a5eb6bd8ec1e1eaeb9d9ad315bbdfced25e4217506fd393ace701325`.
- Opened draft PR #97 on #96. The movie tool now restores the exact saturation
  campaign NPZ, advances it with the deck's production explicit method and CFL
  policy, preserves absolute time and source saturation status, and renders the
  cut-only artifact off-device. A real `4x4x4` CPU continuation resumed at
  `t=10` and wrote its first frame at `t=10.05`; 48 plotting tests, Ruff, strict
  Sphinx, and the architecture gate pass. The tool itself decreases by two
  lines and installed source is unchanged.
- PR #97 raised the live inventory to 66 advertised branches and 18 open PRs.
  Its aggregate patch ID survives replay on the slim #96 head. The refreshed
  no-alternates clone has 19 heads, 23 tags, 3,170 commits, 16,885 objects, a
  9,506,509-byte pack, 473,852-byte index, and 5,884,245-byte archive; strict
  `fsck` passes and no public ref was changed.
- PR #91 commit `a067995b` enables the existing persistent JAX compilation
  cache in the saturation campaign, which previously bypassed CLI setup. Two
  identical `4x4x4` CPU processes took 7.11 s cold and 2.61 s warm (63% less
  end-to-end); lint and the eight nonlinear artifact-contract tests pass. This
  is compile-dominated workflow evidence only, and the running QA `Ny=160` and
  QHS `t=500` jobs use the earlier revision.
- A second literature/code pass added the 2025--2026 W7-X multiscale result and
  the stellarator turbulence SSA analysis. The former makes the scope boundary
  explicit: the current adiabatic-electron `ky*rho_i ~ 2` refinement is an
  ion-scale tail test, not kinetic-electron/ETG convergence. SSA remains a
  reviewer diagnostic for avalanches, not a production stopping dependency.
- PR #95 commit `a4866b23` re-encodes the README turbulence loop from its primary
  release MP4 at 900 px and 5 fps, preserving the six-second interval while
  reducing 828,066 to 346,234 bytes. Strict Sphinx, 142 release/parallel tests,
  and 46 benchmark tests on the synthetic advanced base pass. Replacing it in
  the prior 19-head rehearsal gives a 9,030,044-byte pack and 9,503,896 bytes
  with its index; strict `fsck` passes and the old movie blob is unreachable.
- Rejected a tempting SAT-1 shortcut with causal prefix replay. A 32-suffix
  selector using corrected SEM, `10 tau_ac`, half-window agreement, and linear
  drift gates on Q/Wphi/Wg would stop the QHS `64x128x48` trace at `t~56` with
  `Q~9.96`, before its later `Q~6.5` regime. On 16 existing four-diagnostic VMEC
  traces, sequential current-policy replay with both energy guards accepts
  seven; four are over 5% and one 10.4% from the final tail. No default changes.
- PR #91 commit `91ccf3e7` fixes a narrower production defect exposed by that
  audit: runtime saturation stopping now requires Wg, as well as Wphi, to pass
  the same half-window stationarity gate. The campaign previously computed Wg
  only after integration and could still report a production stop while free
  energy drifted. This is a conservative guard, not validation of the current
  median-crossing burn-in selector; `t_max` remains the fail-safe horizon.
  Forty-one focused statistics/chunk tests, 102 runtime-runner tests, Ruff,
  mypy, and the no-source-growth architecture gate pass.
- The stopping survey now includes fixed-width correlated-output theory,
  consistent batch/spectral variance, multivariate effective sample size, and
  offline/online change-point methods. It corrects the plan's stale description
  of production: GKX currently uses median crossing, first-zero Sokal IAT,
  `10*tau_ac`, 5% relative SEM, and Q/Wphi/Wg half-window gates. The research
  target is causal change-point burn-in plus a held-out persistence batch and a
  consistent long-run covariance estimate; no paper makes that safe on GKX
  without the ongoing prefix-replay campaign.
- A fresh 206-file/96,465-line source inventory and eight-line clone scan found
  immediate one-owner cuts in spectral layout, Diffrax packing, growth fits,
  transport reductions, nonlinear diagnostic setup, and runtime option
  forwarding. `plan/history_rewrite.md` now assigns the owner and CPU/GPU/JIT/
  AD gate for each; moving lines between source and tools is explicitly barred.
- The QHS `64x128x48` continuation reached absolute `t=500` in another
  2,127.0 s. Over `t=250--500`, `Q=6.25381` with 1.93% corrected relative SEM
  and broad Q/Wphi/Wg half-window agreement. That verdict is not persistent:
  over `t=350--500`, Q changes from 6.580 to 5.994 and Wg from 145.2 to 134.3,
  so both gates fail. The heat-flux cutoff/peak remains about 10% and the final
  three positive-ky bins about 1.7% of magnitude. The exact state is therefore
  continuing to `t=750`; `t=500` is not promoted as saturated.
- Continuation analysis exposed that the PR #91 JSON report used segment-local
  times while the NPZ trace correctly stored absolute times. Commit `37a7cf77`
  passes the absolute axis to the shared report function; eight artifact
  contract tests and Ruff pass. This changes reporting only, not integration.
- Opened draft PR #98 as the second source-slimming cut. Artifact I/O imports
  the canonical dealiased spectral layout and deletes its three private copies:
  17 installed lines removed (`96,465 -> 96,448`), with 45 focused tests, Ruff,
  mypy, and the architecture gate passing locally.
- PR #97's nonlinear CI shard completed every assertion at 14m11s, printed
  100%, and was then canceled at the job's 15-minute limit. PR #81 commit
  `d910ac56` gives only `nonlinear-core` a 20-minute budget; the six other
  quick-test shards retain 15 minutes. Workflow parsing, five repository gates,
  and release readiness pass locally. This is a runner-budget repair, not a
  test or physics failure.
- Refreshed the non-destructive history rehearsal through PR #98 and the latest
  saturation/roadmap/CI commits. Its 20 heads exactly equal `main` plus all 19
  open PR heads; all 23 tags survive. A fresh no-alternates clone has 3,184
  commits, 16,981 objects, a 9,042,520-byte pack, 9,519,060 bytes with its
  index, and a 5,408,247-byte source archive. Strict `fsck`, 14 stable patch-ID
  comparisons, and the commit-metadata AI-marker scan pass. No public ref was
  changed.
- Opened draft PR #99 on #98. It delegates the second growth-fit input and
  least-squares path to the canonical window fit, preserving exact legacy
  outputs for complex64/complex128, nonfinite filtering, and four window
  modes. The stacked source is 96,391 lines (`-74`); 52 diagnostic tests, 201
  benchmark/runtime tests, Ruff, mypy, and the architecture gate pass. A
  10,000-point fit is unchanged within noise (478 us before, 476 us after).
- PR #97's clean rerun is green. `nonlinear-core` passed in 14m51s, only nine
  seconds below its old timeout, and the aggregate `ci-required` gate passed.
  This confirms runner jitter/capacity as the failure mode and supports PR
  #81's targeted 20-minute budget.
- The QHS seed-22 continuation reached absolute `t=750` in a third 2,122.5-s
  segment. Over `t=500--750`, `Q=6.113 +/- 0.076` (1.24% corrected SEM), and
  Q/Wphi/Wg pass half-window stationarity; `t=600--750` and `650--750` pass as
  independent late suffix checks. The late cutoff/peak is 9.43% and the last
  three positive-ky bins carry 1.78%. The causal policy still falsely accepts
  near `t=56`, and Ny=96 differs by about 27%, so this is a late-state result,
  not a stop-policy or resolution promotion.
- The matched QA `96x160x48`, seed-22 run completed at `t=250` in 5,327.5 s.
  On the fixed `t=150--250` suffix, `Q=10.643 +/- 0.278`, Q/Wphi/Wg are
  stationary, the Ny=128 difference is -3.30% (0.98 combined SEM), the cutoff
  is 7.59%, and the last three bins carry 1.50%. Seed 31 is running as the
  independent replication; no multi-seed claim is made.
- PR #97 now caps H.264 output at 900 px and reads `Nl` and `Nm` from `[run]`.
  The exact QA Ny=160 production state continued for 30 frames on an RTX A4000
  in 49.1 s, but the command imported an older installed GKX: its coordinate
  profiles were empty and the render used the fallback tube. The result is
  timing evidence only. The corrected branch passes all 51 plotting tests and
  fails closed on missing imported profiles; a clean `PYTHONPATH=src` rerun
  must verify nonempty VMEC profiles before the physical movie is accepted.
- Replayed a frozen trailing-window/persistence hypothesis on the completed QA
  Ny=160 and QHS Ny=128 histories. A 100-time-unit window that passes Q/Wphi/Wg
  continuously for 20 further time units stops at `t=149.7` and `t=299.2`;
  its means differ by 2.7% and 1.3% from the selected late audit means. This is
  post-hoc design evidence only; the independent seed, QHS Ny=160, QI, and 16
  legacy VMEC traces are the held-out score set.
- The 16-trace legacy score rejected that 100+20 hypothesis: 6 stops, four
  errors above 5%, one at 12.5%, and 9.2% median absolute error versus the
  final 100-time-unit mean. A declared training scan over those traces plus QA
  and QHS seed 22 freezes a 75-time-unit window with 60-time-unit persistence:
  3/18 stops, 4.2% worst error, 29% median runtime saved. It is training-biased
  by construction; QA seed 31, QHS Ny=160, and QI are untouched holdouts and
  will reject or retain it without retuning.
- PR #81's nonlinear CI shard passed in 14m22s under its targeted 20-minute
  budget. This confirms that the former 15-minute wall-clock limit was too
  tight for runner cleanup while leaving every other shard at 15 minutes.
- A real QA VMEC CPU movie smoke exposed a 49-coordinate/48-field-plane
  mismatch: the EIK interval is closed while the solver grid is open. PR #97
  now aligns geometry through the production trim before rendering. The rerun
  records 48 finite `R/Z/zeta` samples, `nfp=2`, and deck moments `(4,8)`.
- Added PR #99 to the non-destructive public-ref rehearsal. Before this log
  commit, all 21 retained heads (`main` plus 20 open PRs) and all 28 remote tags
  fit in a 9,107,008-byte pack (9,596,344 with index); the archive is 5,363,108
  bytes, with 3,436 commits and 17,438 objects.
  The final parity audit caught five tags omitted by the earlier refresh
  (`archive/pre-gkx2-condescending-buck`, three `baseline-*` tags, and
  `v0.0.1`) and restored their rewritten targets before this measurement.
  Strict `fsck`, exact patch IDs for all 17 recent commits, and the AI-marker
  scan pass. No public ref moved.
- Replayed PR #97 commit `f114d13f` and this roadmap update onto the slim
  candidate without changing their patches. Immediately before this log commit,
  the fresh no-alternates clone has 3,438 commits, 17,453 objects, all 21 live
  heads, all 28 tags, a 9,112,197-byte pack (9,601,953 with index), and a
  5,363,108-byte archive. Strict `fsck`, all 19 recent patch IDs, ref-name
  parity, and the AI-marker scan pass. No public ref moved.
- The untouched QHS `64x160x48`, seed-22 holdout reached `t=250.320` in
  3,219.4 s. The frozen 75-time-unit window plus 60-time-unit persistence rule
  makes no stop, so it avoids a false acceptance but saves no time. On
  `t=150--250`, `Q=5.6619 +/- 0.1263` (2.23% corrected SEM), but Q halves
  5.8460/5.4793 fail and Wg halves 130.37/124.16 fail; Wphi passes. The heat
  cutoff/peak is 4.88%, last-three-bin mass 0.97%, Phi2 cutoff/last-three
  1.93%/0.45%, and outer-six-kx heat-flux mass 3.54%. This is a Ny-tail pass,
  not a transport-convergence result; exact continuation to `t=500` is
  running. Trace SHA-256:
  `09d67e8679713aa4d872ed76a0ecabcd8001c8a79f05e46f24e70e98bc16f81e`.

## 2026-08-22 — held-out replication and physical movie

- The untouched QA `96x160x48`, seed-31 holdout reached `t=250.186` in
  5,315.5 s. Its fixed `t=150--250` result is `Q=10.9623 +/- 0.2817` and all
  three stationarity gates pass. Seed 22 gives `10.6374 +/- 0.2761`: 3.01%
  spread, 0.82 combined standard errors, and conservative ensemble
  `10.7999 +/- 0.2817`. The frozen 75+60 rule stops at `t=215.947`, saves
  13.7%, and misses its own final mean by -1.09%. Q/Phi2 cutoff-to-peak values
  are 8.05%/8.80%, last-three-bin masses 1.48%/1.37%, and outer-six-kx masses
  0.12%/0.38%. This is one successful fresh stop and a replicated Ny=160
  result; matched Ny=128 seed 31 and QI remain. Trace SHA-256:
  `eb8541d337504d94579f60bcc4f3288ee6c9ff7c7fdec5297d14ce9526bfdc6f`.
- The corrected PR #97 GPU continuation produced 30 finite QA frames from the
  exact Ny=160 saturated state with deck moments `(Nl,Nm)=(4,8)`, `nfp=2`,
  absolute times `250.150--253.949`, and 48-point VMEC `R/Z/zeta` profiles.
  Its schema-3 NPZ is 2.5 MB and the off-device H.264 render is 218,799 bytes
  at 900x472, 10 fps. First/middle/final-frame inspection shows an evolving
  physical two-field-period open tube, with neither the synthetic-torus nor
  disconnected-end artifact. NPZ SHA-256:
  `b5bd4e0a61ecd29885757c69cb882ff053cafa5aaf9793025bc7a99e615712a2`;
  MP4 SHA-256:
  `628b1cff237b5b5f77816579463b78585ab94eb82448598b0757cc41bb53e7a7`.
- Replayed all four post-`f114d13f` PR #97 commits and all seven
  post-`f3655acc` roadmap commits onto the private slim candidate. Before this
  log commit, its 21 heads exactly match `main` plus the 20 open PR heads and
  its 28 tags exactly match the remote. A fresh no-alternates clone contains
  3,450 commits and 17,522 objects; its pack is 9,128,917 bytes, pack plus
  index is 9,620,605 bytes, the complete `.git` directory is 9,953,481 logical
  bytes, and the compressed tree archive is 5,363,108 bytes. Strict `fsck`,
  30 recent stable patch-ID comparisons, and the AI-attribution scan pass. No
  public ref moved.
- Auditing the live office processes exposed an editable-install leak: the PR
  #91 campaign scripts imported GKX from `/home/rjorge/gkx-wt/main` at
  `c749abfa`, not their own checkout. The intended branch changes exact-horizon
  plumbing in three nonlinear solver modules, so all preceding office traces
  are retained as sizing/negative-design evidence but removed from acceptance
  status. Partial continuation/QI jobs were stopped without deleting their
  logs. PR #91 commit `f7da8c49` now fails closed on a mismatched source and
  stores source path, commit, and dirty state in every artifact. QHS Ny=160 and
  QI restarted from zero; both logs begin with the clean in-checkout source
  `f7da8c49b803c738de67971bbab343a196f8f44e`.
- Re-auditing merged PR #48 found that its shipped gradient-window JSON passes
  the declared `1e-6` AD/FD gate through 1024 steps (`2.69e-9`) but fails at
  2048 (`2.48e-5`), contrary to one documentation claim. Its memory-profile
  replay command also requested 2048 steps while the published CPU/GPU JSON
  and README use 1024. PR #100 corrects only those statements and preserves the
  single block-checkpointed discrete adjoint. The private rewrite rehearsal now
  includes #100: before this log commit, 22 heads and 28 tags fit in a
  9,124,252-byte pack and 9,949,649-byte logical `.git` directory; strict
  `fsck`, live-head parity, and the new patch-ID comparison pass.
- PR #91 commit `c2b7284d` extends that fail-closed provenance rule to the
  nonlinear-gradient evidence ladder. A state is admissible only when its
  recorded checkout was clean and its `src/gkx` tree is identical to the
  active solver tree; tool-only commits may differ. The JSON records both
  source trees and commits. Sixty-one focused provenance/gradient tests and a
  local x64 matrix over RK2/RK3/RK4, multispecies, kinetic-electron,
  electromagnetic, collision, hypercollision, and checkpoint-parity paths
  pass. This validates the discrete-window implementation, not a turbulent
  invariant-measure derivative.
- The merged #48 QA optimization example is not yet a production optimization
  workflow. It is 255 lines versus the current 117-line VMEX reference, carries
  a roughly 40-line warm-start mechanism disabled by `max_reuse=0`, spins up
  with fixed `dt=0.05` for 8,000 steps at only `8x8x16`, and never evaluates
  the production Q/Wphi/Wg saturation gate. Its 1,024-step differentiated
  window is inherited from one Cyclone knee rather than measured for the QA
  state, and the detached state is refreshed only between VMEX stages. The CI
  path exercises only two steps on `4x4x8`. Refactor toward the VMEX example
  only after a source-pinned, case-specific saturation and gradient-horizon
  gate passes; remove the inert warm-start path and keep the claim explicitly
  to a fixed finite-window pathwise derivative.
- Direct inspection of the user's `96x96x48` QA NetCDF explains the apparent
  overrun. The file has only 87 diagnostics over `t=0.065--200.620`; `t=50--55`
  contains two samples and gives `Q=8.178`, about 25% below the
  `t=150--200` mean `10.917`. The old summary nevertheless declares an
  averaging interval `[29.834,238.025]`, beyond the file horizon and across the
  nonlinear overshoot; #84 repairs the time accounting and #94 prevents a
  rejected interval from being presented as an average. Over `t=100--200`,
  first-zero IAT gives `Q=11.101 +/- 0.320`, `Wphi=1.649 +/- 0.029`, and
  `Wg=249.5 +/- 5.1`; the final half still trends lower, so independent
  continuation is required before a stop claim.
- That same file ends at `ky*rho_i=1.476`. Over `t=100--200`, the cutoff
  heat-flux bin is 49.4% of the spectral peak and the last three positive-ky
  bins carry 12.9% of their summed flux; over `t=150--200` the values worsen
  to 68.3% and 14.6%. `Phi2` is less edge-loaded (29.7% and 5.5% over the
  first late window), explaining why its plot looks better. This run is not
  Ny-converged. At fixed `Ly=62.8`, increasing `Ny` extends `ky_max` and raises
  FFT/state cost; #87 makes the failure visible, while matched Ny=128/160
  transport and edge-mass convergence decide the production resolution.
- Causal prefix replay of the fixed 75-time-unit candidate window shows why a
  persistence hold is needed: all Q/Wphi/Wg gates pass briefly at
  `t=113.9--120.6`, then fail again through `t=147.5`; the durable pass begins
  near `t=149.8`. A frozen 60-time-unit persistence requirement therefore
  cannot stop this trace before about `t=210`, beyond the available `t=200.6`
  data but far below an unconditional `t=400--750`. Score this rule unchanged
  on the source-pinned QA/QHS/QI holdouts before making it a default.
- PR #85 commit `4c1ae7ee` tightens the startup glossary after an independent
  physics review: it prints `a/L_T=-a d ln(T)/dr` and
  `a/L_n=-a d ln(n)/dr`, identifies gamma/omega with the selected potential
  mode, distinguishes electrostatic field energy Wphi from distribution free
  energy Wg, and labels Q as radial heat flux in gyro-Bohm units. Focused
  runtime output, Ruff, mypy, diff, and architecture gates pass; installed
  source remains one line below the main baseline.
- JSON inventory on the post-#95 slim tree finds 553 files under
  `docs/_static`, 5.09 MB total. Although 248 are not named outside static
  JSON, all 553 are transitively reachable through report provenance, so a
  flat unreferenced-file deletion would silently break the evidence graph.
  The slimming plan now requires a versioned URI+SHA-256 provenance edge and
  one-family-at-a-time migration: compact verdicts stay tracked; raw objective
  histories, profiles, and replicate reports move to verified release assets.
- The clean-source QHS Ny=160 and QI campaigns were extended without changing
  their frozen `f7da8c49` checkout. Exact-state jobs are queued to absolute
  `t=750` for QHS and `t=500` for QI after the already-running `t=250`, QHS
  `t=500`, and QA Ny=128 jobs release their GPUs. Intermediate diagnostics are
  not interpreted as convergence evidence.
- For the user's one-species `(Nl,Nm,Nx,Nz)=(4,8,96,48)` layout at fixed
  `Ly=62.8`, Ny=96/128/160 retain `ky_max*rho_i=1.476/2.000/2.524`.
  A single complex64 distribution state is 108/144/180 MiB, while the
  `Ny log Ny` factor is 1/1.42/1.85. These are sizing estimates only; matched
  GPU wall time, CFL, peak memory, transport, and tail mass determine the
  production choice.
- Re-audited SOLVAX against the actual GKX operator rather than the recurrence
  analogy alone. Streaming is Hermite-neighbor local but contains a spectral
  or twist-linked parallel derivative; mirror couples a two-dimensional
  `(l,m)` stencil; curvature/grad-B reaches `m+/-2` and `l+/-1`; the shipped
  finite-wavelength Coulomb test/field matrices are about 49% nonzero; and the
  nonlinear bracket couples perpendicular modes through FFTs. A SOLVAX direct
  solve is therefore not an exact nonlinear GKX algorithm. The bounded next
  experiment is a frozen-linear banded preconditioner: measure discarded-term
  norm, Krylov iterations, residual, CPU/GPU wall and memory, and transpose-VJP
  parity before adding a SOLVAX dependency or changing the production solver.
- A deeper implicit-solver pass corrected that last sentence: GKX already
  depends on SOLVAX and calls its backend-aware batched tridiagonal solve from
  `hermite-line` and linked/coarse GMRES preconditioners. The default `auto`
  alias selects the diagonal `full` factor, and existing direct tests largely
  check shape/finite output. Do not add another solver path. Benchmark the
  existing `auto`/`pas`/Hermite-line choices on matched residual, iterations,
  compile/warm wall, memory, and VJP on CPU/GPU before changing the default.
- The first clean-source holdout completed on the frozen `f7da8c49` checkout:
  QHS `64x160x48`, seed 22 reached exact `t=250` in 2,906.9 s. Over
  `t=150--250`, `Q=6.5083 +/- 0.0937` (1.44% corrected SEM), and Q/Wphi/Wg
  all pass the half-window gates. The heat-flux cutoff/peak is 4.79%, the
  last-three-bin mass 0.93%, and the Phi2 values 0.010%/0.030%. The frozen
  75-time-unit window plus 60-time-unit persistence rule makes no stop: its
  three pass islands last only 11.7, 6.6, and 5.0 time units. The exact
  continuation remains required. Trace SHA-256:
  `928289d9e14d585a0dcb70b0b57939d556bb4030f71df6c2098fd4d5d6363911`.
  The shipped median-crossing selector begins at `t=31.84`, retains the
  overshoot, and therefore reports `Q=8.591 +/- 1.481` (17.2%) with Wg still
  failing. This source-pinned trace confirms that the present burn-in selector
  can prolong a run after a stationary late window exists.
- The same campaign exposed a small but systematic JSON duplication: with an
  NPZ trace requested, all 662 scalar samples are also embedded in the summary,
  making its JSON 98,879 bytes. A bounded generator change should replace that
  duplicate with the NPZ URI/path, schema, and SHA-256 while retaining inline
  samples only for explicitly JSON-only runs.
- The untouched clean-source QI `96x96x48`, seed-22 holdout reached exact
  `t=250` in 3,462.3 s. On `t=150--250`, `Q=4.1641 +/- 0.0922` (2.21%) and
  Q/Wphi/Wg pass, but Q and Wg fail on `t=200--250`. The frozen 75+60 rule
  correctly makes no stop: its longest pass island lasts 58.85 time units and
  fails again at `t=246.72`. Ny=96 is not resolved for transport: heat-flux
  cutoff/peak is 15.50% and last-three-bin mass 3.71%; Phi2 is only
  0.076%/0.241%. Exact continuation and a matched Ny refinement remain; the
  frozen threshold is unchanged. Trace SHA-256:
  `d7b511db5065e405f2a7511e0f335a8bc9b3cc12dcdf47af2fa9c4166e34ea55`.
- Replayed the latest #85 glossary and this source-pinned roadmap batch onto
  the private public-ref rehearsal with exact stable patch IDs. A forced
  upload-pack clone contains all 22 heads and 28 tags, 3,458 commits, and
  17,584 objects; its pack is 9,133,421 bytes, complete `.git` 9,960,094
  bytes, and current-tree archive 5,363,108 bytes. Strict `fsck` and the
  AI-attribution scan pass. The decimal size margin is only 39,906 bytes, and
  no public ref moves before reviewed prerequisites and a real GitHub clone.
- PR #91 commit `4ef05fb9` implements the campaign JSON deduplication. A
  requested NPZ is now addressed by schema, path, bytes, and streaming SHA-256;
  JSON-only output retains inline samples. On the clean QHS artifact the
  companion JSON falls from 98,879 to 2,128 bytes (97.8%). Fifty-five focused
  artifact/gradient tests, Ruff, and diff checks pass; `src/gkx` is unchanged,
  so the running source-pinned continuations keep exact solver-tree parity.
- PR #85 commit `87f69a9d` removes a stale promise from the shipped shorthand
  deck: it now explicitly sets `run_to = "saturation"` and names `t_max` as the
  hard cap, matching the runtime users already execute. The default-deck,
  shorthand, and three release `run_to` audit tests pass. Stride 50 remains
  unchanged pending a measured 10/25/50 diagnostic-cost and IAT-stability scan.
- Opened PR #101 after tracing the implicit factor construction instead of
  copying its prose description. The `diag` factor contains damping and the
  curvature/grad-B diagonal; the mirror stencil is off-diagonal and was never
  included in that factor. The one-line documentation correction passes strict
  Sphinx and leaves all solver defaults and performance claims unchanged.
- Replayed PR #91 commit `4ef05fb9`, PR #85 commit `87f69a9d`, and PR #101 onto
  the private public-ref rehearsal with exact stable patch IDs. Immediately
  before this roadmap commit, a fresh no-alternates clone advertises all 23
  heads and 28 tags, with 3,462 commits and 17,610 objects. Its pack is
  9,137,778 bytes, complete `.git` is 9,965,379 logical bytes, and the current
  archive is 5,363,108 bytes. Strict `fsck` passes. The remaining decimal margin
  is 34,621 bytes; no public ref moves before the recovery/ref-map and real
  GitHub clone gates.
- The source-pinned QHS exact-state continuation reached absolute `t=500` in
  another 2,846.4 s and preserved clean commit `f7da8c49`. The combined 1,309
  samples reproduce every previously frozen `75+60` decision and still make no
  stop: the longest new pass island is `t=245.04--284.85` (39.81 time units),
  and no candidate window passes after `t=367.76`. The apparently settled
  `t=400--500` mean is `Q=6.2927 +/- 0.1493`, but Q/Wphi/Wg all fail again over
  `t=450--500`. Its heat-flux cutoff/peak and last-three-bin mass remain
  resolved at 5.28%/0.94%, isolating temporal modulation from the Ny question.
  The unchanged `t=750` continuation started automatically. Trace SHA-256:
  `4c757301c2b3aa289e73f82a18d956ee977101017c2667a2d4ebeb5a6edc5407`.
- PR #85 commit `110b10aa` responds to a concrete nonlinear-log ambiguity:
  gamma/omega are now labelled selected-mode diagnostics, while the same line
  says saturation uses Q/Wphi/Wg. The focused header test and Ruff pass; no
  additional runtime work or stored diagnostics were added.
- The refreshed primary-source survey found two 2026 differentiable
  gyrokinetic comparators. iGENE independently observes nonlinear adjoint
  divergence beyond roughly one heat-flux correlation time and optimizes with
  clipped, averaged 16-step gradients plus independent final validation; it
  strengthens the finite-window choice but not an exact-gradient claim.
  gyaradax contributes a bounded performance experiment: pure-JAX two-for-one
  derivative packing and mixed-precision bracket FFTs. Its custom cuFFT FFI has
  no exposed VJP/JVP and is not a candidate for GKX's autodiff path as shipped.
- A temporary pure-JAX packed-bracket prototype on an Apple M3 Max/JAX 0.10.2
  reduced warm CPU time from 0.590 to 0.450 ms at `2x4x32x48x8` and matched the
  forward bracket/real-physical gradient within `3.2e-7`/`3.5e-7` relative L2.
  Its unconstrained complex-state VJP differs by 26.8% because Hermitian
  completion changes the off-manifold derivative. No source PR: first prove
  projected full-step VJP/FD parity, then measure GPU, memory, and transport.
- The clean `f7da8c49` QA `96x128x48`, seed-31 holdout reached exact `t=250`
  in 3,397.1 s. Over `t=150--250`, `Q=10.8522 +/- 0.3675` and Q/Wphi/Wg pass.
  The frozen rule stops at `t=230.99`; `Q=10.9089` differs by 0.52% and saves
  7.6%. The current selector fails narrowly at 5.61% SEM because it retains
  `t=26--250`. Ny=128 still fails the heat-spectrum screen at 15.69% cutoff/peak
  and 3.11% last-three mass. The Ny=160 seed-31 source-pinned match is queued;
  trace SHA-256:
  `91f6768849caa8a315594d7f3bac256cc46a9203829ae0affae65d20ab6b5c45`.
- With 319 GB free and no overlap added, queued clean QA Ny=160 seed 31 behind
  QHS `t=750` on GPU 0 and QI Ny=128 seed 22 behind the QI `t=500`
  continuation on GPU 1. These close independent-seed and failed-Ny screens;
  partial outputs are not evidence.
- Corrected the roadmap replay after finding that its pre-#81 ancestry restored
  the old 15-minute nonlinear CI timeout. Commit `dd2e6b4c` makes every
  non-plan file exactly match the CI-fix base; the roadmap branch is again a
  source-neutral living record.
- Audited the largest remaining current JSON by field and literal consumer.
  Of 423,062 bytes, 344,449 were deterministic reduced diagnostics, chiefly
  the artifact's 44-by-44 LCFS and Boozer grids; the comparison writer already
  retained only shapes, extrema, and scope. Opened draft PR #102 on #92 to
  apply that same contract to single-result sidecars after rendering. The
  nonlinear JSON is now 113,526 bytes (73.2% smaller), while histories,
  nonlinear traces, scans, summaries, and rendered figures remain. Focused tests, Ruff, the
  repository-size gate, and diff checks pass locally.
- Rebuilt the private rehearsal with the corrected roadmap tree and one hosted
  exact-tree roadmap snapshot. Full roadmap commit history remains in the
  verified recovery bundle. Replacing only the exact 423,062-byte raw-grid
  blob with PR #102's compact form leaves older sidecar revisions unchanged.
  A fresh no-alternates clone now has 24 heads, 28 tags, 3,414 commits, and
  17,366 objects; its pack is 9,067,221 bytes, pack plus index 9,554,541,
  complete `.git` 9,887,091, and current archive 5,332,787 bytes. Strict
  `fsck` and the AI-attribution scan pass. This is a rehearsal, not permission
  to force-push before review, recovery publication, and a real network clone.
- The source-pinned QHS `t=500--750` and QI `t=250--500` continuations remain
  active on separate office GPUs. Their queued Ny=160 QA and Ny=128 QI rungs
  have not started, and no partial output has been interpreted.
- Ranked reachable blobs by compressed pack cost rather than working-tree size.
  The largest retained object was the 371,750-byte Landau PNG, costing 360,267
  packed bytes. Opened draft PR #103 on the minimal-render PR #95: its generator
  now writes a deterministic 256-color median-cut PNG, and the tracked panel is
  151,626 bytes at the same 3,028-by-822 resolution. The previous-pixel PSNR is
  49.86 dB with 0.066 mean absolute channel error; full-width visual inspection
  found no readable difference.
- All six Landau physics tests pass. A fresh default figure regeneration
  recovers `T_e/T_i=1` gamma/omega within 0.2462%/0.0643%, `T_e/T_i=10` within
  0.0041%/0.0043%, and keeps the collisionless spectral real-part residual at
  `1.954e-14`. Ruff, the size gate, and diff checks pass.
- Replaced only Git blob `99a3ff82ac7c9e15e66635e1bb054380decb81ad` in the
  private rehearsal and replayed PR #103's source/test patch. Before this log
  entry, a fresh no-alternates clone has 25 heads, 28 tags, 3,415 commits, and
  17,375 objects; pack 8,836,982 bytes, complete `.git` 9,657,226, archive
  5,122,028. Strict `fsck` and attribution scans pass; no public ref moved.
- The clean QHS Ny=160 continuation reached exact `t=750` in 2,841.2 s.
  Concatenating all 1,955 samples preserves the frozen decisions and still
  yields no stop; the longest new pass island is only 14.28 time units. The
  complete-history current selector retains `t=31.43--750` and fails at 9.44%
  relative SEM, while the continuation-only JSON incorrectly promotes its
  `t=501.22--750` segment without seeing the earlier overshoot.
- Over `t=650--750`, `Q=6.1947 +/- 0.0768`, `Wphi=1.9000 +/- 0.0216`, and
  `Wg=136.16 +/- 1.70`; all half-window gates pass, but the shorter terminal
  75- and 50-time-unit windows fail again. Heat-flux cutoff/peak, last-three-ky
  mass, and outer-six-kx mass are 6.46%, 0.94%, and 3.23%; Phi2 is
  2.70%/0.46%/0.63%. Trace SHA-256:
  `e3d1152ce4b679cdb32119a7c59ad7452a51dc024cd2fab9f68391850e05d2c9`.
- The queued QA Ny=160 source-pinned run started on GPU 0. After it finishes,
  a new wrapper runs QA Ny=96 seed 31 and then a clean QHS Ny=128 `t=750`
  match; output names are unique and no existing artifact is overwritten.
- PR #103 exposed a stacked-branch false green in PR #95: five manifest entries
  existed only in GitHub's synthetic merge with the newer #92 base. PR #95 now
  merges that base and requires every absent output to be both a rendered file
  and explicitly marked `regenerate_on_demand`; the 46-test benchmark-contract
  suite passes. PR #103 carries the self-contained contract, and all six x64
  Landau physics tests plus the same benchmark suite pass before its CI rerun.
- Replayed that self-contained #95/#103 stack and this exact roadmap tree into
  the private public-ref rehearsal. A fresh no-alternates clone advertises all
  25 selected heads and 28 tags: 3,417 commits, 17,404 objects, an 8,863,687-byte
  pack, and a 9,684,887-byte complete `.git` file sum. Strict `fsck`, exact
  non-plan neutrality, exact plan-tree identity, and the reachable-metadata
  attribution scan pass. No public ref moved.
- PR #91 commit `53b47e99` closes the continuation-report defect exposed by the
  QHS horizon. A segment still records whether its local statistic passed, but
  `--initial-state` now sets the full-history claim false and records
  `prior_history_not_in_report`. Ruff and all 11 nonlinear-window artifact
  contract tests pass; the solver and active source-pinned trajectories are
  unchanged.
- The clean QI Ny=96 continuation reached exact `t=500` in 3,450.7 s. Its
  segment-only JSON promotes `t=257.67--500`, but the concatenated 1,773-sample
  history makes no frozen `75+60` stop. The complete-history selector keeps
  `t=30.30--500`, with `Q=4.6556 +/- 0.3991` and 8.57% relative SEM. The longest
  new pass island is `t=279.17--302.09` (22.92 time units).
- Over `t=400--500`, `Q=4.1869 +/- 0.0822`, `Wphi=2.4313 +/- 0.0237`, and
  `Wg=124.08 +/- 1.94`; Q and Wg fail half-window stationarity. The terminal
  75- and 50-time-unit windows also fail at least one Q/Wphi/Wg guard. Heat-flux
  cutoff/peak, last-three-ky mass, and outer-six-kx mass are 15.15%, 3.77%, and
  2.61%; Phi2 is 0.083%/0.259%/0.435%. Thus QI is still unresolved in Ny as well
  as time; the source-pinned Ny=128 rung has started on GPU 1.
- QI continuation SHA-256: NPZ
  `644be269e37d7b5cab838be94ba1748ba94ec3d332f9da44190cb46083520480`, JSON
  `5543fbaf0ceb7842d7d726c8558a343791a66bd2e156c5e4111df721f17073a4`, log
  `62c549b17943f045ce038a40a7816e352d3134db61a859565cf20c3e8a488490`.
- Replayed PR #91's continuation fix and the preceding exact roadmap tree into
  the private rehearsal. Its fresh no-alternates clone now has 3,418 commits,
  17,413 objects, an 8,860,670-byte pack, and a 9,682,158-byte complete `.git`
  file sum. All 25 selected heads and 28 tags remain strict-`fsck` clean,
  source-neutral where required, and free of reachable AI attribution markers.
- The remaining largest packed blob was the README turbulence loop. PR #95 now
  keeps the full six-second interval and both physical views as a 720-pixel,
  24-frame, 4-fps WebP: 346,234 becomes 224,766 bytes. Against the rendered
  source frames it has 36.49 dB PSNR and 1.93 mean absolute channel error;
  direct visual inspection keeps all labels and structures readable, and the
  full-rate release MP4 remains linked. The 123 release gates and 46 benchmark
  contracts pass; PR #103 carries the same preview plus six passing x64 Landau
  physics tests.
- PR #95 now also makes the QA equilibrium generator own a deterministic
  256-color median-cut PNG. The 1,561-by-1,189 initial/final LCFS and Boozer
  panel falls from 192,557 to 74,390 bytes with 48.24 dB PSNR and 0.129 mean
  absolute channel error. Full-size visual inspection found no lost label,
  contour, or surface structure. A palette/100-kB gate, 179 affected
  release/optimization/benchmark tests, and Ruff pass; PR #103 carries the same
  compact panel and retains its six passing x64 Landau tests.
- Rewrote both preview blobs across the private rehearsal and replayed the
  preceding exact roadmap tree. The fresh no-alternates clone has all 25
  selected heads, 28 tags, 3,422 commits, and 17,433 objects; its pack is
  8,641,725 bytes and
  complete `.git` file sum is 9,463,873 bytes. Strict `fsck`, exact roadmap
  source neutrality/tree identity, and the attribution scan pass. The old
  828,066/346,234-byte loop and 192,557-byte QA panel are unreachable. No public
  ref moved.
- Wrote and mechanically validated a local 53-row public-to-candidate ref map
  for the 25 selected branches and 28 tags, then built and verified an
  8,635,427-byte complete candidate bundle. Ref-map SHA-256:
  `c2fe14c1f07ed0b3ada00a021c7d89ab4a22853cc0157cc4a41f26d2d8f1bb35`;
  candidate-bundle SHA-256:
  `cfb73f915a6cd959c7b6f0c279210f9942caecbe96c4e93ddf4d0f09192de294`.
  These are local rehearsal artifacts, not a cutover manifest: re-freeze and
  revalidate every GitHub SHA immediately before any coordinated ref move.
- Merged PR #81's CI-only cleanup margin into PR #91 after every nonlinear
  assertion passed but the old 15-minute wrapper killed that branch during
  teardown. The public nonlinear rerun passes in 15m08s, eight seconds beyond
  the old wrapper, and all 41 checks including `ci-required` are green. The
  continuation logic is unchanged.
  Replaying the same merge in the private candidate gives 25 heads, 28 tags,
  3,423 commits, and 17,435 objects. A fresh no-alternates clone has an
  8,632,357-byte pack and 9,454,569-byte complete `.git`; strict `fsck` and the
  reachable-attribution scan pass. The earlier ref map and bundle remain
  rehearsal snapshots and must not be used as the final cutover manifest.
- Replotted the user's saved `96x96x48` QA bundle without rerunning it. The old
  summary and NetCDF both stop at `t=200.6201`, but the rejected saturation
  report reaches `t=238.0251`: the pre-#84 terminal 1,024-step chunk was checked
  before its saved diagnostics were truncated. PR #84's exact remaining-time
  terminal chunk therefore fixes a demonstrated state/trace/statistics defect,
  not only a synthetic edge case.
- The same replot found that PR #94 removed the rejected window only from the
  standalone figures; the one-page summary still shaded and averaged it.
  Commit `b7807032` applies the fail-closed window to all embedded data panels
  while retaining the rejected interval and verdict in metadata. Four summary
  tests, the CLI regression, and a real-bundle render pass. The actual
  second-half view (`t=100.3--200.6`) reports heat-flux and potential
  `ky` tail/peak ratios 0.65 and 0.36, so both need Ny refinement.
- PRs #95 and #103 are fully green after their compact-render and self-contained
  manifest reruns. The clean no-alternates candidate also completes all 2,554
  collected x64 CPU tests and configured mypy. Replaying the new #94 commit
  leaves 25 heads, 28 tags, 3,424 commits, and 17,445 objects; a fresh clone has
  an 8,648,640-byte pack and 9,471,172-byte complete `.git`, with strict `fsck`
  and the reachable-attribution scan passing.
- The source-pinned QA `96x160x48`, seed-31 holdout reached exact `t=250` in
  5,108.6 s with 756 samples. The current full-suffix selector says saturated
  (`Q=11.5032 +/- 0.4196`, 3.65%), but the frozen `75+60` rule makes no stop:
  its final pass island is only `t=220.54--249.67`. A continuation to absolute
  `t=350` is queued without changing source or thresholds.
- On the matched `t=155.844--230.844` window, Ny=128 and Ny=160 give
  `Q=10.9140 +/- 0.4563` and `11.0353 +/- 0.2539`: a 1.11% difference, only
  0.23 combined SEM, with Q/Wphi/Wg half-window stationarity at both
  resolutions. Over the terminal `t=175--250` window the difference grows to
  5.51% (1.14 combined SEM), so this is compatibility, not a convergence
  declaration.
- Raising `ky*rho_i|max` from 2.0 to 2.524 lowers the `t=175--250` heat-flux
  cutoff/peak from 13.87% to 7.55% and last-three-ky mass from 3.04% to 1.45%;
  outer-six-kx mass stays 0.11%. Phi2 cutoff/peak is 1.12%/1.18%. Thus Ny=160
  clears the frozen 10% cutoff screen, but time persistence and an independent
  seed remain open. SHA-256: NPZ
  `4564fd5f9d50f441623040c9329bf22e498f4d7fbadf4ef25abbf195842ad185`, JSON
  `05af33be9bf3f4645750c42e32a32db0ab754d3a0c161537c396066c8f6784e5`, log
  `c1d0fca85be22b465b1d20e41f4cb1cdb0e06ce6152155e19ef98940f88ce49b`.
- GPU 0 now runs the independent QA Ny=96 seed-31 rung and then the clean QHS
  Ny=128 exact `t=750` match; GPU 1 runs QI Ny=128 seed 22. Exact-state QI
  continuation to `t=500` and QA Ny=160 continuation to `t=350` are queued
  behind those jobs, with unique output names and no artifact overwrite.
- PR #94 is fully green after the one-page-summary correction: all 41 current
  checks pass, with only the intended nightly job skipped. PR #74's current run
  is also fully green; its GitHub rollup retains a cancelled pre-rebase run, but
  the later run and `ci-required` both pass. No CI defect remains hidden there.
- Reconciled the top-level roadmap with the current 24 open PRs through #103,
  the source-pinned QA Ny=128/160 evidence, and the latest 8,648,640-byte
  all-live-ref rehearsal. Older 20-head measurements are now explicitly
  historical; no solver source or public ref changed.
- The source-pinned QI `96x128x48`, seed-22 rung reached exact `t=250` in
  4,959.3 s with 968 samples. The frozen `75+60` rule makes no stop; its only
  late pass island is `t=235.03--250`. The shipped selector also rejects the
  run at 24.3% relative SEM with Wg nonstationary.
- On `t=175--250`, Ny=96 and Ny=128 both pass Q/Wphi/Wg half-window gates but
  give `Q=4.1379 +/- 0.1213` and `3.5087 +/- 0.0446`: a 15.21% reduction, 4.87
  combined SEM. Heat-flux cutoff/peak falls from 14.80% to 4.20% and
  last-three-ky mass from 3.71% to 0.95%. Thus Ny=128 clears the necessary
  spectral screen while proving Ny=96 transport unconverged; the exact-state
  Ny=128 continuation to `t=500` started automatically without overlap.
- QI Ny=128 SHA-256: NPZ
  `9d8f3062389780536ce4df7497b03905b2038ad2f64d75895f7891afaedbc52b`, JSON
  `1724347485b1996cc4d0ec1f66e68473d1f6c601cebf0e3bfe6ba4289bd60933`, log
  `730c377f3b57fdf57ad96c45e3d98ef57487e44173d72f36696b617dab298168`.
- Auditing the private PR #82 replay found that its source-neutral snapshot
  contained `plan/` but omitted the root `plan.md`. Rebuilt that one candidate
  ref atomically on the slim CI base with both exact public trees; all non-plan
  paths remain byte-identical to the candidate CI head. No public ref moved.
- A fresh no-alternates clone of the corrected candidate has 25 heads, 28 tags,
  3,424 commits, and 17,446 objects. Its pack is 8,654,462 bytes, pack plus
  index 9,144,022 bytes, complete `.git` 9,476,998 bytes, and current archive
  4,888,710 bytes. Strict `fsck`, exact plan tree/blob identity, zero alternates,
  and the reachable AI-attribution scan pass.
- The source-pinned QA `96x96x48`, seed-31 rung reached exact `t=250` in
  2,200.7 s with 547 samples. The shipped selector rejects it at 11.8% relative
  SEM; the frozen `75+60` rule also makes no stop, with a longest pass island of
  only `t=193.23--243.55` (50.33 time units).
- On the Ny128 frozen-stop window, QA Ny96/128/160 give
  `Q=11.6622/10.9140/11.0353`. Heat-flux cutoff/peak falls from
  41.78% to 13.87% to 7.55%, and last-three-ky mass from 11.53% to 3.04% to
  1.45%. Only Ny160 clears the necessary spatial screen, and it fails temporal
  persistence. The stop decision is therefore not resolution-stable.
- QA Ny96 SHA-256: NPZ
  `41b8e641896d5830cee03c45103d52851e8d83e35a68c4340d3a7aa902d647cc`, JSON
  `6daf27302fd872a3364a6f3d6bbdae27c61dc25be6d0b573cc7cdd8dd3a8df9a`, log
  `f2c054a10b053be97eb315d26e284bc0bf9392049d1c3a92f06af7e038548f8f`.
- The ad-hoc QA-to-QHS wrapper used `&&`; the fixed-horizon QA audit correctly
  returned status 1 for NOT SATURATED after writing valid artifacts, so QHS did
  not start. This is an orchestration error, not a solver failure. Launched the
  source-clean QHS Ny128 `t=750` match explicitly on the now-idle GPU 0; the
  queued QA Ny160 continuation remains gated on its report.
- PR #91 commit `067788f0` makes the frozen `75+60` policy replay reproducible
  without adding another campaign file. It folds source-pinned continuation
  loading, causal pass islands, persistence, and implementation/input digests
  into `nonlinear_saturated_state.py`, then deletes the temporary standalone
  replay tool. The follow-up is 239 additions and 284 deletions relative to its
  parent. Ruff, 95 nonlinear validation tests, architecture, size, and diff
  checks pass; the standard local mypy run reaches one unrelated JAX-stub
  mismatch in `objectives/core.py`, so the pinned GitHub job remains the type
  authority.
- Regenerated every frozen-policy record against that committed owner. Only QA
  Ny128 stops (`t=230.9865`); QA Ny96/Ny160, QI Ny96/Ny128, and QHS Ny160 do
  not. SHA-256 prefixes are QA `9820207a/267d6800/7b478f81`, QI
  `069fdeaf/88fca1ff`, and QHS `3210a3c9`; the full report digests are in
  `plan.md`, with source/implementation digests inside each JSON artifact.
- Rechecked GitHub after the replay push: `main` remains `5f3ab32e`, no PR has
  merged since #80, and the living roadmap remains open. PR #91 is rerunning
  all required checks at `067788f0`; no result is inferred from queued jobs.
  GPU 1 continues the clean QI Ny128 state toward absolute `t=500`, then runs
  QA Ny160 seed 22 and the `cfl=0.5` seed-31 timestep check. GPU 0 runs the clean
  QHS Ny128 match to `t=750`, then continues QA Ny160 seed 31 to `t=350`.
- Replayed both new PR #91 commits and the exact preceding roadmap tree into
  the private rewrite candidate. The public/candidate stable patch IDs match
  (`0f1a8dff` and `aa037281`); the public and candidate `plan.md` blob and
  `plan/` tree IDs are exact, and non-plan roadmap paths are unchanged from the
  rewritten CI base. A fresh no-local clone has 25 heads, 28 tags, 3,427
  commits, 17,469 objects, an 8,723,042-byte pack, a 490,204-byte index, and a
  9,546,342-byte complete `.git` file sum. Strict `fsck`, zero alternates, and
  the reachable-metadata attribution scan pass. The rehearsal ref map and
  bundle remain stale by design; no public ref moved.
- Independent review then found that replay accepted any increasing same-commit
  trace pair, even from different physics cases, and that same-shaped restart
  states could likewise cross case boundaries. PR #91 commits `4cdc8fb6` and
  `85026529` fail closed on both paths. New artifacts fingerprint the deck and
  VMEC file and record case, grid, moments, seed, `alpha`, `npol`, and spectral
  axes; legacy continuation traces must at least form an exact declared endpoint
  chain with matching recorded identity. Compatible tool-only commits still
  pass through exact `src/gkx` tree comparison.
- All 96 nonlinear validation tests, Ruff, architecture, size, and diff checks
  pass locally. Replayed accepted legacy QI/QHS continuations retain every pass
  island and stop verdict. Mixed-case, mismatched-endpoint, incomplete-source,
  and incompatible-state tests reject. The regenerated replay-report SHA-256
  values are recorded in `plan.md`; full required CI restarted at `85026529`.
- A clean `4x4x4`, `(Nl,Nm)=(1,2)` CPU write/restart/replay smoke then exercised
  the real path at `85026529`. The first segment reached `t=0.1`, the accepted
  restart reached absolute `t=0.2`, and the joined four-sample trace replayed
  without identity loss. Both trace files and the continuation state contain
  the same schema, deck digest, grid, moments, seed, `alpha`, and `npol`.
- Adversarial replay of the real legacy files found that QA Ny96 `t=250` plus
  the QI Ny96 continuation still shared every old NPZ identity field and was
  accepted. PR #91 commit `0de845c2` now requires one ordered companion summary
  per legacy segment. It verifies clean matching source provenance and either
  the addressed NPZ digest or every scalar sample, then compares canonical
  case/grid/geometry/seed/field-line identity. The genuine QI and QHS histories
  reproduce unchanged; the mixed QA/QI history rejects. The report hashes in
  `plan.md` now bind both NPZ and JSON inputs.
- A 2026 primary-source refresh adds Wei et al.'s low-dimensional QH geometry
  result as a future surrogate/design-of-experiments option. It does not replace
  nonlinear labels. Re-reading Kim et al. also strengthens the present gates:
  their post-processing used `128x128`, `(Nl,Nm)=(8,16)`, two poloidal turns,
  and still found about 50% variation across field-line label.
- The source-pinned QI Ny128 exact-state continuation reached absolute `t=500`
  in 4,902.5 s. The frozen replay over both segments still makes no stop: six
  pass islands last at most 27.48 time units, and the terminal island lasts
  only 8.19. Its trailing 75-unit window passes Q/Wphi/Wg and gives
  `Q=3.4484 +/- 0.0224`, so the non-stop is a causal-persistence result rather
  than a high-SEM failure.
- On the matched `t=400--500` suffix, QI Ny96 and Ny128 give
  `Q=4.1869 +/- 0.0822` and `3.4194 +/- 0.0333`; the 18.33% change is 8.65
  combined SEM. Ny128 also passes the necessary heat-flux spectral screen at
  4.38% cutoff/peak, 0.97% last-three-ky mass, and 3.19% outer-six-kx mass.
  Ny96 is therefore rejected for QI transport, while no causal saturation
  value is promoted. Continuation SHA-256: NPZ
  `46627a0d9e5256119a80debd0827864481d033c844ecc64b173f3d188689285b`, JSON
  `ef89e389591c1e286754bd4708707eb46dca408571d6c19523a9cd24f5d5fa9e`, log
  `01f66c52ae2c5e97bb5732570489a4d387c9f4af001f057fbc4f71ff30d9438b`.
  The bound frozen-replay report SHA-256 is
  `c8133f3c1de07c430bbef20c4a19629253fe447ca0031d18f9627f5dcbff4ff3`.
- Replayed PR #91 through `0de845c2` and roadmap commit `b6085a4a` into the
  private rewrite rehearsal. The roadmap patch IDs match, and its public and
  candidate `plan.md` blob and `plan/` tree IDs are exact. A fresh no-local,
  no-alternates clone now has 25 heads, 28 tags, 3,432 commits, and 17,508
  objects; pack 8,746,756 bytes, pack plus index 9,238,052 bytes, and complete
  `.git` file sum 9,571,276 bytes. Strict `fsck` and the reachable AI-metadata
  scan pass with zero hits. No public ref moved; the recovery bundle and ref
  map remain intentionally stale until the coordinated freeze.
- PR #91 is fully green at `0de845c2`: all required checks, 24 wide shards,
  coverage, packaging, mypy, Python floor, and the aggregate gate pass. An
  independent PR #94 audit found one remaining documentation mismatch: rejected
  runs suppress the time-domain mean and shading, while spectra deliberately use
  a labelled second-half diagnostic window. Commit `2d5f3cc8` states that
  behavior exactly; its restarted CI is still in progress.
- The private all-live-ref rewrite rehearsal now also replays roadmap commit
  `c9e9cfce` and PR #94 commit `2d5f3cc8`. A fresh no-local, no-alternates clone
  has 3,434 commits and 17,519 objects; pack 8,755,505 bytes, pack plus index
  9,247,109 bytes, and complete `.git` file sum 9,580,377 bytes. Strict `fsck`
  and the reachable AI-metadata scan pass with zero hits. These measurements
  precede this record and remain private; no public ref moved.
- The 2026 source refresh adds GyroSwin and physics-informed neural compression.
  GyroSwin's 241-simulation adiabatic-electron GKW training set and stable learned
  rollouts make it a useful future comparator, but accumulated rollout error,
  smoothed zonal structure, and weak high-`ky` behavior exclude it from current
  transport or derivative evidence. Neural compression is relevant only after
  lossless compact traces; lossy data cannot support regression acceptance
  without preserving every statistical, spectral, restart, and AD observable.
- The source-pinned QHS `64x128x48`, seed-22 match reached exact `t=750` in
  6,224.6 s with 1,740 samples and clean `f7da8c49` provenance. Remote/local
  SHA-256 values match: NPZ `8b32d886`, JSON `a698489a`, log `db3870bc`.
  The bound frozen replay has 14 pass islands, no stop, and a longest duration
  of 48.39 time units; its report SHA-256 is `eb660d56`.
- Over the matched `t=650--750` suffix, QHS Ny128 and Ny160 give
  `Q=6.1130 +/- 0.1260` and `6.1947 +/- 0.0768`: a -1.32% difference, or 0.55
  combined SEM. Ny128 nevertheless fails every Q/Wphi/Wg half-window guard and
  has 10.57% heat-flux cutoff/peak versus 6.46% at Ny160; last-three-`ky` mass
  is 1.78% versus 0.94%. The means are compatible, but Ny128 is not an accepted
  spatial or temporal rung. Ny160 remains the minimum QHS resolution, and no
  causal saturated QHS value is promoted.
- GPU 0 automatically advanced to the exact QA Ny160 seed-31 continuation from
  `t=250` to `350`; GPU 1 continues the independent Ny160 seed-22 run, followed
  by the predeclared seed-31 `cfl=0.5` timestep check. Output names are disjoint.
- Replayed roadmap commit `a51115de` into the private all-live-ref candidate.
  Its stable patch ID and exact `plan.md`/`plan/` objects match the public tree.
  A fresh no-local, no-alternates clone has 25 heads, 28 tags, 3,435 commits,
  and 17,525 objects; pack 8,758,866 bytes, pack plus index 9,250,638 bytes,
  and complete `.git` file sum 9,583,930 bytes. Strict `fsck` and the reachable
  AI-metadata scan pass with zero hits. This measurement precedes this record;
  no public history moved.
- PR #94 commit `2d5f3cc8` is fully green after the independent documentation
  correction: all 24 wide shards, long quick-test jobs, coverage, and aggregate
  gate pass. The PR remains open and unmerged.
- The exact QA Ny160 seed-31 continuation reached absolute `t=350` in another
  2,100.3 s. The joined history makes no frozen stop; its longest pass island is
  29.13 and terminal persistence only 6.31. Over `t=250--350`,
  `Q=10.9902 +/- 0.3072`, all three stationarity guards pass, and heat
  cutoff/peak is 9.98%. The shipped selector passes only after retaining
  `t=24.95--350`, with `Q=11.3449 +/- 0.3128`. NPZ/JSON/log SHA-256 prefixes
  are `d1658274/45917775/bf70889c`; bound replay is `5542c67c`.
- The independent source-pinned QA Ny160 seed-22 run reached exact `t=250` in
  5,007.6 s. It makes no frozen stop; pass islands last at most 42.23, and its
  terminal Wphi/Wg fail stationarity. Over `t=150--250`, seed 22 and seed 31
  give `Q=10.4833 +/- 0.1734` and `11.2606 +/- 0.4042`, a -6.90% difference or
  1.77 combined SEM. Heat cutoff/peak is 12.24% versus 9.46%, so Ny160 is not
  yet a seed-robust accepted QA rung. NPZ/JSON/log prefixes are
  `0f31911b/14715807/6747c8f5`; bound replay is `78164caf`.
- GPU 0 now continues the exact seed-22 state to `t=350`; GPU 1 runs the
  predeclared seed-31 `cfl=0.5` check. Thresholds, deck, source, and output
  cadence remain frozen.
- The exact QA Ny160 seed-22 continuation reached absolute `t=350` in another
  2,065.1 s. The frozen policy stops at `t=312.194` with
  `Q=10.5986 +/- 0.2300` after 60.16 time units of persistence, but the same
  island ends at `t=345.839` and the terminal window fails again. Seed 31 still
  makes no stop through `t=350`, so the rule is not seed-robust and is not
  promoted. Continuation NPZ/JSON/log prefixes are
  `4954bac9/29b96bf8/b9f56670`; bound replay is `607b3271`.
- Over the matched `t=250--350` suffix, seed 22 and seed 31 give
  `Q=10.7420 +/- 0.2200` and `10.9902 +/- 0.3072`, a -2.26% difference or 0.66
  combined SEM. Both heat cutoff/peak values now pass at 8.77% and 9.98%; the
  long-window mean is seed-compatible, while the causal stop is not.
- GPU 0 now runs the source-pinned Ny192 seed-31 rung to `t=250`, providing a
  finest-grid comparison beyond the borderline Ny160 tail. GPU 1 continues the
  Ny160 seed-31 `cfl=0.5` control. No threshold was changed after observing the
  seed histories.
- Independent review of PR #85 found that a glossary branch also changed the
  shipped wout deck to `run_to = "saturation"`. That default is unsupported by
  the seed/resolution failures above. Commit `33d24d6d` removes the default
  flip, keeps the three-line glossary, and passes the two focused runtime tests,
  Ruff, and `git diff --check`.
- Replayed that correction into private rewrite commit `79109d8c`. Its stable
  patch ID `db5c853a960ef9d73ebd8d257b6200e76fa19316` exactly matches the
  public correction; no rewritten public ref moved.
- A post-hoc QA Ny160 window scan is design evidence only. A 125-time-unit
  trailing window plus 30-time-unit persistence would stop seed 22 at
  `t=269.5` and seed 31 at `t=304.2`, but it was selected after seeing both
  traces. The frozen `75+60` rule remains unchanged; QA seed 33 is queued as an
  untouched holdout after Ny192, and a QI Ny160 resolution rung is queued after
  the half-CFL control.
- The source-pinned QA `96x192x48`, seed-31 rung reached exact `t=250` in
  6,594.6 s with 843 samples. Remote/local SHA-256 values match: NPZ
  `289c2f30`, JSON `645b323c`, and log `e0ce2d90`; the bound frozen replay is
  `2996d20f`.
- The Ny192 frozen rule stops at `t=218.783` after 60.17 time units of
  persistence, while Ny160 at the same seed makes no stop. On `t=150--250`,
  Ny160/Ny192 give `Q=11.2606 +/- 0.4042` and `11.1314 +/- 0.2702`, a -1.15%
  change or 0.27 combined SEM. Ny192 Wg fails the full-window half gate, though
  the terminal 75-unit Q/Wphi/Wg window passes.
- Heat cutoff/peak improves from 9.46% at Ny160 to 6.12% at Ny192;
  last-three-`ky` mass falls 1.49% to 1.06%, Phi2 cutoff/peak 11.61% to 7.37%,
  and outer-six-`kx` heat mass remains 0.12%. This supports spatial mean
  compatibility, not a resolution-robust temporal stop.
- Independent review of PR #91 found that timestep overrides were absent from
  machine-readable artifact identity. Commit `35aa7d29` records resolved
  fixed/adaptive mode, requested dt, dt_max, CFL, and method in schema v2. A
  clean VMEC CPU write/restart/replay smoke reaches absolute `t=0.2`; changing
  only CFL rejects before integration. The 174 focused/release tests, Ruff,
  and `git diff --check` pass. Private replay `1dbc39bd` has the same stable
  patch ID `7c96fb49c64e9c6091185b7000f4d820fda5c975`.
- The source-pinned QA Ny160 seed-31 `cfl=0.5` control reached exact `t=250`
  in 9,997.4 s with 1,505 samples. Against `cfl=1`, median adaptive dt halves
  from 0.03255 to 0.01636 and wall time rises by 1.96x. On `t=150--250`,
  `Q=10.6370 +/- 0.2368`, a -5.54% change or 1.33 combined SEM; Q/Wphi/Wg and
  the spatial screens remain compatible. The frozen rule still makes no stop:
  its terminal pass island lasts 59.43 rather than 60 time units. NPZ/JSON/log
  SHA-256 prefixes are `bc3b7b47/d33d01e8/fe3de89c`; the bound replay is
  `6ece6ecf`.
- PR #85 and PR #91 are fully green at `33d24d6d` and `35aa7d29`. The former
  retains the validated fixed-horizon default; the latter records timestep
  identity and rejects cross-policy restarts. Neither PR is merged.
- Replayed the matched-CFL evidence and current roadmap into the private
  rewrite through `bc61d222`. A fresh no-local, no-alternates clone of main,
  all 24 open PR heads, and 28 tags has 3,444 commits and 17,572 objects; pack
  8,793,975 bytes, pack plus index 9,287,063 bytes, and complete `.git` file
  sum 9,620,543 bytes. Strict `fsck` and the reachable AI-attribution scan pass
  with zero hits. No public history moved.
- A July/August 2026 primary-source refresh found no nonlinear gyrokinetic
  shadowing or stationary-adjoint result beyond the already audited iGENE,
  stabilized-march, online-gradient-flow, and wall-turbulence studies. The
  finite-window discrete adjoint therefore remains the one supported GKX API;
  no inaccessible paper is presently blocking the audit.
- The untouched QA Ny160 seed-33 holdout completed exact `t=350`; GPU 0 advanced
  to the queued QHS Ny160 run while GPU 1 continues the QI Ny160 rung. All
  use the same clean source-pinned `f7da8c49` checkout and frozen policy.
- The supplied QA artifact demonstrates the pre-#84 horizon defect exactly:
  saved diagnostics end at `t=200.620`, while the rejected decision uses a
  window through `t=238.025`. PR #84 commit `55d41c09` now asserts that a
  rejected decision ends at the final saved diagnostic even under output
  striding. All 15 chunk-loop tests, Ruff, and all 41 CI checks pass.
- Replayed that regression privately as `bf8d5429`. The complete public and
  candidate PR #84 patches share stable ID `d35a7a70`. A fresh no-local
  candidate clone advertises 25 heads and 28 tags, with 3,446 commits, 17,582
  objects, an 8,778,044-byte pack, and a 9,604,932-byte complete `.git`.
  Strict `fsck`, no alternates, and zero AI-attribution hits pass; no public
  history moved.
- Solved the accepted QA baseline/candidate vacuum boundaries from exact GKX
  inputs using clean VMEX `0362f701`, `ns=101`, `ftol=1e-10`. Both
  converge with `ier_flag=0`; WOUT hashes are `323dd3ef` and `58da1b89`.
  The hashed TOML manifest records inputs, residuals, geometry scalars, and
  sources. Its regenerated 3D-LCFS/Boozer panel (`f4cd87ab`) exactly matches
  the tracked figure. A matched production seed-31 pair is queued after QI.
- PR #91 became conflicting only because its base gained PR #84's exact-horizon
  regression. Merge `14442da2` resolves the overlap by testing the saved-horizon
  identity together with the Q/Wphi/Wg callback. All 42 focused contract tests,
  Ruff, diff checks, and all 41 CI checks pass. Private replay
  `0b1d6ced` has the exact full-PR stable patch ID `5f92a366`. A fresh
  no-local 25-head/28-tag clone has 3,448 commits, 17,594 objects, an
  8,749,159-byte pack, and a 9,576,463-byte complete `.git`; strict `fsck`,
  zero alternates, and zero AI-attribution matches pass. No public history moved.
- Re-reviewed PRs #86/#96/#97 and the source-pinned QA movie. The 122 focused
  geometry/artifact tests and Ruff pass; 48-point finite VMEC coordinates,
  `nfp=2`, open-tube topology, evolving cuts, and the 218,799-byte encoding are
  correct. Schema 3 nevertheless copies only the source path and saturation
  flag, not the source-state hash or full campaign identity. The artifact is
  accepted as a rendering/continuation test, not transport evidence. MOV-1 now
  requires PR #91 identity agreement plus a persisted source-state hash.
- Froze the untouched QA Ny160 seed-33 artifacts at exact `t=350`: NPZ/JSON/log
  hashes start `57a9736b/e891acdc/35d5c050`, wall time is 7,106.8 s, and all
  1,055 scalar samples are finite. The standard `75+60` replay makes no stop.
  The preregistered `125+30` rule stops at `t=184.186`; Q/Wphi/Wg pass there,
  heat-tail ratio is 9.924%, and its Q is within 1.39% or 0.31 combined SEM of
  the unseen final mean. The hypothesis still fails its frozen fifth criterion:
  final `t=250--350` Wphi and Wg are nonstationary. No threshold was retuned.
  Seed-33 terminal Q agrees with seeds 22/31 within 0.015/0.46 combined SEM.
- Found two live QI Ny160 processes with identical output paths: the original
  began at 10:06 and the accidental duplicate at 11:47. The younger writer had
  already truncated the shared log. Terminated only PIDs 1144813/1144812/
  1144811; original PID 1141048 remains healthy and is now the sole GPU-1
  writer. Its partial trace is sizing-only; log/wall evidence is invalid and
  acceptance requires a clean repeat.
- At `t~=274.4/500`, the inadmissible QI sizing run still needed about three
  GPU-hours. Preserved its partial log, stopped only PID 1141048 and its waiting
  QA wrapper, and launched the accepted VMEX baseline/candidate seed-31 pair on
  GPU 1. QHS seed 31 was at `t~=667.5/750`; its wrapper will start the clean,
  uniquely named QI repeat on GPU 0 after completion. No partial trace is used
  as physics evidence.
- PR #91 commit `eebff63b` prevents recurrence with POSIX advisory locks on
  every requested summary/trace/state path. Conflicts fail before simulation
  with owner PID/host; stale files do not retain kernel locks. All 43 focused
  tests, Ruff, and diff checks pass. Private replay `eab1327d` has identical
  patch ID `1243ab2c`; the full PR #91 public/candidate patch ID is `4c2b67d8`.
  A fresh 25-head/28-tag no-local clone has 3,451 commits, 17,614 objects, an
  8,700,585-byte pack, and a 9,528,529-byte complete `.git`; strict `fsck`,
  no alternates, and zero AI-attribution matches pass.
- Independently audited open PRs #98 and #99. PR #98 removes only three exact
  duplicates of the canonical dealiased spectral-layout helpers; array order
  and every call site are unchanged, and 37 runtime-artifact tests, Ruff, and
  mypy pass. PR #99 matches its parent bit-for-bit across randomized
  float32/float64, complex64/complex128, nonfinite-signal, and four bounded-fit
  windows, including R-squared and autocorrelation-corrected uncertainties;
  all 52 diagnostic tests and static checks pass. No defect was found in either
  small source-slimming patch.
- Independently audited open PRs #102 and #103. The initial PR #102 grid
  compaction is faithful, but full regeneration exposed two defects missed by
  its original focused tests: all five scripts used `parents[2]` and therefore
  treated `examples/` as the repository root, and tracked grid/backend
  provenance was stale. Commit `51b55741` fixes the root, regenerates current
  72-by-72 metadata, and replaces the second full nonlinear trace with a
  value-checked reference. The nonlinear JSON is now 87,377 bytes instead of
  423,062; the comparison JSON is 247,507 instead of 273,704. Five focused
  tests, isolated clean-checkout `--help` runs, full x64 regeneration, Ruff,
  the size gate, and diff checks pass. Histories, density scans, retained
  traces, and physics values match. Private replay `eed7eee3` has the exact
  public patch ID; roadmap replay `8349f25d` has the exact public patch ID and
  roadmap payload. A fresh pre-record clone has 3,456 commits, 17,656 objects,
  an 8,960,103-byte pack, and a 9,789,391-byte complete `.git`; strict `fsck`,
  no alternates, and the attribution scan pass. PR #103 preserves the Landau
  panel at 3,028 by 822
  pixels with 255 colors and shrinks it from 371,750 to 151,626 bytes. Against
  the prior RGBA render, mean absolute channel error is 0.066 and PSNR is
  49.86 dB; visual review and all six x64 Landau physics tests pass. Neither PR
  changes a numerical or physics result.
- Clarified the repository-size contract. The private full-clone Git database
  is below the decimal 10-MB gate; PR #104 reduces the expanded stacked tracked
  tree from 19,162,255 to 18,904,270 bytes. A sub-10-MB expanded checkout
  remains open and
  must be reached through provenance-preserving artifact migration; the
  3.35-MB installed source is not the cause of the original 133.70-MiB clone.
- Opened draft PR #104 on #103 to centralize deterministic PNG palette output
  and compact the retained runtime/memory, linear-parity, and eigensolver-reach
  README figures from 462,973 to 205,273 bytes. A final audit rejected locally
  regenerated canvases that differed from the tracked parent by 2--7 pixels;
  commit `885729f8` instead palette-encodes the exact parent canvases. Their
  same-canvas PSNR is 72.15/56.06/62.38 dB. CI caught the first helper version's
  16-line source-budget regression; `673e5da5` trims prose rather than raising
  the baseline, leaving `src/gkx` two lines smaller. Commit `5a74bbea` gates all
  four tracked palettes. Six Landau physics tests, all three generators, mypy,
  architecture, Ruff, size, release, and diff gates pass locally; the old and
  shared Landau helpers are byte-identical under one renderer.
- Replaced only the three exact superseded PNG blobs in a new private rewrite
  rehearsal. PR #104's aggregate text patch and every generator/source/image/
  physics-test head blob match; the release-gate file differs only in its
  intentional branch-only-roadmap comment. PR #105's aggregate patch is exact.
  A fresh ordinary clone has 27 heads plus `origin/HEAD`, 28 tags, 3,466
  commits, 17,728 objects, an 8,912,462-byte pack, and a 9,744,233-byte complete
  `.git`.
  Strict `fsck`, exact roadmap payload, no alternates, zero original/intermediate
  PNG reachability, and zero AI-attribution hits pass. No public history moved.
- Opened draft PR #105 on #81 after the exact-function audit found duplicate
  linear sampling validation and 5D/6D cache resolution. One canonical helper
  now takes each caller's local cache builder explicitly, preserving the
  monkeypatch seam exposed by the first test run. Commit `9204d1a8` also binds
  donated/nondonated integration through one wrapper factory while retaining
  trace-time implementation lookup. The full patch removes 52 source lines and
  is net four Python lines smaller after its two regressions; 133 linear, 74
  runner/runtime, and 69 four-device parallel tests pass. A reconstructed old
  wrapper is bitwise equal with identical JAX cost analysis. Ruff, mypy,
  architecture, and diff checks pass.
- PR #104 is fully green at `885729f8`: all 41 required checks pass and the
  nightly job is intentionally skipped.
- PR #102's follow-up audit/fix is fully green: all 41 required GitHub checks
  pass at `51b55741`; the nightly job is intentionally skipped.
- Refreshed the public inventory: `main` remains at merged PR #80 and 26 PRs
  are open (#74 and #81--#105). PR #74's apparent 39-check failure is a
  cancelled duplicate run on the same head; its later authoritative run has
  all 41 required checks passing. No PR has merged since #80.
- The untouched source-pinned QHS `64x160x48`, seed-31 run reached exact
  `t=750` in 8,469.3 s with 1,920 samples. Remote/local hashes match: NPZ
  `64d51d48`, JSON `742f2c01`, log `910b6a4b`; the remote 51-MB state is
  `914048a6`. The PR #91 frozen replay (`852d2da7`) makes no stop: 11 pass
  islands last at most 28.55 time units.
- Over `t=650--750`, QHS seed 22 and seed 31 give
  `Q=6.1947 +/- 0.0768` and `5.2766 +/- 0.0714`, a -14.82% change or 8.76
  combined SEM. Seed 31 passes the heat spectral screen at 7.10% cutoff/peak,
  1.02% last-three-`ky` mass, and 3.77% outer-six-`kx` mass. Initial Wphi/Wg
  agree within 0.23%, but late zonal-`Phi2` fractions are 94.76% and 88.42%.
  This is an unresolved seed/zonal-state dependence, not permission to average
  the seeds or shorten the window; no QHS transport value is promoted.
- Auditing the QHS-to-QI waiter before handoff found its cwd was `/home/rjorge`,
  so the relative `tools/campaigns/...` command would have failed after QHS.
  Stopped only the idle waiter and replaced it with absolute paths rooted at
  clean PR #91 commit `eebff63b`. The clean QI Ny160 run is active on GPU 0,
  imports that exact checkout, and holds all three output locks.
- PR #105 is fully green at `9204d1a8`: all 41 required checks pass and the
  nightly job is intentionally skipped.
- Replayed the QHS evidence as private commit `1be3af15`; its stable patch ID
  and both changed roadmap blobs exactly match public commit `baa35d70`. A
  fresh ordinary clone of all current candidate refs has 28 remote refs, 28
  tags, 3,469 commits, and 17,744 objects; pack 8,835,722 bytes, pack plus
  index 9,333,626 bytes, and complete `.git` 9,668,005 bytes. Strict `fsck`,
  no alternates, and zero AI-attribution metadata pass. This measurement
  precedes this record; no public history moved.
- A fresh source-deduplication attempt reproduced work already owned by draft
  PR #93. Closed the redundant draft #106 immediately and retained #93 as the
  sole review path. The independent #93 audit found one unused facade constant
  and no direct gate for the canonical signature/`__wrapped__` owner. Commit
  `b309a0af` removes the constant and adds that contract. The complete patch
  removes 29 installed lines and 23 Python lines overall; 55 direct geometry
  tests, all 117 release tests, Ruff, changed-module mypy, diff, and the
  96,436-line architecture gate pass. GitHub CI is active.
- Replayed #93 commit `b309a0af` privately as `1fe08146`. The complete public
  and private patches share stable ID `f214864b`; all three affected head blobs
  are identical. After strict `fsck` and garbage collection, a fresh ordinary
  clone has 28 remote refs, 28 tags, 3,471 commits, and 17,758 objects: pack
  8,736,180 bytes, pack plus index 9,234,476 bytes, and complete `.git` file
  sum 9,568,911 bytes. It has no alternates and zero reachable AI-attribution
  matches, leaving 431,089 bytes below the decimal 10-MB gate. No public
  history moved.
- Opened draft PR #107 at `96911c3b` for the next non-overlapping source cut.
  One frozen runtime fit-option record now owns the eleven fields previously
  redeclared by both single-run dispatch and ky-scan orchestration. Its mapping
  excludes subclass-only run/scan fields. Public signatures, defaults, JAX
  arguments, kernels, fit policy, and schemas are unchanged. The patch removes
  19 installed-source lines (`96,465 -> 96,446`) and adds no file. All 239
  runtime helper/runner/CLI tests, 117 release tests, four exact base/head
  signature comparisons, Ruff, changed-module mypy, architecture, repository
  size, and diff gates pass locally. GitHub CI is active.
- Replayed #107 privately as `168dc834`. Its exact public/private stable patch
  ID is `df901451`, and all four affected head blobs match. After garbage
  collection, a fresh ordinary clone has 29 remote refs, 28 tags, 3,473
  commits, and 17,777 objects: pack 8,687,046 bytes, pack plus index 9,185,874
  bytes, and complete `.git` file sum 9,520,481 bytes. Strict `fsck`, no
  alternates, and zero reachable AI-attribution matches pass, leaving 479,519
  bytes below the decimal 10-MB gate. No public history moved.
- PR #93 is fully green at `b309a0af`: all 41 required checks pass and nightly
  is intentionally skipped. PR #107 is likewise fully green at `96911c3b`.
- Opened draft PR #108 at `f7634a03` after the resolved nonlinear diagnostic
  audit found that heat and particle channel kernels each ran twice per sampled
  state. Totals now reuse the one ES/Apar/Bpar reduction. The 58-field schema
  and every channel array are unchanged; float64 total differences are at
  `6.34e-16` relative and float32's worst max-scaled difference is `2.52e-6`
  from reassociation. A warmed `2x4x8x32x32x24` CPU kernel falls from 0.742 to
  0.603 ms (-18.8%), with XLA FLOPs -7.6%, bytes accessed -4.5%, and JAXPR
  equations -46.4%. This is not an end-to-end solver timing claim. The patch
  removes ten installed-source lines and 34 test lines. All 102 owned
  diagnostic/runtime tests, 117 release tests, Ruff, changed-module mypy,
  architecture, size, and diff gates pass. Two broader float32-only failures
  reproduce identically on the untouched base under local JAX 0.9.2 and office
  JAX 0.11.1. Both pass on office under the actual CI
  `JAX_ENABLE_X64=true` contract, together with the three direct diagnostic
  state tests. CI is active, and the GPU benchmark waits for a free device.
- Replayed #108 privately as `8f2444d8`. Its exact public/private stable patch
  ID is `62fef1c4`, and all four affected head blobs match. A fresh ordinary
  clone has 30 remote refs, 28 tags, 3,475 commits, and 17,798 objects: pack
  8,690,625 bytes, pack plus index 9,190,041 bytes, and complete `.git` file
  sum 9,524,832 bytes. Strict `fsck`, no alternates, and zero reachable
  AI-attribution matches pass, leaving 475,168 bytes below the decimal 10-MB
  gate. No public history moved.
- Opened draft PR #109 at `0561e5dc` after an AST audit found exact copies of
  velocity-shape inference, state sharding, and packed-state placement in the
  linear and nonlinear Diffrax paths. Their existing core now owns that setup.
  Public signatures and the invalid-shape error are exact; solver, cache,
  physics-term, packed-layout, differentiation, and traced-arithmetic contracts
  are unchanged. Base/head two-step linear and nonlinear outputs, including
  single-device sharding, have byte-identical state and field-history hashes.
  The patch removes 19 installed-source lines and adds no file. All 14 focused
  Diffrax tests, 117 release tests, Ruff, mypy, architecture, size, and diff
  gates pass locally. CI is active.
- Replayed #109 privately as `90656d0c`. Its exact public/private stable patch
  ID is `b094d3f5`, and all three affected head blobs match. A fresh ordinary
  clone has 31 remote refs, 28 tags, 3,478 commits, and 17,818 objects: pack
  8,696,888 bytes, pack plus index 9,196,864 bytes, and complete `.git` file
  sum 9,531,837 bytes. Strict `fsck`, no alternates, and zero reachable
  AI-attribution matches pass, leaving 468,163 bytes below the decimal 10-MB
  gate. No public history moved.
- The accepted VMEX QA baseline reached exact `t=350` at `96x160x48`, seed 31,
  in 7,429.7 s. Its production median-crossing interval starts at `t=22.67`
  and is correctly rejected at 9.97% corrected relative SEM. The fixed
  terminal 75 units instead give `Q=8.0466 +/- 0.1739`, `tau_int=2.532`, and
  passing Q/Wphi/Wg half-window gates. Heat cutoff/peak, last-three-ky, and
  outer-six-kx masses are 1.96%, 0.375%, and 2.19%. The frozen `75+60` replay
  first stops at `t=244.341` with `Q=8.4230 +/- 0.2759`; the terminal mean is
  4.47% lower. This is one baseline sizing trace, not a stop-rule or transport-
  reduction claim. Remote/local hashes match: JSON `49ca6d7e`, NPZ `205dafd0`,
  state `050cadea`, log `6e399265`; replay `754f5d85`.
- The baseline returned the intended not-saturated verdict, but its candidate
  waiter used an unanchored `pgrep` that matched the waiter's own command line
  and exited. Output locks prevented duplicate writers. After an anchored
  exact-Python-process check showed no candidate and no artifact, started one
  candidate at source `f7da8c49` on GPU 1. Record the orchestration defect;
  partial candidate output is not evidence.
- Confirmed a separate progress-display defect: low-level step/time/ETA resets
  on every 128-step campaign chunk. The solver state is continuous. A future
  host-only fix must show cumulative time and one wall clock without changing
  traced arguments, diagnostic cadence, or compile reuse.
- PR #108 is fully green at `f7634a03`: all 41 required checks pass and nightly
  is intentionally skipped. Its noninterfering GPU diagnostic benchmark still
  waits for a free device; no end-to-end speedup claim is promoted.
- Replayed roadmap commit `d52afb2f` privately as `aede71a7`. Their stable
  patch ID is `7f53b322`, and both changed blobs match exactly. After garbage
  collection, a fresh no-local clone has 31 remote refs, 28 tags, 3,480
  commits, and 17,829 objects: pack 8,699,674 bytes, pack plus index 9,199,958
  bytes, and complete `.git` file sum 9,534,975 bytes. Strict `fsck`, no
  alternates, and zero reachable AI-attribution matches pass, leaving 465,025
  bytes below the decimal 10-MB gate. No public history moved.
- The source-pinned QA baseline's isolated `t=50--55` window has 18 samples,
  span 4.78 below `10 tau_int=7.88`, 8.68% corrected relative SEM, and failing
  Q/Wphi/Wg half-window gates. Its Q/Wphi/Wg means exceed the stationary
  terminal-75 means by 4.46%, 66.5%, and 31.0%. The full trace at `t=55` still
  has 22.1% relative SEM. A short Wphi average cannot certify heat transport.
- PR #91 commit `cd2a30f5` now labels compiled inner lines `[gkx:segment]` and
  forwards host-only cumulative chunk time/wall/ETA in the saturation campaign.
  It changes no traced argument, integration, diagnostic cadence, or physics.
  Owning tests, 117 release contracts, Ruff, mypy, architecture, size, and diff
  checks pass locally; CI is active. The running clean QI artifact remains
  source-pinned to `eebff63b` and retains its old labels.
- Replayed that PR #91 commit privately as `e2a36d0b`; stable patch ID
  `6100a7b1` and all five blobs match. A fresh no-local clone has 31 remote refs,
  28 tags, 3,482 commits, and 17,853 objects: pack 8,702,838 bytes, pack plus
  index 9,203,794 bytes, and complete `.git` file sum 9,538,907 bytes. Strict
  `fsck`, no alternates, and zero reachable AI-attribution matches pass, leaving
  461,093 bytes below the decimal 10-MB gate. No public history moved.
- PR #109 is fully green at `0561e5dc`: all 41 required checks pass and nightly
  is intentionally skipped.
- Opened draft PR #110 at `adf3234b` after the exact-AST audit found duplicate
  float32/x64 finite-difference tolerances and finite scalar parsing in the
  objective layer. Existing owners now serve both callers. Public signatures,
  defaults, table/report schemas, thresholds, and policy outputs are unchanged;
  float32/x64 base/head parity digests match. The patch removes 14 installed
  lines (`96,465 -> 96,451`) and adds no file. All 122 owning tests pass locally
  and on office JAX 0.11.1; 117 release tests, Ruff, architecture, size, and
  diff gates pass locally. CI is active. The office runtime environment lacks
  PyYAML, Ruff, and mypy, so their authoritative rerun remains GitHub CI.
- Replayed #110 privately as `9c0626f6`; stable patch ID `0f6549b3` and both
  changed blobs match. A fresh no-local clone has 32 remote refs, 28 tags,
  3,484 commits, and 17,865 objects: pack 8,704,955 bytes, pack plus index
  9,206,247 bytes, and complete `.git` file sum 9,541,515 bytes. Strict `fsck`,
  no alternates, and zero reachable AI-attribution matches pass, leaving
  458,485 bytes below the decimal 10-MB gate. No public history moved.
- GitHub squash-merged PR #81 externally as `0ff569c3` on 2026-08-22. It had
  all 41 required PR checks green and no approving review; `main`'s post-merge
  workflow is still running. The public inventory is now 74 merged and 29 open
  PRs (#74, #82--#105, and #107--#110). The living roadmap remains open.
- Mapped that squash into the private rewrite without pointing `main` at the
  topic branch. Private `00fb4dae` has parent `2f8521ab`, tree `b1c356ee`
  exactly equal to private topic `0701f268`, and the public squash message,
  timestamp, Rogerio author, and GitHub committer. Because private `main`
  already contained the one-line run-summary typing repair, this commit adds
  only the four-addition/five-deletion CI workflow delta. A fresh no-local
  clone has 32 remote refs, 28 tags, 3,486 commits, and 17,871 objects: pack
  8,524,088 bytes, pack plus index 9,025,548 bytes, and complete `.git`
  9,360,868 bytes. Strict `fsck`, no alternates, and zero reachable AI markers
  pass. No public ref was force-pushed.
- The post-merge `main` workflow at `0ff569c3` completed successfully: 38 push
  jobs and the aggregate `ci-required` gate passed; nightly was intentionally
  skipped. CI-1 is closed. PR #91 still has only its nonlinear and four-device
  shards running, and #110 remains active; neither has a failure.
- Audited the 18 open PRs still based directly on `fix/main-ci-mypy`. Its tree
  equals `main` at `0ff569c3`, but their merge base is pre-#81 `5f3ab32e`:
  changing only the base would make GitHub's three-dot diff show #81 again.
  Keep the base branch until coordinated cutover. Rebase each direct head from
  `d910ac56` onto rewritten main, prove exact head-tree and intended patch
  identity, push with `--force-with-lease`, then retarget and rerun CI. This
  avoids both misleading review diffs and a redundant pre-cutover force-push.
- PR #91 is fully green at `cd2a30f5`: all 41 required checks pass and nightly
  is intentionally skipped. Its nonlinear shard passed in 16m13s, which would
  have exceeded the former 15-minute wrapper and directly validates #81's
  nonlinear-only 20-minute cleanup budget. This closes the progress-display
  implementation gate; the source-pinned QI run remains active and unchanged.
