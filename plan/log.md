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
