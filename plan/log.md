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
  `kx`, `ky`, and resolved spectra. A local `16^2 x 16`, `t=100` CPU pilot took
  66.3 s and failed saturation with `Wg` still rising. A shared RTX A4000
  `32^2 x 32`, `t=250` pilot took 469.8 s and was also pre-saturation: mean `Q`
  rose from 0.147 over `t=100--150` to 2.45 over `150--200` and 7.25 over
  `200--250`; the highest retained positive-`ky` mode was the flux-spectrum
  maximum. These are diagnostic pilots, not timing baselines or converged
  transport results.
