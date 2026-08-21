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
