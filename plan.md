# GKX: focused research plan

Updated 2026-09-06. **Planning revision; no solver changes, merge, or release.**
Baseline: main [a99dac89](https://github.com/uwplasma/GKX/commit/a99dac898334414d31733f6d286bd4c36983702e).
This is the proposed execution authority once reviewed, not a claim that its gates pass.

## 1. Outcome and scope

Deliver a reproducible local gyrokinetic workflow that can distinguish a useful
stellarator design change from numerical error and sampling noise, with
trustworthy derivatives and measured CPU/GPU cost.

**First research result:** electrostatic ion-temperature-gradient (ITG) transport
in nested-surface equilibria, kinetic ions and adiabatic electrons, one validated
collision/closure policy, and vacuum QA optimization with finite `a/L_T,a/L_n`.
Validate a tokamak control and one stellarator before expanding geometry coverage.
Vacuum describes the equilibrium, not zero turbulence drive.

Keep the broader ambition: kinetic electrons, multispecies, electromagnetic
fluctuations, finite-beta equilibria, advanced collisions, flow, and coil fields.
They require separate admission tests; implementing every combination is **not**
the next milestone. A verified code is not automatically an experimentally
validated, predictive model.

| Priority | Outcome | Dependency |
| --- | --- | --- |
| P0 | One defensible baseline, resolved correctness regressions and claims | None |
| P1 | Reproducible ES transport and gradient benchmark | P0 |
| P2 | Faster validated QA optimization, using warm states where beneficial | P1 |
| P3 | Bounded Coulomb implementation with an honest shipping envelope | P0; does not block Dougherty-based P1/P2 |
| P4 | Usable research package, evidence bundle, manuscript and release review | Required P0–P3 gates or explicit scope decisions |

Work on **one correctness change and one bounded measurement at most**.
No unlimited resolution or algorithm sweeps. After each decision experiment,
record adopt / reject / unresolved and its next discriminating check.

## 2. What changed in this plan

- Replace the old architecture-first, version-number-driven program with
  **observable correctness → useful gradient → total optimization cost**.
- Keep one nonlinear derivative: checkpointed finite-window reverse AD.
  Distinguish it from implicit eigenvalue/equilibrium differentiation and from
  the unknown derivative of the invariant turbulent measure.
- Validate the warm-start code already present; do not create another framework.
- Restore the published GX absorber convention as a **comparison baseline**,
  subject to matched tests. Fixed-rate damping is not a release prerequisite.
- Retain field-aligned Hermite–Laguerre coordinates. Consider domain/parallel
  coordinate changes only when a measured resolution bottleneck warrants them.
- Stop expanding closure, absolute-quasilinear calibration, and arbitrary-order
  Coulomb interpolation experiments until the relevant baseline fails a
  discriminating test.
- Preserve successful refactoring; do not impose another folder migration or
  arbitrary LOC target before measuring duplicated numerical ownership.
- Fix contradictory public status now; preserve failed results and historical
  provenance, without treating old logs as the active queue.

Technical rationale, equations, comparative sources and decision experiments:
[research program](docs/research_grade_program.rst).
Fresh tests, code observations, PR ledger and reproduction:
[September 6 audit](plan/baseline/review_2026_09_06.md).

## 3. Baseline: what is and is not established

| Area | Main evidence | Limit affecting the plan |
| --- | --- | --- |
| Linear solver | Dense and matrix-free routes; implicit eigenmode derivatives; tracked scans | Mode selection, timestep/boundary conventions and resolution matter; CBC and KBM discrepancies remain |
| Nonlinear AD | Fixed-window RK objective; checkpoint/plain and finite-difference tests | Initial state is detached; no general long-time statistical derivative |
| Warm restart | `SaturationWarmStart` and prepared initial-state execution exist | QA example disables warm reuse; policy stores NumPy arrays; no demonstrated saturated warm-optimization saving |
| Transport statistics | IAT correction; heat-flux, field and distribution guards | Median-crossing window and repeated stopping remain heuristics; nonuniform-time and near-zero handling need qualification |
| QA result | Initial/final shapes, flux traces, nominal 12.26% reduction | 4/48 nominal traces fail drift; not a resolved transport-reduction claim |
| Coulomb | Drift-kinetic low-moment and finite-wavelength 8/18-moment tables | Finite-wavelength coefficient/derivative corrections are unmerged; not arbitrary-order or unlike-species |
| Parallel | Independent scans/ensembles; explicit species–Hermite implementation | Full nonlinear/AD scaling is not generally qualified; split-Hermite conserving collisions are refused |
| Geometry/EM | Analytic/Miller/VMEC paths, EM equations and small tests | Neither arbitrary island transport nor broad finite-beta nonlinear validation is established |
| CI | Main CI run 33752936621 succeeded | Fresh CPU precision-specific test failure still exists; green CI is not a scientific certificate |

Measured main: **184 Python source files, 88,910 physical source lines,
562 tracked JSON files, 503 README lines**. Counts include blank/comment lines;
they are baselines, not complexity or performance scores.

## 4. P0 — settle the baseline before more expensive physics

### P0.1 Adjudicate the open PRs

| PR | Proposed disposition (review, not automatic merge) |
| --- | --- |
| [196](https://github.com/uwplasma/GKX/pull/196), [200](https://github.com/uwplasma/GKX/pull/200), [201](https://github.com/uwplasma/GKX/pull/201) | Treat as one CPU-f32 repair stack. Test actual bracket values/VJPs in isolated processes; prefer the repair over permanent broad skips; retain GPU layout checks. |
| [197](https://github.com/uwplasma/GKX/pull/197), [199](https://github.com/uwplasma/GKX/pull/199) | Review legacy per-step absorber and route consistency together. Reproduce original regression, both step maps and observables; document which eigensolver can represent that policy. |
| [202](https://github.com/uwplasma/GKX/pull/202) | Do not merge as a bundle. Separate independently justified streaming/coefficient repairs from fixed-rate damping and new table experiments; preserve evidence. |
| [198](https://github.com/uwplasma/GKX/pull/198), [203](https://github.com/uwplasma/GKX/pull/203) | Preserve plans/logs as history. This review reconciles their competing recommendations; request explicit supersession, not deletion or automatic closure. |

The published GX paper and old preprint differ on absorber amplitude.
Run the small comparison specified in the research program; do not choose by
PR chronology. Guard external reference kernel launch coverage and record local
patches/binary hashes. A repaired external binary is a distinct reference.

**Exit:** f32 crash probe, route-identity, original regression and boundary-policy
checks pass on the selected CPU/GPU environments; every open PR has a reviewed
disposition. No solver change in this planning PR.

### P0.2 Evidence and API contracts

- Recompute claims from their inputs. Mark unavailable reference decks
  (especially HSX) and historical artifacts separately from fresh current-head runs.
- Make precision explicit in physics tests; test f32/f64 in separate processes.
  Do not weaken tolerances to hide a physics error.
- Qualify the no-argument demo separately from a converged eigenmode: the fresh
  run completes but emits CFL and under-resolved-growth warnings. Make the
  teaching preset safe and its transient diagnostic labels unambiguous.
- Check that prepare/load/solve preserve resolved `Nl,Nm`, species, geometry,
  fields, collision model, accepted-time history and numerical policy.
- Measure real compilation: `PreparedSimulation.warmup()` is currently a no-op,
  so “prepared” alone must not certify that compilation has finished.
- Keep diagnostics and optimized maps identical. A source branch or a passing
  metadata test is not evidence that the full physics route is supported.

## 5. P1 — the smallest convincing physics/derivative benchmark

Use **CBC + one documented stellarator case**, then the initial/candidate QA pair.
Use exact source decks, geometry hashes, normalizations and reference revisions.
First choice: the existing W7-X ITG comparison, after checking its actual
equilibrium provenance. Missing inputs block that claim, not the CBC controls.
Do not substitute bundled QHS/QI examples for actual HSX/W7-X benchmarks.

1. **Operator checks:** projected manufactured solution, Hermite/Laguerre
   identities, streaming/mirror weighted adjointness, nonlinear bracket
   free-energy balance, zonal field response and field-solve residual.
   Manufacture forcing independently of the production RHS.
   Spectral convergence is not a fixed finite-difference order.
2. **Linear checks:** growth/frequency and eigenfunction, mode overlap/branch
   separation, domain and moment convergence; directional AD versus finite
   differences, including geometry and profile parameters.
3. **Nonlinear checks:** stationary physical heat flux, zonal/nonzonal energy,
   free-energy budget and resolved spectra. Establish matched comparison-code
   transport intervals, not trajectory identity after chaos separates phases.
4. **Refinements:** time sampling and `dt`, perpendicular box versus cutoff,
   parallel domain versus `Nz`, `Nl/Nm` versus closure. One factor at a time,
   followed by one joint refinement to detect compensation.
5. **Gradient checks:** same detached state, numerical map and physical horizon
   for AD/FD; perturbation ladder; checkpoint/plain agreement. Then test whether
   directions improve an independent statistical objective.

**Target acceptance (proposed, predeclare before running):** linear observables
within 1% of a converged matched reference away from zero; transport discretization
change below 5% and below the intended design effect; matched-code transport
difference compatible with uncertainty and a declared 10% comparison envelope.
Near zero use absolute tolerances. A failure changes the claim, not the threshold.
Directional finite differences require a stable perturbation plateau, not one
fortunate step size.

First complete one clearly turbulent point. Add a small three-gradient bracket
of the CBC nonlinear threshold as a falsifiable saturation/zonal test.
A full Dimits-shift publication scan is subsequent work, not a reason to postpone
all usable ES results.

**Exit:** replayable benchmark bundle with pass/fail gates, uncertainty and cost;
no nominal reduction promoted while drift, tails or refinement gates fail.

## 6. P2 — useful derivatives and warm QA optimization

### P2.1 Warm restart is a continuation policy, not a new adjoint

Use the existing Python integrator with a device-resident turbulent distribution.
Recompute fields with the new geometry/species; do not carry an old potential as
an independent solution. Preserve physical amplitude.
Freeze seeds/states throughout each local objective/line search and refresh only
after acceptance. Record numerical topology and normalization in the state contract.

Measure cold, warm, and warm-plus-re-equilibration from the same accepted design.
Check at least small and larger geometry perturbations, three independent seeds,
and two re-equilibration budgets; compare on **equal total wall-time budgets**.
Do not infer convergence from the existing 5% geometry / quarter-spin-up heuristics.
Initial implementation should keep host orchestration outside AD and the
distribution on-device; differentiated windows stay pure JAX.
An array in Python is not automatically a differentiable in-memory pipeline.

**Adopt warm reuse only if** independent flux distributions agree within the
predeclared bias budget, descent directions remain useful, and end-to-end time
to the same uncertainty decreases. Otherwise retain cold restarts.

### P2.2 Three examples, one geometry/objective interface

| Example | Objective | Required result |
| --- | --- | --- |
| Linear QA ITG | Smooth aggregate of unstable eigenvalues over selected `ky,s,alpha` | Feasible geometry and verified derivative/branch handling |
| QA quasilinear screening | Declared spectral weights/saturation rule | Held-out ranking against nonlinear cases, including failures |
| QA turbulent transport | Fixed-window physical heat flux plus VMEX constraint tuples | Cold held-out seeds, timestep/resolution/field-line checks show resolved reduction |

Share setup and plotting through existing helpers; retain short readable scripts.
Use VMEX's tuple convention. Fix physical `s` rather than an index that changes
with equilibrium resolution; hold scale, profiles and constraints explicit.
Reserve validation seeds and neighboring radii/field lines/gradients before
optimization. Check closeness to low-order rational iota and domain sensitivity.
QL or reduced resolution must not exclude a good design solely by an unvalidated proxy.

Compare checkpointed AD against finite-difference directions and a bounded SPSA
control; choose on verified improvement per wall time, not derivative exactness
for a surrogate window. No shadowing or statistical-implicit-adjoint project now.

### P2.3 Performance on the admitted physics

- Profile cold compile, cached RHS, FFT/bracket, field solve, diagnostics,
  transfers, spin-up, window value/VJP and validation separately.
- Synchronize JAX timings, use repeated medians/spread, report peak resident and
  temporary memory, compiler/device/thread settings and numerical identity.
- Use independent cases first. Assess the existing species–Hermite route next,
  with width-two halos and field reductions; do not revive whole-state FFT
  sharding before a communication profile justifies it.
- Precondition the already implemented stiff/shifted solves only when profiling
  identifies them as limiting. Require unpreconditioned residuals and transpose
  tests; count setup and memory. A better preconditioner cannot speed an explicit
  ExB-limited step by itself.
- Proposed adoption threshold: at least 20% total workload saving outside timing
  noise, no failed physics/gradient gate, and no unreported memory regression.

## 7. P3 — ship Coulomb honestly

Do not make full multispecies Landau a prerequisite for the first ES optimization.
Do not describe the finite-wavelength tables on main as already validated.

1. Reproduce unmerged coefficient repairs independently; declare moment
   ordering/shape, frequency convention, FLR argument and supported range.
2. Check particle-coordinate invariants, like-species entropy metric and `kperp=0`
   limits; finite-k gyrocenter moments must retain classical diffusion.
3. Test interpolated values and wavelength/parameter JVPs at zero, between nodes
   and at range boundaries, against independent high-accuracy coefficients.
   Reject unsupported bases/ranges, including equal-size but different `Nl,Nm`.
4. Validate the coupled field/source/operator map in f32/f64, then collisional
   relaxation, ITG and zonal-flow cases against a specifically named GENE or
   stella collision model.
5. Publish the narrow validated envelope or keep it experimental. Arbitrary
   moment order, unlike species and new interpolation forms require a separately
   justified extension, not an oversized correction PR.

Details and GENE distinctions: [Coulomb contract](docs/research_grade_program.rst).

## 8. What is deliberately deferred, and when to reconsider

| Deferred work | Admission trigger |
| --- | --- |
| Universal long-time AD / shadowing | Warm short-window directions demonstrably fail the design task despite horizon/ensemble tests |
| New closures / compressed Coulomb representations | A converged baseline identifies closure error or coefficient storage as the bottleneck |
| General energy–pitch velocity coordinates | Trapped-particle convergence remains the dominant cost after moment/collision/domain controls |
| Non-twisting flux tube | High shear / long domain forces excessive perpendicular resolution |
| Full-flux-surface / FCI island physics | Locality or flux-surface assumptions fail the target application; needs a separately funded model/validation program |
| Broad EM, TEM, unlike-species and Er optimization | ES workflow is qualified; add one specific physical regime with KAW/KBM, collision or shearing tests first |
| Absolute QL model / transport-profile coupling | Held-out nonlinear calibration meets a predefined accuracy gate |
| New release / performance publication claims | Open PR disposition and essential admitted physics, examples and reproducibility gates pass |

Retain existing code and historical tests unless a deliberate, separately tested
deletion is approved. Deferral is not an unsupported feature silently disappearing.

## 9. P4 — slimming, documentation, publication and release

- Keep one active plan, one append-only log, and one evidence entry per decision.
  Older plans remain accessible through pinned Git history; no mass JSON deletion
  or force-history rewrite in this task.
- Contract source by numerical ownership: remove duplicated stage/RHS/field
  dispatch and wrappers only with call-graph, API and benchmark evidence.
  Avoid merging unrelated tests into giant files merely to hit a file count.
- Classify JSON as runtime interchange, reproducible evidence, redundant summary,
  or bulky generated data. Deduplicate summaries; retain schema/provenance;
  externalize large immutable outputs with hashes, small fixtures and replay tools.
  Measure wheel, checkout and full clone separately.
- Documentation: quickstart → scenario recipes → equations/contracts →
  validation/results → developer reference. README shows a few results, runnable
  entry points and clear scope; do not copy input reference tables or logbooks.
- Examples: smoke, teaching and research presets, expected physics, cost/memory,
  plot definitions and failure interpretation; explicit commands, no hidden
  benchmark data downloads. Initial/final QA surface/Boozer/flux plots stay visible.
- First paper: verified differentiable local GK and cost-to-accuracy on CPU/GPU,
  with warm/cold optimization only if independently validated.
  A later physics/collision paper needs genuinely new results, not a new feature
  count. Do not claim the first differentiable GK code: iGENE and gyaradax exist.
- Release only after user review of open PR dispositions, current-head/wheel
  tests, admitted benchmark gates, documented limitations and manuscript data.
  No version bump, tag, release upload or automatic merge in this plan.

## 10. Resume protocol and logbook

1. Read this plan, [the audit](plan/baseline/review_2026_09_06.md), and the latest
   [log entry](plan/log.md); then inspect actual branch/worktree/PR state.
2. Pick the first unmet P0 item. State the falsifiable question, smallest test,
   cost cap and stop condition **before** starting a long run.
3. Log commit, environment, exact command, inputs/reference hashes, result,
   failed/skipped checks, elapsed time, artifact location and next decision.
4. For remote jobs also save host, directory, PID/job ID and last verified state.
   Unknown is not finished; never duplicate an unverified running campaign.
5. Commit small, deliberate changes as Rogerio Jorge; no AI coauthors.
   Push for review, never merge without explicit approval.

Historical authorities (superseded ordering, preserved evidence):
[main plan at a99dac89](https://github.com/uwplasma/GKX/blob/a99dac898334414d31733f6d286bd4c36983702e/plan.md),
[September 4–5 log at 71c5cc48](https://github.com/uwplasma/GKX/blob/71c5cc480411e3c568b38ce9010b4b9093fa2453/plan/log.md),
[external review at 1643f0e7](https://github.com/uwplasma/GKX/blob/1643f0e72a6310068b3387cceb94e39922020597/plan/baseline/review_2026_09_05_external.md).
