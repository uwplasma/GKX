# GKX: focused research plan

Updated 2026-09-06, final planning pass. **No solver changes, merge, or release.**
Baseline: main [a99dac89](https://github.com/uwplasma/GKX/commit/a99dac898334414d31733f6d286bd4c36983702e).
This is the proposed execution authority once reviewed, not a claim that its gates pass.

## 1. Outcome and scope

Deliver a reproducible local gyrokinetic workflow that can distinguish a useful
stellarator design change from numerical error and sampling noise, with
trustworthy derivatives and measured CPU/GPU cost.

**First optimization control:** electrostatic ion-temperature-gradient (ITG) transport
in nested-surface equilibria, kinetic ions and adiabatic electrons, one validated
collision/closure policy, and vacuum QA optimization with finite `a/L_T,a/L_n`.
Validate a tokamak control and one stellarator before expanding geometry coverage.
Vacuum describes the equilibrium, not zero turbulence drive.

**Required research-grade scope:** electrostatic (ES) **and electromagnetic (EM)**
local gyrokinetics, including kinetic electrons, `phi,A_parallel,delta B_parallel`,
and qualified linear/nonlinear cases in tokamak and stellarator geometry.
EM correctness is a core deliverable, not an optional extension after optimizing
QA. An ES design result alone does not complete this program or justify the
research-grade release.

Keep the broader ambition: additional kinetic species, advanced collisions,
equilibrium flow and ESSOS coil fields. Arbitrary combinations require separate
admission tests. A verified code is not automatically an experimentally
validated, predictive model.

The plan must carry enough context, exact next actions and failed evidence for
independent resumption. Its size is not an optimization target. Keep the README
short; keep this execution plan complete, with equations in the technical page
and chronological evidence in the log. The earlier 294-line revision compressed
the context too aggressively and incorrectly deferred core EM qualification.

| Priority | Outcome | Dependency |
| --- | --- | --- |
| P0 | One defensible baseline, resolved correctness regressions and claims | None |
| P1 | Required ES **and full-EM** physics, transport and derivative qualification | Shared P0 contracts; staged ES and EM tests below |
| P2 | Faster validated QA optimization, using warm states where beneficial | Relevant P1 subset; ES experiments may proceed while core EM is qualified |
| P3 | Bounded Coulomb implementation with an honest shipping envelope | P0; does not block Dougherty-based P1/P2 |
| P4 | Usable research package, evidence bundle, manuscript and release review | Required ES/EM P1 gates, P2 evidence, and explicit P3 scope decision |

Work on **one correctness change and one bounded measurement at most**.
No unlimited resolution or algorithm sweeps. After each decision experiment,
record adopt / reject / unresolved and its next discriminating check.

### 1.1 Context that must survive the refactor

- GKX's differentiable Hermite–Laguerre formulation is a means to a research
  workflow: equilibrium/geometry → linear or nonlinear dynamics → physical
  observables → useful design directions → independent verification.
- VMEX supplies vacuum/finite-pressure equilibria and geometry derivatives.
  ESSOS supplies coil fields and their derivatives. A smooth Biot–Savart
  derivative does not prove nested surfaces or a valid local GK equilibrium.
- SOLVAX supplies reusable structured linear algebra. Preserve moment
  recurrences and matrix-free execution; move a primitive upstream only after
  demonstrating its need and its transpose contract in GKX.
- Target both local CPUs and office GPUs, with explicit precision and memory
  policies. Independent cases and distributed states are distinct forms of
  parallelism and need separate timing/accuracy evidence.
- Retain linear stability, quasilinear screening and nonlinear transport as
  complementary applications. Their derivative contracts and validation
  standards differ; none automatically certifies the others.
- Preserve the work already done: implicit eigenmode sensitivities,
  checkpointed windows, geometry bridges, restart/output contracts, comparisons,
  performance measurements and initial/candidate QA visualizations.
  Revalidate changed contracts; do not restart the entire project from scratch.
- A finite-time AD identity, a CI success, a short finite trace and a visually
  quiet `Wphi` window answer different questions. Record which question a test
  actually answers.
- The nominal QA reduction is useful evidence for designing the next test,
  not a resolved claim. Missing source decks and unknown remote-job states are
  explicit evidence gaps, not reasons to manufacture replacement references.
- Changes stay small, concise and deliberate. No source/JSON deletion that
  loses unique physics or provenance merely to reduce counts; no new release
  before reviewed PR dispositions and essential goals are achieved.

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
- Make the three-field EM contract a required P1 outcome, with independent
  references, channel-resolved transport, finite-beta geometry and derivative
  tests. ES optimization is an early application, not the scope ceiling.
- Restore implementation order, numerical ownership, campaign budgets and
  completion criteria. Do not replace a complete roadmap with an executive summary.

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

### 3.1 Numerical ownership and concrete starting points

| Contract | Existing starting point | What the next test must establish |
| --- | --- | --- |
| Resolved public input | `api/prepared.py`, runtime TOML/resolution helpers | CLI/Python agree on requested moments, fields, species and numerical policy |
| Fields | `terms/fields.py` | Independent quasineutrality and both Ampère residuals, not wrapper self-agreement |
| Linear map | `operators/linear/`, matrix-free solver modules | Same physical operator in initial-value, eigenmode and gradient paths |
| Nonlinear map | `terms/nonlinear.py`, `solvers_nonlinear_state_integration.py` | Complete ES/EM stage coupling, projection and field refresh |
| Physical transport | `operators/moments.py`, `operators/fluxes.py` | Species/channel normalization and independently reconstructed flux |
| Stopping | `diagnostics/saturation.py`, runtime chunks | Qualified burn-in, irregular-time handling, effective sample size and false-stop rate |
| Warm state | `workflows/runtime/warm_start.py`, QA example | Device residence, compatible physical state, no trial-history-dependent objective |
| Parallel execution | `parallel/integrators.py` and nonlinear distribution helpers | Correct moments/halos, species reductions, field response and adjoint |
| EM evidence | KAW/KBM decks, field tests and finite-beta geometry tests | Separate reduced two-field evidence from complete three-field qualification |

Paths in this table are under `src/gkx/` unless identified as an example/test.
These are ownership boundaries, not instructions to create new modules.

The current KBM runtime deck uses kinetic ions/electrons with `use_apar=true`
and `use_bpar=false`. Its contract test preserves a transitional operator; it
does not test the omitted compressional field. Existing tests do exercise
nonzero `apar/bpar`, custom field VJPs, the zonal beta limit and channel sums.
They are valuable starting checks, not a saturated full-EM benchmark.

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

### P0.3 Implementation order and evidence discipline

1. Record current main, each candidate SHA and the exact worktree; reproduce
   the smallest f32/compiler failure before testing a repair. Keep bad/good
   subprocess outputs and a f64 control. Do not silently skip the failing shape.
2. Compare legacy/per-step and fixed-rate damping at a selected affected mode,
   holding everything else fixed. Resolve route identity before scanning beta.
   Preserve the published two-field reference when testing a three-field extension.
3. Write independent field residual checks before changing EM coefficients.
   Distinguish an incorrect normalization, missing term and singular gauge mode.
4. Fix load/prepare/solve and seed-state contracts in small, separately tested
   PRs. Verify a real execution, not only the prepared summary dictionary.
5. Freeze the accepted baseline and reference builds for P1. Rerun affected
   controls after each numerical change; retain original failed traces.

Each implementation PR must name the equation/contract changed, minimal
reproducer, affected configurations, precision/backend tests and cost impact.
Attach a passing operator test and a discriminating physics/numerics check;
regression agreement with an old implementation is insufficient by itself.

Start with local algebra/small grids. A comparison with missing geometry,
nonfinite output or unidentified collision model fails the setup gate before
any long integration. No simultaneous large scan of all open hypotheses.

## 5. P1 — required ES and electromagnetic qualification

P1 has two required outcomes sharing one solver, geometry, statistics and
evidence system. Start EM field/linear qualification as soon as the relevant P0
contracts pass; do not wait for a full QA optimization campaign.

### P1.1 ES control and reusable benchmark protocol

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

### P1.2 Electromagnetic model contract — required

The core model is local, low-frequency, delta-f GK with kinetic Maxwellian ions
and electrons. Qualify finite mass ratio, charge/density balance and a named
collision model. Additional species reuse the same field sums but need their
own charge/current and collisional-exchange tests.

Publish a configuration table with three distinct modes:

| Mode | Evolved kinetic response / fields | Permitted claim |
| --- | --- | --- |
| ES | Kinetic ions, adiabatic or kinetic electrons; `phi` | ES validation only |
| Reduced EM | Kinetic ions/electrons; `phi,A_parallel`, explicitly `delta B_parallel=0` | Named approximation and matched two-field reference |
| Full EM | Kinetic ions/electrons; `phi,A_parallel,delta B_parallel` | Full local EM only after all required gates below |

An adiabatic-electron magnetic-field calculation is not a substitute for a
kinetic-electron EM benchmark. Unsupported flag/species/basis combinations
must refuse or be explicitly labeled reduced models; flags must never silently
discard requested fields. Retain intentional approximations as named modes.

**Derive and freeze before optimization:**

- The stored `G` distribution, its relation to `h` and `delta f`, and the
  field-dependent transformation. Trace inductive response through that
  transformation instead of adding a duplicate `partial_t A_parallel` term.
- Quasineutrality, parallel Ampère and perpendicular pressure/Ampère balance,
  including FLR weights, polarization, skin response, signs and beta factors.
- The generalized gyroaveraged potential in streaming, drifts, drive,
  collisions and nonlinear brackets. Toggling a field changes both field
  solve and RHS consistently.
- Parallel electric response, magnetic flutter, compressional magnetic
  coupling, species-resolved particle/heat flux and particle–field energy
  exchange. Declare the retained gyrokinetic ordering.
- The zero-mode/gauge policy, small-`k_perp` and small-beta limits, linked
  boundaries, Jacobian weights and conservation properties.
- The source/operator combination for every admitted collision model,
  including physical collisional dissipation and any numerical absorber.

Separate **equilibrium beta** from **fluctuation beta**. A frozen-geometry beta
scan reproduces a local benchmark, but is not a self-consistent VMEX
finite-pressure equilibrium sequence. For the latter, track pressure-gradient
geometry, curvature versus grad-B, species profiles and normalization together.

### P1.3 EM execution ladder and decision gates

All stages below are required for the claimed local ES/EM research-grade
milestone. Small test cases precede expensive runs; passing an earlier stage
does not close the later ones.

**EM0 — field algebra and complete term routing**

1. Assemble independent small field systems and their source moments. Check
   each equation's residual, the coupled `phi/B_parallel` block and `A_parallel`.
   Do not only compare a wrapper with the implementation it calls.
2. Excite density, parallel-current and perpendicular-pressure responses
   separately. Include zonal and nonzonal modes, two kinetic species, unequal
   temperature, varying geometry and finite FLR.
3. Toggle `A_parallel` and `B_parallel` independently; verify expected zero
   channels and nonzero responses. Check retained first-order terms against
   direct velocity quadrature on a small independent grid.
4. Check the field VJP/JVP, full RHS linearization and an independent weighted
   free-energy identity. The EM invariant includes particle and magnetic
   contributions in the chosen distribution convention, not just `Wg+Wphi`.
5. Test f32/f64 and a normalized residual over a parameter ladder. Measure
   cancellation/conditioning in long-wavelength, light-electron limits.
   Denominator clipping is not a substitute for a valid constrained solve.

**Exit:** no missing field/RHS/diagnostic connection; residuals scale with
conditioning and precision; invariants converge under refinement.

**EM1 — analytic waves and limiting behavior**

1. Use a homogeneous, periodic, no-drive slab to test shear-Alfvén frequency
   in its MHD asymptotic range. Extend to a kinetic-Alfvén dispersion/damping
   reference at resolved `k_perp rho` and mass ratio.
2. Test finite-beta and low-beta limits, finite/increasing velocity resolution,
   timestep convergence, relative field amplitudes/phases and parallel current.
   State the applicability of the dispersion relation; do not use an MHD
   approximation as an exact finite-FLR reference.
3. Test compressional pressure response even where `B_parallel` is small.
   The low-frequency GK model does not aim to recover the fast magnetosonic branch.
4. Treat the `beta=0` switch and `beta→0+` limit separately. A nonuniform
   Alfvén timestep restriction or a branch in the field solve must not be
   mistaken for a failed physical ES limit or a smooth AD point.

**Exit:** converged frequencies, damping and field ratios agree with their
declared limits; no nonphysical unstable branch or hidden magnetic suppression.

**EM2 — matched linear ITG–KBM and full-field comparison**

1. Reproduce the published GX/GS2 CBC beta scan with `B_parallel=0` at
   `ky rho_i=0.3`. First test low beta, the ITG–KBM transition neighborhood
   and a clearly KBM-dominated point; refine around the transition only after
   mode identification is reliable.
2. Diagnose the current KBM discrepancy using eigenfunctions, residuals,
   parity and growth/frequency, not fitted scale factors. Resolve thermal-speed,
   beta, geometry, collision, boundary and timestep conventions first.
3. Add a full-field comparison against stella and GS2 (GENE where available).
   Use the same resolved physical deck; stella's current `EM_KBM.in` has
   different geometry, gradients, beta and `ky` from GKX's deck. Its filename
   is not a declaration of matched inputs.
4. Perform paired `B_parallel` on/off checks at finite beta, followed by
   velocity/domain/timestep and one combined refinement. Explain a small
   compressional effect physically instead of accepting a disconnected field.
5. Require a second independent discretization for selected anchor points;
   GX alone is insufficient to rule out shared moment/convention errors.

**Exit:** proposed 1% growth/frequency agreement away from marginality on
matched converged modes; absolute tolerances near zero; transition brackets
agree within resolved beta uncertainty. Field shapes and residuals must also pass.

**EM3 — nonlinear tokamak transport and energy**

1. Use a kinetic-electron low-beta CBC control, a finite-beta ITG point and
   a KBM-side point that remains within local delta-f ordering.
   A case that leaves that ordering is reported as model failure, not saturation.
2. Benchmark total and per-species `Q_ES,Q_Apar,Q_Bpar` and particle flux.
   Sum independently reconstructed channels; compare magnetic spectra and
   field energy. Total flux agreement must not hide cancelling channel errors.
3. Match field subset, geometry, physical collision model and numerical
   dissipation across codes. If collision models differ, use a declared
   collisionless limit or qualify the difference; do not call it exact parity.
4. Verify short-time same-state RHS/flux comparisons, then statistically
   compare stationary runs. Apply the P1.1 uncertainty and refinement protocol
   to dominant individual channels as well as totals.
5. Check drives, collisional loss, numerical damping and boundary terms in the
   discrete energy budget. Resolve current/electron and magnetic tails even
   when the ion heat-flux spectrum looks satisfactory.

**Exit:** matched mean-flux uncertainty and discretization gates, resolved
magnetic response and energy accounting; no claim based on a two-step sum test.

**EM4 — finite-beta stellarator qualification**

1. Freeze one actual nested-surface W7-X equilibrium/flux tube and profile set.
   Check the geometry against an independent interface before running EM.
2. Run a matched frozen-geometry beta scan with kinetic electrons and both
   magnetic fields; then one physically consistent finite-pressure equilibrium
   case. Record which beta and which geometry derivatives changed.
3. Compare selected linear modes and a stationary nonlinear point with stella
   or GENE using the same equilibrium. Use GX as an additional moment-method
   control where its corresponding path is qualified.
4. Repeat a neighboring field line and parallel-domain refinement; magnetic
   shear, pressure-gradient drifts and boundary connection must be preserved.
5. Do not use a global EUTERPE/GENE-3D result as direct local-code parity without
   a matched local limit. Global results motivate scope checks, not arbitrary
   adjustments to a local benchmark.

**Exit:** one reproducible full-EM stellarator case with qualified geometry,
linear response and stationary transport. A tokamak-only EM pass is insufficient.

**EM5 — derivatives, restart and execution identity**

1. Extend existing small finite-window tests to full kinetic-electron EM
   parameter, geometry and collision derivatives. Check each flux channel's
   derivative, not only a possibly compensating total.
2. Compare JVP/VJP directional products, finite-difference ladders and
   checkpoint/plain results. Branch selection, stopped initial states and
   numerical topology remain fixed; test beta derivatives at positive beta.
3. Verify an in-memory or NetCDF restart regenerates all fields and gives the
   same short continuation. Warm shape changes re-equilibrate current and
   magnetic energy as well as ion heat flux.
4. Check CPU/GPU values and gradients, then species–Hermite moment reductions
   and transpose communication. Unsupported collision/mesh combinations refuse.
5. Profile kinetic-electron streaming, field response and communication before
   selecting IMEX/preconditioning changes. Test the physical residual after
   preconditioning, and include setup/rebuild and VJP memory in timing.

**Exit:** a documented full-EM Python value/gradient/restart example and
verified supported CPU/GPU execution. Distributed EM claims require their own
identity, adjoint and scaling gate, not an ES timing extrapolation.

### P1.4 References, matching contract and minimum evidence bundle

| Reference | Use in this plan | Important qualification |
| --- | --- | --- |
| [GX, `6.1 and appendix G](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/gx-a-gpunative-gyrokinetic-turbulence-code-for-tokamak-and-stellarator-design/2C4BB81955E7E749B95B8B8141E997FA) | EM2 reduced-field beta scan and EM3 low-beta kinetic control | Published ITG–KBM scan explicitly omits `B_parallel` |
| [stella EM tests, pinned source](https://github.com/stellaGK/stella/tree/2b8e269f2addd0baa5991057eafa022135e04498/AUTOMATIC_TESTS/numerical_tests/test_7_electromagnetic) | Term isolation, full-field KBM and restart methodology | Frozen outputs are regression evidence; translate and converge the actual input |
| [GS2 documentation](https://gyrokinetics.gitlab.io/gs2/) | Independent energy/pitch discretization and EM field response | Pin revision, namelists, collision model and field switches |
| [Pueschel, Kammerer & Jenko (2008)](https://doi.org/10.1063/1.3005380) | Finite-beta CBC transport/KBM benchmark design | Recover exact field and geometry assumptions before numerical targets |
| [Davies thesis (2022)](https://etheses.whiterose.ac.uk/id/eprint/32068/) | EM stella/GS2 algorithm and benchmark context | Historical implementation; not a claim about today's solver or all regimes |
| [GENE magnetic compression (2025)](https://doi.org/10.1016/j.cpc.2024.109410) | Why finite-FLR `B_parallel` deserves an independent test | Global implementation is not a drop-in local GKX discretization |
| [Global stellarator EM (2023)](https://doi.org/10.1017/S0022377823000363) | Finite-beta stellarator physics and locality checks | Do not compare global/local fluxes as identical models |

Every reference record contains: code SHA, patches/binary hash, machine/compiler,
species `Z,m,n,T` and gradients, collision operator/frequency convention, all
field flags, beta definition, geometry hash and pressure gradients, coordinate
and boundary conventions, box/cutoffs, velocity truncation, timestep, closure
and absorber, seed, diagnostic units and averaging rule.

Use one compact evidence bundle per benchmark, shared by CI, docs and figures:
resolved TOML/namelist inputs, small tabular observables, provenance, explicit
pass/fail/unknown gates and reproduction command. Store large distributions and
movies outside Git with content hashes. Do not introduce another JSON framework.

### P1.5 Bounded execution schedule

| Batch | Work | Start/stop rule |
| --- | --- | --- |
| A | P0 reproduction + EM0 field/RHS tests | Local small grids first; stop on missing terms/nonfinite fields |
| B | ES CBC control + EM1 wave/limit checks | Freeze accepted normalization; no parameter sweep before convergence |
| C | EM2 three-point beta comparison, then transition refinement | Exact decks and healthy reference builds required |
| D | ES stellarator and EM3 nonlinear pilot | Use short pilots to set resolution/cost; pilots are not saturation evidence |
| E | Fixed-budget replicated transport/refinement and EM4 stellarator | Launch only after pilot setup/tails pass; check remote job state first |
| F | P2 warm QA plus EM5 differentiated/restart workloads | Use relevant qualified physics; keep independent validation seeds |

Predeclare a wall-time/memory cap and physical accuracy goal for each batch.
If a cap is reached, save state and report unresolved; do not shorten the
averaging interval retrospectively to make a gate pass. Resolved deterministic
field/linear checks need no ensemble; turbulent means do.

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

Implementation sequence:

1. Define a typed accepted-state record containing the device array, physical
   time, geometry/species/normalization identity, field mode and resolution.
   Extend the existing helper; do not create a competing cache abstraction.
2. Separate accepted-state ownership from temporary line-search states.
   Rejected proposals cannot change future objective values.
3. Rebuild geometry/field coefficients for a new shape and solve all enabled
   fields from the distribution. In EM, `A_parallel` participates in the
   distribution convention: derive any necessary state transformation before
   assuming the ES reuse map applies.
4. Test equal-parameter restart identity first, then changed-shape response,
   then bias/cost over independent seeds. Measure transfers and peak memory.
5. Keep gradient-window length and statistical validation horizon separate.
   Refresh the accepted ensemble only at documented acceptance points.
6. Promote the simplest passing policy; save cold restart as the fallback.

### P2.2 Three examples, one geometry/objective interface

| Example | Objective | Required result |
| --- | --- | --- |
| Linear QA ITG | Smooth aggregate of unstable eigenvalues over selected `ky,s,alpha` | Feasible geometry and verified derivative/branch handling |
| QA quasilinear screening | Declared spectral weights/saturation rule | Held-out ranking against nonlinear cases, including failures |
| QA turbulent transport | Fixed-window physical heat flux plus VMEX constraint tuples | Cold held-out seeds, timestep/resolution/field-line checks show resolved reduction |

Each example has a small smoke preset, a teaching case with interpretable plots,
and a documented research preset. Add a full-EM evaluation/gradient recipe to
the shared workflow after EM5; an expensive EM optimization campaign is not
needed to prove basic three-field differentiation.

The eventual finite-beta optimization objective must include the intended
species and EM flux channels, with equilibrium constraints and profile choices
explicit. An ES-optimized candidate is not automatically an EM improvement:
perform a held-out EM evaluation before making that broader claim.

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

Required profile workloads: ES adiabatic-ion-scale control; kinetic-electron
two-field control; full-EM finite-beta case; short/full checkpointed VJP; warm
refresh and independent ensemble. Use one accepted numerical baseline per
comparison. Report strong scaling separately from ensemble throughput.

For a streaming/Alfvén-limited workload, compare the existing explicit and IMEX
maps with verified order/stability and a field-aware preconditioner if needed.
Do not replace the nonlinear derivative with a statistical implicit solve.
Use equal-arclength or non-twisting coordinates only through the measured
decision tests in the technical page; changing coordinates also changes
quadrature, boundaries and geometry derivatives.

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

EM adds a necessary current-response test: an operator that preserves the right
density moments can still give incorrect parallel-current relaxation, magnetic
flutter transport or field-source cancellation. Test the admitted model with
both magnetic fields before advertising that model as EM-compatible.
Until that passes, keep a specifically qualified collision model for core EM;
do not silently substitute a different model or claim full multispecies Coulomb.

## 8. What is deliberately deferred, and when to reconsider

| Deferred work | Admission trigger |
| --- | --- |
| Universal long-time AD / shadowing | Warm short-window directions demonstrably fail the design task despite horizon/ensemble tests |
| New closures / compressed Coulomb representations | A converged baseline identifies closure error or coefficient storage as the bottleneck |
| General energy–pitch velocity coordinates | Trapped-particle convergence remains the dominant cost after moment/collision/domain controls |
| Non-twisting flux tube | High shear / long domain forces excessive perpendicular resolution |
| Full-flux-surface / FCI island physics | Locality or flux-surface assumptions fail the target application; needs a separately funded model/validation program |
| Broad MTM/TAE/energetic-particle campaigns, TEM and Er optimization | Core EM0–EM5 is qualified; add the ordering/species/collision/shearing checks for the new regime |
| Arbitrary unlike-species Coulomb | Pair coefficients and mass/temperature exchange tests exist; not supplied by more like-species table nodes |
| Absolute QL model / transport-profile coupling | Held-out nonlinear calibration meets a predefined accuracy gate |
| New release / performance publication claims | Open PR disposition and essential admitted physics, examples and reproducibility gates pass |

Retain existing code and historical tests unless a deliberate, separately tested
deletion is approved. Deferral is not an unsupported feature silently disappearing.
**Not deferred:** kinetic-electron EM, both magnetic fields, canonical nonlinear
EM transport, finite-beta stellarator qualification and full-EM derivatives.

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
- First code paper: verified ES and EM local GK, differentiated workloads and
  cost-to-accuracy on CPU/GPU,
  with warm/cold optimization only if independently validated.
  A later physics/collision paper needs genuinely new results, not a new feature
  count. Do not claim the first differentiable GK code: iGENE and gyaradax exist.
- Release only after user review of open PR dispositions, current-head/wheel
  tests, admitted benchmark gates, documented limitations and manuscript data.
  No version bump, tag, release upload or automatic merge in this plan.

### P4.1 Documentation and research deliverables

Keep the scenario index aligned with the model contract:

| Reader/task | Required example or page |
| --- | --- |
| First-time student | Safe linear run; growth/frequency/energy/flux glossary; expected transient versus eigenmode |
| Turbulence user | ES and full-EM input recipes, species/field flags, saturation and resolution checks |
| Physics researcher | Field/kinetic equations, ordering, normalization and collision/closure support matrix |
| Optimization user | Linear/QL/nonlinear tuples, exact differentiated quantity, warm-state and held-out validation |
| Performance user | CPU/GPU sizing, supported sharding, synchronized timings, memory and accuracy constraints |
| Paper reviewer | Source-deck provenance, independent references, failed gates and replayable plots |

Required EM figures, generated from accepted data rather than schematic claims:
Alfvén dispersion/damping; ITG–KBM beta scan with field-mode labels; normalized
three-field eigenfunctions; channel-resolved nonlinear flux with uncertainty;
finite-beta stellarator geometry/flux; convergence and energy-budget residuals;
full-EM value/VJP cost and qualified scaling. Retain QA boundary/Boozer/flux plots.
The README selects high-impact accepted figures; the documentation carries
the complete evidence and does not hide failures.

### P4.2 Completion checklist

- [ ] P0: reviewed PR dispositions and exact public input/state/field contracts.
- [ ] P1 ES: converged CBC and provenance-checked stellarator controls.
- [ ] EM0–EM2: independent field/limit/linear three-field qualification.
- [ ] EM3–EM4: stationary nonlinear tokamak and finite-beta stellarator evidence.
- [ ] EM5: full-EM derivatives, restarts and supported CPU/GPU identity.
- [ ] P2: independently resolved QA design result; warm reuse only if qualified.
- [ ] P3: publish the exact admitted collision envelope and remaining exclusions.
- [ ] Performance: ES and EM time-to-accuracy/value/VJP measurements; no blanket
      distributed speed claim from a single configuration.
- [ ] Package/docs: working examples, wheel checks, source/data provenance,
      uncertainty plots and a manageable, reproducible repository.
- [ ] User review before any merge/release/publication submission.

No checklist item is marked complete merely because this planning PR exists.
An ES-only result may support an interim scoped report; it does not remove EM
from the required code milestone without a new explicit user decision.

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
