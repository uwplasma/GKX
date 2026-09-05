# GKX research and publication roadmap

Updated **2026-09-04** against **2.0.0**, `a99dac898334414d31733f6d286bd4c36983702e`.
This is the execution plan, not a statement that the work below is complete.

## Start here

1. Read the [current audit](plan/baseline/review_2026_09_04.md), then resume **R0**.
2. Follow the dependencies below. Take one small, reviewable change at a time.
3. Update the relevant status here and append a short result to [the log](plan/log.md).
4. Keep PRs open for maintainer review; **do not merge them**. History rewriting is a separate reviewed operation.

This revision replaces the conflicting phase queues in the previous 2,881-line
plan. Its measurements and rejected experiments remain in Git history:
`git show a99dac89:plan.md`. Older baselines, the [PR ledger](plan/pr_ledger.md),
[historical audits](plan/pr_audit.md), and [references](plan/references.md) remain
evidence, not alternative instructions. Do not repeat completed migrations
because an old baseline describes them as unfinished.

## 1. Destination and current truth

Deliver a **research-grade, predictive, differentiable gyrokinetic code**, with
CPU/GPU parallel execution: electrostatic/electromagnetic; linear, quasilinear
and nonlinear; adiabatic/kinetic multispecies; several physical collision and
closure models; with/without equilibrium radial electric fields. Cover vacuum
and finite-beta fusion configurations, VMEX equilibria and ESSOS coil fields,
including an explicitly developed model for islands. Match GX's relevant
physics/numerics portfolio and extend it where independently verified science
and measured performance justify the extension. Compact implementation is a
means to maintain this capability, **not a restriction to a small local solver**.

Demonstrate stellarator optimization in three stages: **linear microstability →
quasilinear screening → independently validated nonlinear transport**. Retain
the efficient local model while developing broader geometry/background models;
do not advertise planned extensions as current release capabilities.

| Area | Current evidence | Remaining acceptance |
|---|---|---|
| Product | `load/solve/scan/plot/prepare`; real Case/results/prepared object; six CLI commands; native integration; first-run wheel gate | Simplify internal ownership; preserve working scripts/schemas |
| Physics | Local Maxwellian δf, Hermite–Laguerre/Fourier, ES/EM terms, adiabatic/kinetic species, analytic/imported/VMEX geometry | Per-model verification; kinetic-electron, EM, collision, zonal and nonlinear convergence gaps |
| Release correctness | Main CI green at audited SHA | Damping regression #192; proposed #197 is **open**, not landed; rate migration #194; f32 backend/tolerance coverage #196 |
| Autodiff | Eigenmode sensitivities; checkpointed finite-window physical heat-flux derivative | Useful gradient horizons, geometry/mesh derivatives, held-out nonlinear descent; no invariant-measure derivative claim |
| Optimization | Reduced examples, local derivatives, extensive positive and negative campaign records | No promotion-ready, resolution-resolved QA transport reduction; no completed GKX–ESSOS transport workflow |
| Parallelism | Independent scans/ensembles; diagnostic state, velocity and domain routes | Actual CPU/GPU distributed trajectories **and VJPs**, conservation, transport equivalence and performance |
| Size | 184 source Python files / 88,910 lines; 76 test files / 86,377 lines; 95 tools / 77,823 lines | Remove duplicate responsibilities, not assertions; audit 562 JSON files |
| Documentation | 33 RST pages / 18,857 lines; useful equations/figures | One task-oriented map; no contradictory claims or duplicate governance |

The audit distinguishes fresh measurements, source inspection, historical
artifacts and PR-author reports. None substitutes for a rerun of all physics.

### Current model versus target scope

- Current core: radially local, low-frequency δf gyrokinetics about Maxwellians; nested
  surfaces; appropriate periodic or twist-and-shift parallel boundary.
- Vacuum VMEC with prescribed finite GKX density/temperature drives is an
  **electrostatic test model**, not a self-consistent finite-pressure equilibrium.
- Closed periodic VMEX mirror tubes need the validated closure/metric adapter.
  Islands/non-flux geometry and nonlocal backgrounds are explicit R8 targets.
  Full-f/nonlinear collisions require consistent additional equations (C4/R8),
  not a switch in the local solver. Open-field/sheath and strong-rotation models
  need their own ordering and boundaries before those regimes can be claimed.
- Adiabatic-ion ETG, adiabatic-electron ITG, TEM, EM and multiscale turbulence
  require separate evidence. Equilibrium E×B shear remains a separately gated
  extension; sheared-coordinate infrastructure is not the full rotation model.
- CPU and NVIDIA GPU are required. Logical CPU devices test collective semantics,
  not multi-socket scaling. Multi-node, AMD and TPU performance remain unclaimed.

### Capability ledger and GX comparison

R1 maintains **one** equation→implementation→test→benchmark ledger, consumed by
the public verification matrix. For each *combination* record geometry, species,
fields, collisions, closure, Er, stepping, precision, primal/AD and device layout.
Use `planned / implemented / verified / benchmarked`; reserve **predictive** for
a declared domain with quantified uncertainty and independent validation.
Never infer the Cartesian product from individual feature checkmarks. Unsupported
combinations fail before tracing; no silent collision/field/precision substitution.

GX reference: office and upstream HEAD both
`3865a53778862e1686f414bf6f416339e24887c9` on 2026-09-04. Recheck before each
campaign. Audit `parameters.cu`, `linear.cu`, `exb.cu`, `fields.cu`,
`grad_parallel*`, `closures.cu`, tests and example decks against the
[GX formulation](https://arxiv.org/abs/2209.06731) and
[input/numerics reference](https://gx.readthedocs.io/en/latest/Reference.html).
Its source exposes kinetic species, phi/Apar/Bpar, ExB shear, moment closures
and model collisions; this is an inventory, not an independent GX validation.
Normalize equations, enabled terms, collision frequencies, dissipation, boundary,
box and flux units before either accuracy or equal-accuracy timing comparisons.

## 2. Execution order

```text
R0 release correctness + evidence baseline
 ├─ R1 model/numerical contracts ─ R2 benchmarks + statistical validation
 ├─ R3 profiling + CPU/GPU sharding ──────────────────────────────┐
 └─ R4 source/data slimming + R5 docs/examples                    │
R1 + R2 ─ R6 linear/QL optimization ─ R7 nonlinear optimization
R1 + R2 ─ R8b–d island/background model development (can start independently)
R6 + R7 ─ R8a ESSOS nested-field realization/robustness
R8a + verified R8b–d ─ R8e direct-coil/island transport optimization
R0–R7 evidence ─ R9 methods/design papers; R8 adds coil/geometry papers
```

R4/R5 can proceed during long R2/R3 experiments, but must not change equations
or reference values incidentally. Replication starts only after inputs and
acceptance criteria are frozen.

| ID | Status | Exit condition |
|---|---|---|
| R0 | **in progress** | Damping/precision repairs reviewed; release claims regenerated from repaired operator; clean-install f32/f64 probes |
| R1 | open | Combination ledger; independent manufactured forcing; ES/EM, multispecies, C0–C4 collisions, closures, E0–E2 Er, boundaries and transposes |
| R2 | open | Tokamak + QA/QH/QI portfolio; stopping tested against future data; transport and full resolution ladders |
| R3 | diagnostic baseline measured | Phase/memory profiles; working CPU/GPU distributed primal+VJP; useful layouts selected by measurement |
| R4 | partially landed | Fewer duplicate implementations/artifacts, measured size reduction, preserved public contracts and defect detection |
| R5 | partially landed | Organized docs and runnable student/research examples; equations, commands, results and scope agree |
| R6 | reduced examples only | VMEX ITG/QL design examples with held-out modes/surfaces/families and certified derivatives |
| R7 | finite-window AD implemented | Independently resolved nonlinear QA reduction, including convergence and constraints |
| R8 | planned | Nested realized-field robustness; verified non-flux geometry/island/background model; direct-coil transport/AD |
| R9 | planned | Reproducible manuscript bundles; every abstract/README claim linked to current evidence |

### R0: immediate small-PR queue

Resume checkpoint (2026-09-05): plan approved; first R0 slice is
[PR #199](https://github.com/uwplasma/GKX/pull/199), **open**, stacked on #197.
Worktree
`/Users/rogeriojorge/local/GKX-worktrees/r0-damping-path-consistency`, branch
`fix/r0-damping-path-consistency`, commit `b5dca15a`, base #197 `9074dd87`.
CPU/GPU boundary sentinel passed; 8 analytic RK/JVP cases plus the existing
Euler check passed; removing dt scaling fails the new test. Nonlinear RHS also
omits dt, not just the four field-supplied call sites: #194 must reconcile these
contracts without silently changing nonlinear results. No numerical change yet.
Second R0 slice: [PR #200](https://github.com/uwplasma/GKX/pull/200), **open**,
stacked on #196, commit e36e5bd8. Branch `test/r0-f32-backend-probe` at
`/Users/rogeriojorge/local/GKX-worktrees/r0-f32-backend-probe`, base #196 fdfb1a13.
The real f32 AD test passes on Mac JAX 0.10.2/0.11.1 but SIGSEGVs on Linux
x86-64/0.10.2 (f64 passes). Replaced the blanket future-version skip by isolated
execution, an exact observed-failure bound, and default-f32 CI coverage. This is
containment, not a solver repair. Probe environments
and reports: `/tmp/gkx-f32-20260905.alM6Fy`; office
`/home/rjorge/gkx-r0-f32-20260905.qIKcGz`. Third R0 slice:
[PR #201](https://github.com/uwplasma/GKX/pull/201), **open**, repair in
`/Users/rogeriojorge/local/GKX-worktrees/r0-f32-bracket-rank`, branch
`fix/r0-f32-bracket-rank`, commit `53d86f01`, based on #200 e36e5bd8.
Squeezing shared singleton batch axes cures both Linux 0.10.2 f32 AD crashes,
but unconditional squeezing regresses GPU VJP time/temporary memory; reject that
variant. CPU-only lowering passes actual Linux/Mac/GPU f32 gradient checks,
51 bracket cases and two-device CPU/GPU RHS/trajectory parity. Short nonzero
heat-flux window values/gradients match, with unchanged temporary memory;
no material speedup claim. Removed the crash exemption. Linux 0.11.1
passes the minimal rank-7 reproducer, not yet the application test.
Scratch `rank_profile.py` and `rank-*-f32.csv` in the probe directories preserve
the rejected experiment; full commands/results are in the logbook.
**Next:** review #199–#201 CI and finish damping route coverage. Broader f32,
sharded AD and production-horizon validation remain open.
No local/SSH tests remain active. Check remote CI on #199–#201 before relying
on full-suite status; both were still awaiting complete CI during this slice.
No merge is authorized; update this
checkpoint and append commands/results to `plan/log.md` before changing workstreams.
User checkout and other existing worktrees remain untouched.

1. **Damping:** review #197 against #192 independently. Compatibility restores
   the old per-step map; it is not #194's continuous-rate migration. Audit serial,
   adaptive, eigenoperator, implicit, Hermite-sharded and field-supplied RHS
   together. Freeze units in inputs/outputs; migrate affected decks explicitly.
2. **Detection:** add an unstable f64 sentinel at the benchmark timestep, not
   only a tiny f32 finite-value check. Verify the damp-only stage map analytically.
   A true rate ν must converge to `exp(-νt)` at fixed physical time; this is not
   the contract of an explicitly legacy per-step fraction.
3. **Precision:** review #196's upstream crash and skip bounds. Test default f32
   in subprocesses alongside f64. A skip preserves CI execution, not validation
   of that user path. Record supported versions; repair FD tolerances with scaled
   errors/step ladders without weakening physics tolerances.
4. **Rebaseline:** run installed-wheel startup, relevant physics sentinels and
   external scans. Keep failed/unsettled modes in denominators. Refresh release
   evidence only from repaired results; a readiness percentage is not validation.
5. **Public truth:** remove the assertion that QA passed all transport gates.

### Existing entry points to reuse

Inspect each script's help/config before running; source presence is not a
validated result. Keep raw outputs outside tracked documentation by default.

| Work | Current entry point |
|---|---|
| Bounded tests | `tools/release/run_test_gates.py` |
| Linear physics | `benchmarks/{cyclone,kinetic,tem,etg}_linear_benchmark.py`, `benchmarks/kbm_linear_comparison.py` |
| Threshold/convergence | `tools/campaigns/dimits_shift.py`, `convergence_protocol.py` |
| Saturation/uncertainty | `tools/campaigns/nonlinear_saturated_state.py`, `nonlinear_replicates.py`, `heat_flux_autocorrelation.py` |
| Gradient/QA evidence | `tools/campaigns/nonlinear_gradient_window.py`, `gradient_gates.py`, `qa_transport_validation.py` |
| Runtime/AD profiling | `tools/profiling/profile_runtime_kernels.py`, `profile_startup_and_cache.py`, `profile_nonlinear_adjoint_checkpointing.py` |
| Parallel profiling | `tools/profiling/profile_parallel_workloads.py`, `profile_nonlinear_sharding.py`, `profile_device_z_pencil_transport_window.py` |

Within a table cell, subsequent filenames use the first filename's directory.

## 3. R1: model and numerical contracts

For fixed discretization write one model statement:

\[
 f_s=F_{0s}+\delta f_s,\qquad
 g_s=\sum_{\ell,m}G_{s\ell m}\Psi_\ell(\mu B)H_m(v_\parallel),
\]
\[
 \dot G=L(\mathcal G,p)G+N(G,\mathcal F(G;\mathcal G))+C(G),\qquad
 G_{n+1}=\Phi_{\Delta t}(G_n;\mathcal G,p).
\]

Define the precise g/h convention, polynomial signs/weights, species ordering,
normalization, quadrature and projection. Derive quasineutrality, parallel Ampère,
magnetic compression and the discrete free-energy balance for the implemented
variables. A distribution norm alone is not automatically the total invariant.

| Contract | Owners under `src/gkx` | Falsification test |
|---|---|---|
| Basis/FLR | `core_velocity.py`, `operators/linear/cache_builder.py` | Orthogonality, recurrence, quadrature exactness, Bessel limits, high-order conditioning; independent coefficients |
| Spectral reality | `core_grid.py`, `operators/nonlinear/` | FFT scaling, Nyquist/zero modes, direct dealiased convolution, projector idempotence/VJP |
| Parallel geometry | `geometry/`, streaming/mirror terms | Weighted integration by parts, twist-and-shift links, field periods/orientation, shear and finite-beta drifts |
| Fields/flux | `terms/assembly.py`, `terms/fields.py`, `operators/fluxes.py` | Field residuals, zonal electron response, species sums, heat/energy/particle definitions, gauge and weighted energy balance |
| Stepping | `solvers_time_*`, `solvers_nonlinear_explicit.py` | Stage maps/order, CFL during growth, exact horizon/chunk/restart, damping units, fixed/adaptive contracts |
| Collisions | `operators/linear/collisions.py`, `collision_tables.py` | Published coefficients, pair conservation, entropy in derived metric, finite-k particle/gyrocentre identities and interpolation derivatives |
| AD/algebra | `objectives/`, `solvers_linear_krylov*`, SOLVAX | Residual/conditioning, non-normal eigenpairs, complex-real transpose identity, Taylor/FD including geometry/fields |

### Manufactured solutions and invariants

- Manufacture nontrivial multi-species/multi-mode states and fields. Derive forcing
  independently: using GKX's RHS as its own reference only tests plumbing.
- Isolate streaming, mirror, curvature, drive, collisions and nonlinear brackets;
  then assemble them. Use smooth periodic and valid linked-boundary solutions.
- Measure expected RK order before roundoff with spatial error controlled.
  Fourier/Hermite–Laguerre convergence is spectral for appropriate smooth data:
  measure coefficient/error decay, not a fictional fixed order in N.
- Check conservative limits and the full source–sink balance with dissipation.
  Derive boundary/FLR terms before requiring Euclidean matrix symmetry or zero
  gyrocentre-moment loss at finite k.
- Mutate signs, recurrence factors, normalizations, masks, links and damping units.
  Each relevant test must detect the injected defect.

### Collisions and closure

| Step | Scope | Exit |
|---|---|---|
| C0 | Exact species/moment/temperature/k support | Reject unsupported cases before compile; distinguish physical and artificial dissipation |
| C1 | Arbitrary-order drift-kinetic moments | Independent analytic/quadrature reference, Spitzer–Härm/relaxation ladders, cost scaling |
| C2 | Finite-k like-species | Field terms/metric, b→0, k-interpolation convergence/AD, moments beyond shipped 8/18 tables |
| C3 | Finite-k multispecies | Pair momentum/energy exchange, mass/temperature limits, projected entropy balance |
| C4 | Jorge–Frei–Ricci nonlinear Coulomb | Bilinear moments with controlled truncation/cost; linearization recovers C1–C3; consistent evolving background/full-f contract, conservation and entropy |

Tabulated Coulomb support is not arbitrary-order or arbitrary-species validation.
Unequal-temperature Maxwellians are not generally equilibria of full interspecies
Landau collisions. State the differing exact/approximate adjointness conditions
for [Sugama 2009](https://nifs-repository.repo.nii.ac.jp/record/388/files/5317%20PhysPlasmas_16_112503.pdf) and
[improved Sugama](https://arxiv.org/abs/1906.07427).

Measure recurrence versus moments, collisions and closure. The free-streaming
estimate `t_rec ∝ sqrt(Nm)/(|k_parallel| v_th)` depends on Hermite normalization;
it is not a universal turbulent validity horizon. Do not tune damping to hide a
bad zonal-flow result.

For each collision family document what is exact, modeled and truncated:
conserving Lenard–Bernstein/Dougherty, original/improved Sugama, linearized
Frei Coulomb, and nonlinear Jorge Coulomb. C4 is not implemented by evaluating
a linearized matrix on a nonlinear turbulent state. Derive
`C[F0+δf,F0+δf] = C[F0,F0] + C[F0,δf] + C[δf,F0] + C[δf,δf]` pairwise;
retain or order out terms consistently with the kinetic/background equations.
Test common-equilibrium nullspaces, pair exchange and entropy separately from
unequal-temperature relaxation. Inspect coefficient conditioning and assembly,
application, communication and VJP cost at research moment counts before
choosing tables, recurrences or matrix-free contractions.

Closures get a separate ladder: resolved/high-moment reference → truncation →
hypercollision → kinetic/gyrofluid closure. Compare phase mixing, echoes, zonal
response, linear modes and nonlinear flux at fixed physical collisions. Record
irreversible closure error and its effect on gradients, not only speedup.

### Electromagnetics, species and radial electric field

Promote phi, Apar and Bpar as a coupled energy-consistent system. Derive the
beta→0 limit, zonal/gauge nullspaces, magnetic flutter, magnetic compression,
species current and field-energy exchange. Separate equilibrium beta/pressure
gradient from fluctuating-field beta; test finite-beta VMEX geometry and real
electron mass ratio. Include impurities and unequal temperatures with charge
neutrality and matched collisions. Kinetic-electron stiffness must appear in
both accuracy and time-to-solution measurements.

| Step | Model | Required evidence |
|---|---|---|
| E0 | Prescribed local perpendicular ExB shear | Corrected remap `kx(t)=kx(0)-ky γE t`, stage/cache/field consistency, band-loss budget, zero-shear limit; analytic waves and matched GX shear scan |
| E1 | Stellarator equilibrium `Φ0` and `Er=-dΦ0/dr` | Derive normalized drift, radial/field-line coupling and local-ordering limits; distinguish Er from its shear; zonal response and full-surface/global comparison where needed |
| E2 | Transport-consistent Er | Prescribed profiles versus ambipolar root `Σs qs Γs(Er)=0` explicitly separated; include required neoclassical transport, root multiplicity/stability and implicit-derivative conditioning |

Current shearing-wave kernels/research integration are evidence for parts of E0,
not E1/E2 or toroidal rotation/PVG. Do not stop the tangent of a physically
moving event without measuring the resulting gradient contract: test across
remaps, finite-step perturbations and refinement, as well as between events.

## 4. R2: benchmarks and statistical validation

Every case needs a frozen deck, reference/version/digest, normalization transform,
observable, tolerance, runtime tier and regeneration command. Separate equation
**verification**, code comparison and experimental **validation**.

| Benchmark | Tests | Anchor/refinement |
|---|---|---|
| Free streaming / Landau | Phase mixing, signs, recurrence | Analytic dispersion; `Nz,Nm,dt`, closure and quiet-time ladder |
| Manufactured ES/EM | Operator coefficients/convergence | Independent forcing, isolated/full terms, spatial/velocity/time refinement |
| Tokamak zonal/GAM | Polarization, zonal response, drifts | Rosenbluth–Hinton/Merlo; small-k/aspect assumptions, residual/damping/frequency |
| W7-X zonal | Non-axisymmetric phase mixing | stella/GENE; `kx,Nl,Nm,Nz,nperiod`, closures; retain failed late envelopes |
| Cyclone s-alpha/Miller ITG | Threshold, eigenfunction, frequency | Dimits/CBC/GX; `ky,Nz,Nl,Nm,dt`, damping and settling |
| TEM / kinetic-electron ITG | Trapping/streaming/species response | stella/GENE/GX; matched collisions, real mass ratio and electron resolution |
| ETG | Electron scales/normalization | GX/GENE; distinguish adiabatic/kinetic ions; no multiscale claim |
| KAW / KBM | EM fields, cancellation, branch transition | Analytic KAW, beta scan; mass ratio, eigenfunctions and residuals |
| Microtearing / nonlinear EM | Parity, electron collisions, flutter heat flux | [GENE/GKV benchmark](https://www.jstage.jst.go.jp/article/pfr/11/0/11_2403011/_article), [STEP linear comparison](https://arxiv.org/abs/2307.01670); beta/collisionality/mass-ratio ladders, ES limit and total energy balance |
| Collision relaxation/conductivity | Invariants, entropy, collisional transport | Frei/Sugama/Spitzer–Härm; moment/k/species ladders |
| Equilibrium Er / ExB shear | Advection, remap, decorrelation and zonal response | Analytic shearing wave; corrected-remap/GX comparison; E1 stellarator/global reference; E2 ambipolar-root residual |
| Nonlinear Cyclone / Dimits | Saturation and zonal regulation | [Hoffmann–Frei–Ricci](https://arxiv.org/abs/2308.01016); seeds, gradient scan, hysteresis/cold–warm starts |
| W7-X/HSX and VMEX QA/QH/QI | Stellarator transport/domain coverage | [stella–GENE](https://arxiv.org/abs/2107.06060)/GX; surfaces, alphas, tube lengths and all resolutions |
| Linear/QL/nonlinear gradients | Declared numerical derivative | Analytic small systems, directional Taylor/FD, CPU/GPU/sharded transpose identity |
| ESSOS fields / islands | Geometry, equilibrium and parallel/perpendicular transport | Analytic Biot–Savart and island fields; MMS and nested limit; R8 matched GENE-X/XGC-S model, topology/mesh/background ladders |

Set tolerances before comparison using numerical, published, digitization and
normalization uncertainty. Proposed starting targets: ≤1% resolved linear
eigenvalue error (absolute floors near zero), ≤5% nonlinear discretization error,
and a tighter budget for small design gains. Justify each case's actual limits;
do not fit them to today's answer. Wide overlapping intervals alone are not an
equivalence test.

### Resolution/stopping campaign

Use low/standard/cautious settings on tokamak and QA/QH/QI equilibria, including
near-threshold, overshoot, intermittent and slowly drifting cases. CPU handles
small ladders; office GPUs handle long paired runs. Pilot cost before escalation.

1. Refine `Nx,Ny` at fixed box for cutoff; enlarge `Lx,Ly` at fixed maximum k
   for box effects. Record **retained dealiased** modes. Refine `Nz` separately
   from physical tube length/turns and alpha.
2. Refine `Nl,Nm` separately/jointly with physical collisions fixed; vary closure
   and artificial dissipation. Grid refinement alone does not test their bias.
3. Halve dt under one damping contract. Use common physical-time sampling;
   unequal adaptive intervals need weighted quadrature/resampling, not plain means.
4. Retain `Q_s(t)`, `Gamma_s(t)`, `Wphi`, distribution/total free energy,
   zonal/nonzonal power and averaged `kx,ky,l,m` spectra.
5. Replay stop candidates on **causal prefixes**, then test against unseen future
   data and independent seeds. Score burn-in bias, stationarity persistence,
   ACF/batch uncertainty, interval coverage and false-stop rate.
6. Choose the cheapest policy meeting predeclared accuracy/coverage. Flat Wphi
   at t=50–55 is insufficient: Q, slow Wg and the correlation time matter.

For equally spaced stationary data,

\[
 \bar Q=N^{-1}\sum_nQ_n,\quad
 N_{eff}=\min\left(N,{N\Delta t_s\over2\tau_{int}}\right),\quad
 \mathrm{SEM}(\bar Q)=s_Q/\sqrt{N_{eff}}.
\]

The current first-zero ACF cutoff and `10 tau` minimum are heuristics, not
coverage proofs. Compare batch means/spectral variance on oscillatory ACFs and
intermittency. Calibrate sequential stopping: fixed-window 95% coverage does not
automatically survive optional stopping. Use
[Oberparleiter et al.](https://publications.lib.chalmers.se/records/fulltext/247070/local_247070.pdf)
and [Vaezi–Holland](https://arxiv.org/abs/1902.10879) as anchors.

Estimate paired differences/ratios with long-run covariance; separate between-seed
and within-trace errors. Predeclare ratio-of-means versus mean-of-ratios. Keep
failures and inconvenient seeds. Final selection uses new held-out seeds and a
lower confidence bound above a practical gain after discretization error; no
universal seed count guarantees this. Extend precision when uncertainty is larger
than the desired gain.

## 5. R3: performance and actual parallelization

### Protocol and workloads

Record SHA/deck/state hashes, software, hardware/topology, dtype, thread/device
placement and allocator/cache settings. Separate import, geometry/setup,
transfers, trace/compile, warm primal, warm value+gradient, diagnostics/I/O and
total time-to-accepted-result. Synchronize work; use fresh-process cold runs and
≥5 warm repetitions. Report medians/spread, peak RSS and device peak. Compiler
temporary bytes are not measured peak memory.

| Workload | Sweep |
|---|---|
| Linear ITG/kinetic/EM | Eigenpair + ky scan, matched residual/time-to-accuracy |
| Nonlinear Cyclone/stellarator | Small/medium/production; RHS, RK step, physical horizon, accepted mean |
| Saturated-window VJP | Window × parameter count × checkpoint policy; nonzero transport, Taylor checks |
| Independent work | 1/2/4/8 CPU workers, 1/2 GPUs; ky, surfaces/alphas, seeds, QL |
| Single distributed trajectory | Species/Hermite versus domain layouts; 1/2/4 logical CPUs for correctness, 1/2 GPUs on actual topology |

First make provenance work for archive installs and separate warm traces from
compilation. Do not call a cross-JAX-version CPU/GPU ratio a hardware-only speedup.

Use a matched JAX/dependency environment when possible, with the existing dated
profiles as a baseline, not a substitute for a full workload sweep. First profile
one nonzero-transport ES case, one kinetic-electron EM case and one advanced
collision case on each backend. Trace streaming/mirror/drifts, collision setup
and application, field solve, FFT/bracket, projector, diagnostics, transfers and
collectives separately. Record bytes/state and retained checkpoints as well as
allocation peaks. Build a measured cost model before changing layout or algebra.
Expand to QA/QH/QI, resolution and parameter-count ladders only after these
profiles explain the dominant cost. Every performance PR repeats affected
end-to-end cases; a faster RHS with a slower accepted mean is not a win.

### Sharding acceptance ladder

1. Share RK/projector/field/damping/collision semantics. Exercise nonlinear terms,
   kinetic species, linked boundaries and all promoted operators; reject gaps early.
2. First candidate: species×Hermite ownership, local perpendicular FFTs, low-order
   field reductions and width-two Hermite halos. Verify global moment indices,
   Laguerre truncation, species coupling and reduction precision.
   Exercise meshes `(1,2)` and `(2,2)` where devices permit, not only `(2,1)`:
   splitting species alone never tests an inter-device Hermite halo. Dense
   collision coupling may require a different communication plan; sparse
   streaming does not make all collisions nearest-neighbor.
   Current `_reject_unsharded_hermite_terms` rejects conserving collisions when
   m is split: first implement global low-order moment reduction and its VJP,
   with collision-on/off and supported-species tests. Include staging/resharding
   overhead and test the traced path separately from host initialization.
3. For spatial decomposition specify pencil layouts, transforms and boundaries.
   Split/reassemble metadata is not a distributed nonlinear solve. Compare
   communication/memory with velocity decomposition on office's PCIe topology.
4. Test actual sharded RHS, trajectory, spectra/flux, restart and VJP:
   `Re<v,J u> = Re<J* v,u>` with the declared weights/complex convention.
5. Short trajectories require numerical identity; chaotic long runs require
   statistical equivalence, not pointwise matching.
6. Fix or explicitly bound CPU FFT-layout failures with subprocess tests. A skip
   and four logical devices do not establish working state sharding.
7. Promote correct execution separately from speedup. Report strong/weak scaling,
   `S_p=T_1/T_p`, `E_p=S_p/p`, memory and communication. Keep serial below the
   measured crossover. Do not promise speedup for every small case.

Use current [JAX sharding](https://docs.jax.dev/en/latest/201/sharding.html),
[shard_map](https://docs.jax.dev/en/latest/201/shard-map.html) and
[AD/sharding](https://docs.jax.dev/en/latest/301/sharding-ad.html) contracts.

### Algorithm experiments, in order

- Reuse prepared kernels/bounded caches; remove traced host conversions; save
  online scalars rather than distribution histories when possible.
- Profile materialized copies, full/half-spectrum completion and FFT layout.
  Accept microkernel work only if full primal **and VJP** benefit.
- Streaming couples `m±1`, drifts can couple `m±2`, Laguerre recurrences couple
  nearby orders. Derive block/banded preconditioners and their transposes.
  Fields, collisions and nonlinear Fourier coupling mean the full turbulent
  Jacobian is **not** a simple block-tridiagonal solve.
- Reuse SOLVAX only when a matched stiff workload beats explicit RK in accuracy,
  time and memory. Preserve the old plan/log's rejected stiff/FFT experiments;
  another integrator family needs a new measured reason.
- Mixed-precision refinement, offload and custom kernels are later options;
  require conservation/residual/AD tests and CPU fallback.

## 6. R4: slimming without removing science

Measure tracked bytes, fresh-clone pack, wheel, physical/AST lines, files,
dependency graph, import and collection/runtime separately. The original
**<10 MB ordinary full clone** goal remains (use the stricter 10,000,000-byte
transfer target and also report MiB); a <20 MiB working tree is not its replacement.
History reduction needs backup, ref inventory, tested rewrite and old→new map.
Deleting blobs or changing authors cannot preserve identical full history.

| Change, in order | Preserve | Evidence |
|---|---|---|
| Resolve model/config and validation once | Defaults, rejection, errors, extension seams | Fewer independent decisions; schema/CLI regression |
| Share RK/projector/field/damping | All promoted primal/AD paths | Removed duplicate algebra; stage-map CPU/GPU tests |
| Simplify objective/report adapters | Scientific API/custom physics | Consumer graph; 346 lazy-name resolutions or explicit deprecation |
| Consolidate tools by workflow | Regeneration, provenance, negative results | Fewer independent generators/commands |
| Share test fixtures/invariants | Distinct cases and diagnostic failures | Node-ID map, assertion/mutation equivalence and CI time |
| Remove redundant JSON/traces | Runtime tables and auditable evidence | Consumer graph; NetCDF arrays, CSV/TOML tables, equivalent regeneration |

The previous ≤45 source files/45k lines and ≤30 test files/35k lines are
**long-term design targets, not permission to delete capability or tests**.
Re-estimate after a domain pilot and keep targets only with a concrete reduction
map. Avoid cap-driven splitting, blind helper inlining and mega-modules. A test
reference alone does not prove code is useful or reachable in production.

Keep one TOML input and NetCDF result/restart contract. Retain compact summaries
only with consumers; zero JSON is not a reason to break collision data. Large
runs/traces/media belong in immutable release/research archives with hashes and
retrieval tests. Git keeps old plans; do not duplicate them in new tracked archives.

## 7. R5: documentation and examples

Keep Sphinx/RST/MathJax initially; a theme/markup migration is not necessary for
organization. Preserve URLs when consolidating pages. Follow the
[Diátaxis](https://diataxis.fr/) reader-task separation.

| Reader need | Canonical content | Consolidate from |
|---|---|---|
| Learn | Install; first linear/nonlinear/stellarator/AD lessons | `quickstart`, beginner `examples` |
| Do research | Resolution/stopping, restart, CPU/GPU, scans, VMEX/optimization | `inputs`, `outputs`, `parallelization`, `stellarator_optimization` |
| Understand | Model/normalization, moments, geometry/boundaries, fields/collisions, numerics/statistics/AD | `theory`, `linear_model`, `operators`, `numerics`, `algorithms`, `solvers`, `nonlinear_autodiff` |
| Look up | API/CLI/schema, benchmark/limitations, citations | `api`, `benchmarks`, `verification_matrix`, `release_scope`, `references` |
| Develop | Ownership, test/benchmark commands, contributions | `architecture`, `testing`, `code_structure`; governance in `plan/` |

Each topic needs purpose, assumptions, defined equations, owner, runnable command,
observable/units, plot, convergence check and failure interpretation. Use equations,
tables and captions instead of repeated motivation. Follow VMEX's visible editable
parameters and concise examples, testing against pinned companion versions.

### Gallery

One script/deck per purpose, using real library APIs. Distinguish explicit
**smoke / teaching / research** presets; a small API demonstration must not emit
a research convergence verdict.

| Example | Teaching result | Research extension |
|---|---|---|
| ITG eigenmode | Gamma/omega/eigenfunction with units | ky/gradient threshold and convergence |
| Nonlinear Cyclone | Q/Wphi/Wg and accepted/rejected window | Dimits/closure/cost ladder; seeds |
| Kinetic/EM | Species response, KAW/KBM/TEM | Real mass ratio, beta/collision scans |
| Collisions/closures | Relaxation, entropy, recurrence | C1–C4 support/convergence; transport sensitivity at fixed physical collisionality |
| Er/shear | Analytic shearing wave and suppressed/enhanced response | E0–E2 scans; no universal suppression claim; stellarator root/ordering checks |
| VMEX→GKX | 3-D surface, geometry, linear spectrum | QA/QH/QI, surfaces/alphas |
| Linear QA optimization | VMEX tuples + ITG term | Mode tracking, constraints, held-out unstable branches |
| QL QA optimization | Normalized spectral-flux objective | Family holdout/ranking and nonlinear follow-up |
| Nonlinear QA | Saturate→detach→window value/grad | R7 restartable campaign and independent acceptance |
| ESSOS coils | Fit accepted target surface | R8a topology/transport robustness |
| Coil-field/island geometry | Poincaré plot, manufactured streaming | R8b–e nested limit, physical background and converged transport/gradient |
| Parallel scan/window | Serial/distributed identity | CPU/GPU crossover and memory |

Expose physics, seed, resolution, devices and output path without new one-off
CLIs. Test smoke scripts out of tree from an installed wheel; optional dependency
failures must be clear. Research presets state hardware, wall time and output size.
Each tutorial changes one physical question, explains a failed convergence check,
and offers one short exercise. Use shared library/campaign components for restart,
UQ and plotting; an example should not contain a second solver. Generate the
capability/limitations table from the canonical ledger and link unsupported
presets to the corresponding milestone rather than shipping plausible outputs.

README order: purpose/scope → install/result → equilibrium → Python/AD →
parallel/performance → validation/limitations → examples/docs/citation.
Use a few regenerable panels: benchmark, time/memory, derivative validation,
and **after R7 passes** initial/final 3-D LCFS, common-scale Boozer LCFS and
matched Q(t)/UQ. Existing QA figures remain preliminary. Add coils after R8.

Startup concisely defines gamma, omega/sign, Wphi, Wg, Q/species, tprim/fprim
with the deck's normalization, electrons, collisions/damping, grid/dt and stop
targets. Movies use solved physical frames, real time, stellarator nfp/coordinates
and correct seams. Lightweight xy/3-D previews link to full data. Every figure
has units, accepted interval, seed/resolution, provenance and claim scope.

## 8. R6–R7: derivatives and optimization

### One nonlinear method

Keep checkpointed reverse AD of the **exact projected finite-step map** as the
single public nonlinear method. Checkpoint scheduling changes memory, not the
derivative. For fixed seed and window,

\[
 J_N(x;G_0)={1\over N_w}\sum_{n=N-N_w+1}^{N}Q(G_n,x),\quad
 G_{n+1}=\Phi_{\Delta t}(G_n,x),\quad G_0=\mathrm{stop\_gradient}(G_{sat}).
\]
\[
 \lambda_n=(\partial_G\Phi_n)^*\lambda_{n+1}
       +{\mathbf1_{n\in W}\over N_w}\partial_GQ_n,\qquad
 d_xJ_N=\sum_{n=0}^{N-1}(\partial_x\Phi_n)^*\lambda_{n+1}
       +{1\over N_w}\sum_{n\in W}\partial_xQ_n.
\]

Terminal condition: `lambda_N = 1_{N in W} partial_G Q_N/N_w`. Use the
real-loss complex-adjoint convention. Freeze topology, initial state, dt/window
and stop decisions within evaluations; include explicit geometry/field/diagnostic
dependence. State that spin-up and adaptive stopping are excluded.

For state size S and N steps, block length b retains `O(S(N/b+b))` state memory,
minimized near `b≈sqrt(N)`, plus caches/temporaries/requested outputs. It does not
remove chaos. An implicit steady-state adjoint assumes a converged residual;
a turbulent stationary distribution is not a fixed point `F(G)=0`.

[iGENE](https://arxiv.org/abs/2605.03086) reports biased useful short-window
directions and eventual divergence; [gyaradax](https://arxiv.org/abs/2604.06085)
is another 2026 differentiable GK code. Neither “first differentiable GK code”
nor “exact saturated transport gradients in almost all cases” is an acceptable
novelty claim. Shadowing/linear response remain research alternatives if window
directions fail, requiring assumptions, unstable-subspace cost and independent
derivative evidence before adding a second public method.

### Linear and QL

For a simple eigenpair `A r=lambda r`, `l* A=lambda l*`, `l*r=1`,

\[
 d\lambda=l^*(dA)r.
\]

Gate both residuals, conditioning and branch separation. Near crossings, track
several modes and report smooth-envelope dependence. A matrix-free implicit
eigenpair derivative is a different mathematical problem from nonlinear-window
AD; retain it where valid ([Acton et al.](https://arxiv.org/abs/2403.12621)).

QL combines eigenmode flux weights with a declared saturation model. Distinguish
ranking proxy, calibrated predictor and transport closure. Train on training
equilibria only; hold out whole families/gradients, retain stress outliers/OOD
rejection, compare simple gamma/mixing-length baselines and report rank/sign
errors. ES calibration does not promote EM transport prediction.

### VMEX and R7 acceptance

Follow current VMEX's tuple-based **scalar loss** and reverse implicit equilibrium
gradient; avoid a large residual Jacobian when a scalar gradient suffices. Keep
its visible controls/reporter/scaling and add one normalized GKX term. Keep Ln/LT
finite and fixed in vacuum QA tests; drives must not become escape parameters.

1. Verify boundary→solved equilibrium→geometry/mesh→GKX directional derivatives,
   including minor radius, shear, Jacobian, drifts and flux quadrature. Gate
   equilibrium residual and nestedness before evaluating transport.
2. Start with 2–8 controls; progress to 50–200 only after cost/memory scaling and
   low-dimensional held-out success.
3. Use several surfaces/alphas and relevant ky branches; hold out others. Keep
   QA/aspect/iota/well and finite-beta constraints explicit.
4. Measure gradient variance/sign/Taylor remainder over windows and independent
   saturated states. Compare FD step ladders and equal-budget derivative-free/
   SPSA controls. Use physical time and measured divergence, not a universal
   “512 steps” rule.
5. Use bounded/trust-region, gate-aware steps. Detached seeds remain fixed in the
   inner solve and refresh at recorded accepted outer stages. Test warm starts
   against cold seeds and verified re-equilibration.
6. Freeze the chosen boundary, then run **new** paired ensembles with identical
   physics/horizons/sampling. Apply R2's time/box/spatial/moment/closure ladders
   to both designs; resolve the difference and constraints, not just baseline Q.
7. Publish inputs, WOUT/geometry/seed hashes, Q(t), spectra, UQ and initial/final
   panels, including failures. Independently reproduce one matched-model result.

## 9. R8: ESSOS, non-flux geometry and islands

```text
boundary x → solved VMEX → validated flux-tube geometry → GKX J(x)
       ↓ accepted target surface
ESSOS coils c → Biot–Savart B(c) → normal-field/topology checks
       → realized equilibrium or validated nested-surface reconstruction
       → new geometry → held-out GKX transport
```

**R8a — nested realization.** First fit coils with normal-field, length,
curvature, separation, current and
manufacturability constraints. Then check Poincaré surfaces, islands/iota and
stochastic regions: small surface B·n alone does not validate the interior.
Recompute transport on the realized nested equilibrium and under coil/current
errors. Free-boundary VMEX or a validated vacuum-surface adapter is a prerequisite,
not an assumed capability. This route rejects islands; the following route is
specifically responsible for supporting them. Start its model work alongside
R1/R2, not after the final nonlinear optimization campaign.

| Step | Deliverable | Exit before promotion |
|---|---|---|
| R8b | Direct-field geometry from ESSOS `B(x;c)` and derivatives | Analytic coil tests, div B, Jacobians/metric/orientation, Poincaré/iota/island widths; smooth-coil AD against FD; no nested-surface assumption |
| R8c | Non-flux spatial representation | Compare FCI/field-line interpolation with suitable global meshes; derive conservative parallel derivative, gyroaverage and polarization/Ampère solve, boundaries and transposes; MMS including an island/separatrix and nested limit |
| R8d | Consistent background and nonlinear evolution | State δf validity, profiles/sources and equilibrium residual or implement required evolving-background/full-f extension; ES then EM; conservation, topology/mesh/velocity/dt convergence and independent code comparison |
| R8e | Coil-field transport optimization | R8b–d + R7 statistics; gradients away from topology transitions, documented event/non-smooth limits, held-out coil errors and transport; direct versus reconstructed field comparison |

In an island, a globally nested flux label need not exist. A prescribed
`F0(ψ)` from VMEC cannot simply be pasted onto that field. For a proposed
background, explicitly evaluate its kinetic residual

\[
 R_{0s}=\dot{\mathbf R}_0\cdot\nabla F_{0s}
       +\dot v_{\parallel0}\partial_{v_\parallel}F_{0s}
       -\sum_b C_{sb}[F_{0s},F_{0b}]-S_{0s}.
\]

Report its size relative to the retained ordering. A uniform Maxwellian with
manufactured forcing is a verification case, not a physical island transport
prediction. Finite-beta ESSOS coil fields alone omit plasma-current response:
specify total equilibrium field/force balance or a justified kinetic background.
Open field lines need explicit particle/energy boundary conditions; a periodic
cut must not silently replace losses. Preserve the optimized local Fourier
backend; introduce only the spatial abstraction needed by the verified model.

Use [GENE-X's stellarator extension](https://doi.org/10.1016/j.cpc.2026.110138)
and [XGC-S verification](https://arxiv.org/abs/1905.05653) to design independent
geometry/transport checks, not to assert identical physical models. FCI still
requires conservative interpolation and an appropriate perpendicular field
solve; it is not just a replacement for the parallel derivative.

Joint coil/equilibrium/GKX AD requires the reconstruction/field-map Jacobian and
topology-aware perturbation tests. Extend the accepted applications to QH/QI,
finite beta, kinetic electrons and transport coupling. Pyrokinetics can help
interchange/normalization, but adapter success does not replace a matched-model audit.

## 10. R9: publication and release

| Candidate paper | Claim | Evidence/figures |
|---|---|---|
| Methods/code | JAX moment GK, verified algorithms, bounded-memory AD, CPU/GPU execution | R0–R5; equations, MMS/linear/zonal/nonlinear convergence, measured time/memory and primal/VJP scaling |
| Differentiable design | Useful high-dimensional gradients and resolved QA reduction | R6–R7; Taylor/window/variance, equal-budget controls, held-out Q, constraints/geometry |
| Coil-realizable design | Benefit survives coils/errors | R8; coils, normal field/topology, realized equilibrium and robust transport |
| Collision/geometry methods, if novel | Efficient advanced Coulomb or differentiable non-flux GK | C1–C4 or R8b–e; independent coefficient/operator/physics checks, conservative limits, accuracy-cost/AD scaling; separate from a generic code-feature claim |

These are candidates, not promised novelty. Combine papers if contributions do
not justify separation. Experimental validation needs its own measurement study.
After benchmark closure, select a device/discharge with equilibrium, profiles,
uncertainties, heating and fluctuation/transport diagnostics. Predeclare synthetic
diagnostics and validation metrics; propagate input and numerical uncertainty,
retain discrepancies and use held-out conditions. Agreement with GX alone is
not predictive experimental validation. A code/methods paper need not wait for
all future models, but its abstract must state the validated model subset.

### Test and CI tiers

- Tier 0: algebra/contracts/tiny derivatives; <60 s focused CPU target.
- Tier 1: bounded integration, default f32/f64, wheel examples, restart/I/O,
  docs/links; measured <10 min shards where feasible.
- Tier 2: nightly/release physics, CPU/GPU and sharded primal/VJP ladders.
- Tier 3: long external/statistical/optimization campaigns; raw data retained,
  fast digest/metric checks in CI.

Tests run independently of lint/type checks. Isolate backend crashes and expire
skips with supported-version probes. Coverage/collection are secondary checks:
retain scoped ≥95% statement/branch ambitions, but report missing physics and
mutation gaps separately. Solver changes run their physics sentinels even when
no assertion changed.

### Every PR records

`scope/non-goals → equation/API contract → tests and expected failure → measured
accuracy/runtime/memory/size delta → evidence digest → accepted/rejected/partial
→ rollback → next ID`.

Pin companion versions/raw sources and preserve upstream licenses/provenance.
No AI co-author trailers; do not erase legitimate upstream credit. Shared
algorithms/data weaken independence of code-to-code agreement.

Release only when claimed rows pass on the release wheel and examples regenerate
their results. Archive manuscript inputs, observations, scripts and environment
under an immutable identifier and reproduce in a clean environment. Update
README/abstract claims last, directly from accepted results.
