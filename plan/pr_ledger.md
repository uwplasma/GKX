# Pull-request ledger, #1–#197

Reconciled with GitHub on 2026-09-04 at `a99dac89`: 191 pull requests,
180 merged, 9 closed unmerged, 2 open. Issue-only or absent numbers in this
range: #2, #9, #173, #178, #192, #194. The original #1–#162 inventory was
recorded at `e19336dc`; its entries remain below.

Historical detailed reviews live in `plan/pr_audit.md`; the September review is
`plan/baseline/review_2026_09_04.md`. Listing a merge is not an independent
scientific certification of its changes.

## Open at this audit

| PR | Head | Title |
|---:|---|---|
| [#196](https://github.com/uwplasma/GKX/pull/196) | `fdfb1a13` | Skip the compressed nonlinear gradient test where jaxlib 0.10.2 segfaults |
| [#197](https://github.com/uwplasma/GKX/pull/197) | `9074dd87` | fix: restore the per-step end-damping contract that 2.0.0 removed (#192) |

## Closed without merge

| PR | Title |
| ---: | --- |
| #4 | [codex] Add differentiable velocity-map primitives |
| #5 | Add mapped linear velocity operators |
| #6 | Wire velocity maps into linear RHS factors |
| #25 | eigensolver: raise on unknown shift-invert preconditioner names |
| #82 | Living research-grade roadmap (do not merge) |
| #106 | Deduplicate the VMEC geometry facade |
| #133 | Fix runtime startup profiler handoff |
| #159 | geometry: add the canonical VMEX WOUT adapter |
| #182 | refactor: rejoin four modules that were split rather than encapsulated |

## Merged

| PR | Title |
| ---: | --- |
| #1 | Adding internal generator of geometry eik file inside SPECTRAX-GK ins… |
| #3 | feat: Add CLI overrides for vmec_file, geometry_file, init_file |
| #7 | [codex] Plan differentiable architecture refactor |
| #8 | Refactor toward GKX 2.0 |
| #10 | release-prep: unpin solvax, drop MANIFEST.in and plan.md, add readthedocs |
| #11 | release: bump to 1.7.1 and fix the first-publish PyPI version check |
| #12 | ci: give wide-coverage shards a realistic per-file timeout |
| #13 | collisions: honour the TOML collision_operator selection at runtime |
| #14 | collisions: ship drift-kinetic and gyrokinetic Coulomb operators |
| #15 | physics: verify the collision operators, fix Laguerre conditioning, ship higher resolutions |
| #16 | objectives: opt in to non-symmetric eigenvector derivatives |
| #17 | ci: unblock Codecov-side failures; physics: gate Spitzer-Harm conductivity |
| #18 | docs: add a capability comparison; restore strict Codecov uploads |
| #19 | perf: measure and bound the finite-Larmor collision cost |
| #20 | tools: make the generator's b->0 check follow the requested resolution |
| #21 | geometry/fields: three silent physics defects found by source audit |
| #22 | Reflectionless Hermite closure for recurrence, plus README model documentation |
| #23 | Differentiable matrix-free eigensolver for linear and quasilinear GKX |
| #24 | runtime: keep adaptive chunk diagnostics on the host |
| #26 | Nonlinear turbulence gradients: correlation time, saturated state, windowed adjoint |
| #27 | Nonlinear window statistics: independent samples, not output count |
| #28 | A4: Cyclone linear ITG threshold, measured on the case's own operator |
| #29 | Name the gradient parameters for the quantity the operator consumes |
| #30 | Give the linear objective a velocity-space closure so it can report stability |
| #31 | Raise on unknown shift-invert preconditioner names (replaces #25) |
| #32 | fix: Copy each strided diagnostic chunk in the runtime chunk loop instead of viewing it |
| #33 | Report what a nonlinear run's simulated time cost, and which CFL term set it |
| #34 | Re-measure device-z pencil two-GPU scaling at production granularity |
| #35 | Route the parallel policy into the nonlinear solver path |
| #36 | Compute the device-z pencil bracket with the local fused transform |
| #37 | Close the Laguerre truncation in the sharded diamagnetic drives |
| #38 | Compute the device-z observable sums with the local fused transform |
| #39 | Retire the staged pencil transforms now that nothing calls them |
| #40 | Compensate the device-z transport observable reductions |
| #41 | Make the declared Python floor real instead of advertised |
| #42 | Play the turbulence movie in the README instead of a still |
| #43 | Pin exact dot precision on the contractions that carry a conserved quantity |
| #44 | Finish the TF32 audit: pin the two contractions that need it, and guard by shape |
| #45 | Measure linear parity against GX across six cases and close the two-device bracket number |
| #46 | Make fit_signal = "auto" choose a signal, not a fit window |
| #47 | Measure every residual gate against the precision that produces it |
| #48 | Consolidate nonlinear heat-flux autodiff and QA optimization |
| #49 | Research-grade plan and work log |
| #50 | Fail loudly on overflowed fits, plot ky scans, and pin dependency floors |
| #51 | Run a VMEC equilibrium directly: gkx wout_XXX.nc |
| #52 | Certify the Krylov linear branch instead of trusting it |
| #53 | Add a publication figure library and share the snapshot renderers |
| #54 | Stop nonlinear runs when the heat flux has converged |
| #55 | Mark the Cyclone GX probe as transient, not a parity target |
| #56 | Report whether a growth rate is resolved, not just its value |
| #57 | Give the differentiable bridge the curvature drift it was missing |
| #58 | Plot every run and stop recompiling the same kernels |
| #59 | Differentiate through the standard sheared flux tube |
| #60 | Say which thermal velocity the normalization means |
| #61 | Differentiate the compressed real FFT path |
| #62 | Shard a nonlinear run across devices by species and Hermite |
| #63 | Give the equilibrium shorthand a coverage owner |
| #64 | Read another code's output without naming it in the figure code |
| #65 | Log the post-merge defects and the next wave of plan work |
| #66 | Run the Cyclone linear examples to a converged horizon |
| #67 | Baseline the zonal artifact where it is a measurement |
| #68 | Refuse to call a trace that never left zero saturated |
| #69 | Log the Merlo re-baseline, the saturation regression, and the GX flag result |
| #70 | Read host values with numpy, not by round-tripping through jnp |
| #71 | Carry converged state between related runs, off by default |
| #72 | Log the traced-read sweep, warm start, and a methodological warning |
| #73 | Audit every deck against the stop policy, and stop trusting a flat trace |
| #74 | Check the reused scan integrated the new state, not that time is finite |
| #75 | Let the tests run when the type check fails |
| #76 | Differentiate the cases nobody had, and make the claims regenerable |
| #77 | Keep the bracket out of the linear operator's fusion |
| #78 | Log wave 5: the audit, CI signals, autodiff coverage, and the fusion |
| #79 | Say what a file is when it is not the expected TOML |
| #80 | Give a wout run a summary page and a real output file |
| #81 | Fix main CI typing and nonlinear timeout |
| #83 | Keep the research roadmap out of main |
| #84 | Stop nonlinear runs at the exact physical-time horizon |
| #85 | Explain nonlinear inputs and live diagnostics at startup |
| #86 | Render nonlinear flux tubes in physical VMEC coordinates |
| #87 | Warn when nonlinear spectra reach the ky cutoff |
| #88 | Remove unreferenced generated documentation assets |
| #89 | Fail closed on QA transport stationarity |
| #90 | Require auditable nonlinear promotion evidence |
| #91 | Capture fixed-horizon saturation audit traces |
| #92 | Decouple generated renders from release evidence |
| #93 | Delegate the differentiable geometry facade to its core implementation |
| #94 | Do not present rejected saturation windows as averages |
| #95 | Keep only the public documentation figure set |
| #96 | Make turbulence movies physical and lightweight |
| #97 | Continue production states for turbulence movies |
| #98 | Use one dealiased spectral layout |
| #99 | Use one growth fit input path |
| #100 | Match nonlinear adjoint claims to evidence |
| #101 | Correct the diagonal preconditioner description |
| #102 | Compact reproducible optimization sidecars |
| #103 | Compact the Landau validation preview |
| #104 | Compact retained README plots |
| #105 | Deduplicate linear integration policy |
| #107 | Use one runtime linear fit option record |
| #108 | Reuse resolved transport channel evaluations |
| #109 | Use one Diffrax state setup path |
| #110 | Share objective policy helpers |
| #111 | Share nonlinear diagnostic dependency bindings |
| #112 | Share gyrokinetic moment shape |
| #113 | Share species parameter order |
| #114 | Share spectral geometry contractions |
| #115 | Keep the research roadmap out of main |
| #116 | Run the QHS/QI adaptive gates at the cached fixture's resolution |
| #117 | Scan-calibrated defaults and a deterministic resolution estimator |
| #118 | Check a fixed step against the CFL bound nobody was checking |
| #119 | Restyle the examples to direct-parameter scripts |
| #120 | Repoint verification claims at surviving evidence and fix stale docs |
| #121 | Stop shipping an ETG example that integrates NaN |
| #122 | Living research-grade roadmap (do not merge) |
| #123 | Evict research-campaign governance from the installable package |
| #124 | Reorganize the README around tasks and pin the linter |
| #125 | Stop streaming the nonlinear state through a degenerate GEMM |
| #126 | Make gkx wout.nc --linear run |
| #127 | Cover the three defects that made --linear fail |
| #128 | Name artifact sidecars the way the writers do |
| #129 | Fix linear-integrator benchmark imports |
| #130 | Repair optional diagnostic dependency paths |
| #131 | Track Phase 0 performance baseline |
| #132 | Fix prepared runtime profile fingerprints |
| #134 | Freeze GKX 1.8.2 output schemas |
| #135 | Record GX-derived software provenance |
| #136 | Repair nonlinear adjoint profiler invocation and grid override |
| #137 | Fix runtime startup profiler handoff |
| #138 | Complete Phase 0 ownership and evidence audits |
| #139 | Add zero-copy Case and Result contracts |
| #140 | Add thin load, solve, and scan workflows |
| #141 | Promote scan and plot CLI aliases |
| #142 | Version runtime TOML and NetCDF schemas |
| #143 | Reduce the advertised GKX root facade |
| #144 | Consolidate electrostatic field moments |
| #145 | Add the in-memory Python plot contract |
| #146 | Use dtype-aware tolerance for adaptive JIT parity |
| #147 | Add the prepared nonlinear Python contract |
| #148 | Consolidate cached linear RHS assembly |
| #149 | Consolidate Hermitian ky projection |
| #150 | Keep end damping independent of step size |
| #151 | Promote native linear example defaults |
| #152 | Consolidate native linear stepping |
| #153 | Support implicit linear diagnostic sampling |
| #154 | Record native stiff-solver GPU triage |
| #155 | Record coupled fixed-work implicit rejection |
| #156 | Add canonical VMEX state geometry adapter |
| #157 | geometry: consume closed VMEX mirror flux tubes |
| #158 | tests: land the WOUT linear regression coverage on main |
| #160 | geometry: add the canonical VMEX WOUT adapter |
| #161 | docs: review open-ended mirror model admission |
| #162 | geometry: remove synthetic Boozer closure |
| #163 | plan: replace the root plan and rebaseline current main |
| #164 | core: inventory the import graph and delete the toml shim |
| #165 | api: give Case and the result types their contracted behaviour |
| #166 | api: make PreparedSimulation a real public object |
| #167 | cli: complete the six product commands |
| #168 | geometry: evict the parity and sensitivity report builders |
| #169 | solvers: remove Diffrax from the base product |
| #170 | tests: consolidate the nonlinear unit domain from nine files to three |
| #171 | docs: restructure the README around install and usage, and cut AI prose patterns |
| #172 | tests: consolidate the solvers unit domain from nine files to three |
| #174 | tests: add a shared fixture layer for the linear domain, and measure its reach |
| #175 | tests: consolidate the runtime integration domain from eleven files to four |
| #176 | tests: consolidate the physics-gate domain from ten files to four |
| #177 | Say what the closed VMEX mirror ratio actually measures |
| #179 | Make the tree ruff-format clean, and gate it in CI |
| #180 | examples: repair the differentiable-geometry bridge and stop gating on deleted capability |
| #181 | plan: measure what src/gkx lines actually do, and plan simplification from it |
| #183 | plan: measure the directory structure and plan a flat source layout |
| #184 | Measure import laziness in a fresh interpreter, not in a polluted one |
| #185 | refactor: flatten containers, fuse split modules, and lift the case type |
| #186 | docs: gate the documented first run and the reference links |
| #187 | perf: compile the seed nonlinear diagnostic once instead of per primitive |
| #188 | release prep: first-run gate on the release wheel, and bump to 2.0.0 |
| #189 | Keep every scan-level parallel strategy out of the per-ky solve |
| #190 | fix: seed linear_rhs_demo on a mode the grid actually has |
| #191 | Build the closed VMEX mirror case on a solved equilibrium, and delete the record with no generator |
| #193 | test: recompute the README's parity table from the scans it cites |
| #195 | Make the regenerated release artifacts reproducible and fail CI on staleness |
