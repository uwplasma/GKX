# GKX 3.0 research-grade modernization plan

**Status:** active ground truth and agent handoff  
**Replaces:** the current root `plan.md` at GKX `main`  
**Audited revision:** `e19336dc2202b721d12df4f27ab84835b1360de7`  
**Audit date:** 2026-08-30  
**Final merger to `main`:** `rogeriojorge`  
**Minimum Python:** 3.11  
**Required execution platforms:** CPU and NVIDIA GPU  

This file states the accepted product scope, the current repository state, the work already completed, the remaining gaps, and the ordered implementation program. A new agent should be able to clone GKX and the named companion repositories, read this file, select the first unfinished task, implement it in a focused branch, and leave a complete handoff.

Historical measurements, rejected experiments, and detailed run narratives belong in `plan/log.md` or `plan/archive/`. Do not append day-to-day logs to this file. Update this file only when the product contract, architecture, phase status, or ordered work queue changes.

---

## 1. How to use this handoff

At the start of every work session:

1. Read this file completely.
2. Read the last two entries in `plan/log.md`.
3. Run `git status --short --branch` and `git rev-parse HEAD`.
4. Fetch `main`, inspect open pull requests, and confirm that no other branch owns the same task.
5. Run the current architecture, repository-size, release, and focused scientific gates before editing.
6. Work on one task from the **Immediate pull-request queue**. Do not combine unrelated cleanup, physics, performance, documentation, and generated artifacts.
7. Record exact before/after file counts, line counts, public names, tests, coverage, runtime, memory, and numerical changes that are relevant to the task.
8. Append a concise entry to `plan/log.md` using the template near the end of this file.
9. Push a reviewable branch and pull request. Do not merge it. `rogeriojorge` is the final merger.

When code and this plan disagree, inspect the Git history and tests. Correct the plan in the same pull request that deliberately changes the contract. Do not silently reinterpret a completed gate.

---

## 2. Accepted scope and decisions

These decisions are closed unless `rogeriojorge` explicitly reopens them.

### 2.1 Physical scope

GKX 3.x is a **radially local, Maxwellian-background, delta-f, flux-tube gyrokinetic code** for tokamaks and stellarators.

The stable scope includes:

- linear, quasilinear, and nonlinear local gyrokinetics;
- electrostatic and electromagnetic fluctuations;
- adiabatic or kinetic electrons;
- arbitrary kinetic ion and impurity species within the supported model;
- analytic, Miller, imported standard-file, and differentiable VMEX geometry;
- Hermite-Laguerre velocity space;
- model, Sugama-family, and linearized Coulomb collisions;
- automatic differentiation, sensitivities, uncertainty workflows, and optimization;
- coupling to equilibrium, transport, and common gyrokinetic-analysis tools.

The following are outside GKX 3.x:

- global radial gyrokinetics;
- full-f evolution;
- magnetic-axis, separatrix, divertor, scrape-off-layer, wall, sheath, or open-end mirror physics;
- particle-in-cell algorithms;
- nonlinear full-f Coulomb evolution for distributions far from a Maxwellian;
- AMD GPU, TPU, and production multi-node support as release requirements.

A closed periodic VMEX mirror field line may supply a geometry contract to GKX. It is still a periodic local delta-f calculation. It is not an open-mirror confinement, end-loss, sheath, or Pastukhov-potential model.

### 2.2 Collision scope

For `f = F0 + delta f`, the Landau operator expands as

`C[f,f] = C[F0,F0] + C[F0,delta f] + C[delta f,F0] + C[delta f,delta f]`.

For a Maxwellian `F0`, the first term vanishes. Standard delta-f gyrokinetics retains the two cross terms as the linearized collision operator. The quadratic `C[delta f,delta f]` term is outside the small-perturbation ordering. Therefore:

- GKX shall implement scalable arbitrary-order **linearized** original Sugama, improved Sugama, and Coulomb operators;
- finite-perpendicular-wavelength and multispecies effects are required for the production collision program;
- the nonlinear Jorge-Frei-Ricci full operator is a literature and algorithm reference, not a GKX 3.x milestone;
- adding a nonlinear full-f collision operator requires an explicit scope change.

### 2.3 Product decisions

- Broad undocumented 1.x internal imports may be removed.
- A short, documented migration layer may remain for promoted user workflows, then be deleted on schedule.
- Equilibrium `E x B` shear is deferred to GKX 3.1 unless a later bounded readiness review reopens it.
- Rotation, parallel-flow shear, momentum flux, and momentum transport are 3.x work, not 3.0 blockers.
- The quasilinear model may be promoted within a clearly declared domain even when it is not uniformly accurate. It must publish limitations, uncertainty, and out-of-domain behavior.
- High-dimensional, multi-surface nonlinear stellarator optimization is the first major end-to-end research target.
- Native explicit integration is the current promoted time owner.
- A stiff implicit or IMEX path is retained only if it improves time-to-accuracy or enables a required case. GKX 3.0 will not ship an inferior method merely to satisfy an IMEX checklist.
- Duplicate native and Diffrax paths shall be removed after migration gates. Diffrax remains a temporary oracle until that decision is closed.
- TOML is the human-authored input format.
- NetCDF is the canonical result and restart format.
- JSON is the compact summary and machine-interchange format.
- External code comparisons are maintained locally. Raw outputs from GX, GENE, stella, GS2, CGYRO, or other projects are not committed or published by default.
- Permanent GKX tests must be self-contained and future-proof. External comparisons should be converted into analytic, mathematical, literature-anchored, or independent-algorithm tests.

### 2.4 Hard architecture targets

GKX 3.0 does not release until all targets below pass:

| Measure | Hard target |
| --- | ---: |
| Installable `src/gkx/**/*.py` files | at most 45 |
| Installable Python source lines | at most 45,000 |
| `tests/**/*.py` files | at most 30 |
| Test Python lines | at most 35,000 |
| Advertised top-level `gkx` names | at most 30 |
| Package statement coverage | at least 95% |
| Retained promoted-module branch coverage | at least 95%, unless a reviewed exception records why |
| Python floor | 3.11 |
| Required devices | CPU and NVIDIA GPU |
| Maintained developer commands | at most 12; target 8 |
| Python files under `tools/` | zero at final topology |

File-count targets must not be met by creating giant modules. Public facades should remain below 400 lines, ordinary modules below 900 lines, and the median module should remain below 500 lines. Reviewed schema or I/O owners may exceed a soft ceiling only when splitting would duplicate state or violate one-owner design.

---

## 3. Current repository state

### 3.1 Git and pull requests

At the audited revision:

- `main` is `e19336dc2202b721d12df4f27ab84835b1360de7`;
- it is 79 commits ahead of release 1.8.2 (`e89c7fed31657f32b638e653c7b266e33cded805`);
- the repository has one branch, `main`;
- there are no open pull requests;
- PR #162 is the latest merged change;
- the PR-head CI for #162 passed;
- the post-merge `main` workflow was cancelled, so the next housekeeping pull request must obtain a green post-merge `main` run before further architecture work is trusted.

The historical PR audit in `plan/pr_audit.md` covers PRs through #81 and records several inherited mega-merges, red-head merges, and later repairs. It is useful history but not a complete current ledger. PRs #82-#128 are represented by the frozen 1.8.2 baseline and Git history. PRs #129-#162 are summarized below. The replacement plan requires a compact machine-readable ledger for all historical PRs before obsolete planning files are deleted.

### 3.2 Current size and coverage

All counts below were regenerated at the audited revision `e19336dc` by PR H0-1 and are exact, not approximate. Non-`__pycache__` Python files, counted by line: `src/gkx` 199 files / 90,857 lines; `tests` 101 files / 87,725 lines; `tools` 90 files / 72,461 lines; `examples` 37 files / 4,749 lines; `benchmarks` 12 files / 1,673 lines; `docs` 33 reStructuredText files / 18,869 lines. `scripts/` holds no Python. The advertised root API is 14 names in `gkx.api.__all__`, and the lazy compatibility registry is exactly 352 `_EXPORT_TARGETS` entries.

The architecture manifest was corrected in the same pull request: the test targets of 36 files and 55,000 lines predated the approved contract in section 2.4 and now read 30 and 35,000, and four stale baselines were ratcheted to the measured tree so the gate no longer carries slack that merged deletions had already earned.

| Area | Current state | GKX 3 target | Gap |
| --- | ---: | ---: | ---: |
| Source files | 199 | <=45 | remove or merge at least 154 |
| Source lines | 90,857 | <=45,000 | remove at least 45,857 |
| Test files | 101 | <=30 | remove or merge at least 71 |
| Test lines | 87,725 | <=35,000 | remove 52,725 |

Composition of the test lines, measured 2026-08-30 at 85,393 lines over 100
files, because the shape of the gap decides the method:

| Category | Lines |
| --- | ---: |
| 2,385 top-level test functions | 69,527 (mean 29, median 22) |
| 311 helpers and classes | 5,481 |
| module level: imports, constants, docstrings | 10,485 |
| exact-duplicate function bodies | 157 |
| near-duplicates, same structure and differing constants | 205 |

Redundancy is 362 lines, 0.4 per cent. The suite is not bloated by copy-paste,
so consolidation and deduplication cannot close a 50,000-line gap: reaching
35,000 by deletion means removing about 1,700 of 2,385 tests, which is the
opposite of the detection-power constraint in section 17.

Where the lines actually are: `tests/` holds exactly one `conftest.py`, with
zero fixtures, while the tree makes roughly 1,500 repeated construction calls --
392 `SAlphaGeometry`/`from_config`, 287 zeroed-state inits, 235
`build_spectral_grid`, 227 `GridConfig`, 182 linear cache/terms builds, 176
`LinearParams`, 105 `CycloneBaseCase`. Every test rebuilds its world from
scratch. Extracting that setup into shared and factory fixtures touches no
assertion, so it is the one lever that reduces lines without reducing
detection. Whether it is a large enough lever was measured on a `tests/unit/linear` pilot
(PR C2-1) before any sweep. It is not.

Pilot result: 51 of 134 tests migrated onto six factory fixtures, 6.6 lines
saved per migrated test gross, 3.1 net of the 182-line fixture layer. The domain
went 4,485 -> 4,325 lines, a 3.6 per cent reduction, with the collected node-ID
set identical and 145 passed either way.

The pilot also measured *why* the lever is small, which matters more than the
rate. The roughly 1,500 repeated construction calls are overwhelmingly already
single physical lines: `jnp.zeros((` 279 single-line against 0 multi-line,
`SAlphaGeometry` 193 against 11, `build_spectral_grid` 196 against 50,
`GridConfig` 152 against 38, `LinearParams` 124 against 52. Replacing a one-line
construction with a one-line fixture call saves nothing. Savings come only from
collapsing multi-statement blocks and multi-line literals, and tree-wide only
about 33 such blocks remain, worth roughly 386 lines. A full fan-out lands
between 1,000 and 3,000 lines: 85,393 down to about 82,500-84,400.

**Conclusion, on evidence rather than judgement.** Deduplication offers 362
lines, fixture extraction offers 1,000-3,000. Together they supply 2 to 6 per
cent of the 50,393-line gap. There is no non-destructive route to 35,000 test
lines. Reaching it requires deleting roughly 1,700 of 2,385 tests, which is a
decision about which coverage to abandon and therefore belongs to the
maintainer, not to a refactor. Until that decision is made, the 35,000 figure
should not gate any pull request, and the manifest baseline should continue to
ratchet on measured reductions only.

The fixture layer is still worth landing on its own merits: tests written
against it are shorter and more uniform, which is a maintainability gain even
though it is not a line-count strategy.
| Tool files | 90 | 0 under `tools/` | relocate/delete 90 |
| Tool lines | 72,461 | 0 under `tools/` | relocate/delete 72,461 |
| Advertised root API | 14 | <=30 | nominally passes |
| Lazy compatibility registry | 352 targets | no hidden broad API | must be removed |
| Aggregate package coverage | 96.41% | >=95% | passes aggregate only |
| Modules below their coverage target | 50 of 181 tracked | 0 retained promoted modules | open |

The current architecture gate is a **no-regression gate against a legacy allowance**. A green result does not mean that the GKX 3 topology has been reached. PR H0-1 corrected the manifest's stale test targets and ratcheted four baselines to the measured tree; the remaining gap to the GKX 3 topology is real deletion, not accounting.

The current coverage artifact measures all 199 package modules and reports 96.41% aggregate coverage. Fifty tracked modules remain below their declared target. The lowest include:

- `gkx.objectives.vmec_boozer_context`: 34.92%;
- `gkx.geometry.vmec_boozer_constants`: 55.56%;
- `gkx.workflows.runtime.resolution`: 58.97%;
- nonlinear device-z reporting and execution: 72.22% and 77.33%;
- `gkx.api`: 80%;
- several geometry, saturation, objective, Diffrax, orchestration, and parallel modules between 85% and 95%.

This pattern is a deletion map as much as a testing map. Do not add tens of thousands of test lines to preserve low-value compatibility, report, or duplicated modules. Delete or merge an unnecessary module first; add concise behavioral coverage only for retained ownership.

### 3.3 Historical comparison with the attached 1.7.0 source

The attached older checkout had:

| Area | Files | Lines |
| --- | ---: | ---: |
| Source | 190 | 86,260 |
| Tests | 82 | 74,915 |
| Tools | 54 | 57,378 |
| Python examples | 38 | 4,931 |
| Documentation files | 1,416 | 33.9 MB |

The older source had docstrings on about 43.7% of function/class definitions. Its examples contained `argparse` in 28 files and a `__main__` guard in 33. The documentation tracked 301 PNG files, 790 JSON files, and 278 CSV files. Static generated evidence dominated the tree.

The current code has better interfaces, diagnostics, claims discipline, tests, provenance, and packaging, but its installable source is larger than the older tree and its test/tool surface has grown substantially. Modernization has so far improved behavior more than architecture. The remaining program must deliver real deletion and ownership consolidation.

### 3.4 Current product capabilities

| Capability | Current status | Required disposition |
| --- | --- | --- |
| Linear electrostatic solve | mature named pathways with time and Krylov modes | retain one public workflow and one numerical owner per algorithm |
| Nonlinear electrostatic solve | operational, restartable, diagnostics-rich | retain; simplify orchestration and prove statistical policies |
| Electromagnetic fields | implemented on tested lanes | complete systematic linear/nonlinear and kinetic-electron validation |
| Kinetic electrons | operational on selected cases | improve stiffness strategy only when it wins; broaden physics gates |
| Stellarator geometry | WOUT/EIK plus exact VMEX live/WOUT adapters | delete duplicated VMEC/Boozer reconstruction |
| Closed VMEX mirror geometry | adapter and bounded showcase | keep clearly scoped as periodic geometry only |
| Quasilinear model | diagnostics and candidate calibration machinery | choose, version, document, and promote the best bounded-domain model |
| Collisions | model operators and low-order/research linearized Sugama/Coulomb paths | build arbitrary-order matrix-free runtime operators |
| Linear differentiation | strong JVP/VJP/eigenvalue machinery | simplify API and preserve branch/conditioning gates |
| Nonlinear differentiation | finite-window checkpointed discrete adjoint | quantify statistical usefulness; do not call it an infinite-time gradient |
| Stellarator optimization | substantial reduced and campaign machinery | remove campaign governance from package; rebuild through generic API |
| Parallel execution | independent work and experimental state sharding | CPU/NVIDIA single-device first; keep only demonstrated useful routes |
| Packaging | PyPI wheel/sdist, Python 3.11, optional validation extra | reduce dependencies and add clean minimum/latest environment gates |
| Documentation | extensive but flat and internally focused | rewrite information architecture and user journey |
| Examples | many TOMLs and scripts, inconsistent hierarchy | replace with a small canonical gallery |

---

## 4. Work completed since release 1.8.2

### 4.1 Repairs and Phase-0 baseline

PRs #129-#138 repaired shipped benchmark imports and optional dependency failures, froze architecture/API/dependency/schema/provenance baselines, corrected profiler evidence, and closed most Phase-0 inventory work.

Important outcomes:

- pandas is now an explicit validation extra rather than an undeclared import requirement;
- the undeclared Rich branch in one progress path was removed;
- wheel and sdist installation were rechecked;
- numerical fingerprints, geometry ownership, integrator ownership, output schemas, external-comparison policy, and GX provenance were recorded;
- CPU and NVIDIA measurements were added for selected paths;
- the baseline established that the passing architecture gate was only a ratchet, not completion.

### 4.2 Product-surface compatibility layer

PRs #139-#147 added:

- public `Case` and result aliases;
- `gkx.load`, `gkx.solve`, `gkx.scan`, `gkx.plot`, and `gkx.prepare`;
- shorter `scan` and `plot` CLI aliases;
- version fields for TOML and NetCDF;
- an advertised 14-name top-level surface;
- dtype-aware tolerances.

These changes improve discoverability and provide migration entry points. They do **not** complete the GKX 3 API:

- `Case` and result types are aliases to existing runtime internals;
- `solve` accepts broad `**options` rather than a stable typed contract;
- `scan` is primarily a linear `k_y` scan, not a general parameter-scan API;
- `prepare` supports a narrow nonlinear diagnostics route and returns a weakly typed object;
- the lazy registry still maps roughly 350 historical names;
- low-level VMEC and solver-objective functions remain in the advertised root list.

Treat this wave as compatibility scaffolding. The next architecture phase must implement the actual types and delete the broad registry.

### 4.3 Core numerical consolidation

PRs #148-#155:

- consolidated cached linear RHS assembly;
- consolidated Hermitian projection;
- corrected timestep-independent terminal damping;
- made native linear integration the documented default;
- made the native step kernel the single owner of explicit and diagonal-IMEX step algebra;
- routed diagnostics-rich implicit sampling through the existing implicit owner;
- tested and rejected two additional stiff-integration candidates that failed prospective performance gates.

The rollback discipline was correct. ARS-style and Crank-Nicolson candidates did not beat the stable explicit route in representative kinetic-electron time-to-accuracy and memory. Their source was removed. Future stiff work must reuse and improve the existing coupled implicit machinery rather than add another solver family.

### 4.4 Geometry ownership and deletion

PRs #156-#162:

- added canonical live-state `from_vmex` geometry;
- added the closed periodic VMEX mirror adapter;
- added and regression-tested the canonical VMEX WOUT adapter;
- recorded that a true open mirror would require a separate full-f/open-boundary model;
- removed the Boozer-spectrum route that invented smooth metric and drift arrays;
- removed obsolete API exports and existence-only tests;
- preserved exact VMEX live/WOUT geometry, generic mapping validation, and Boozer diagnostics.

PR #159 was a stacked duplicate of the WOUT adapter and closed unmerged. PR #160 rebased the same work onto `main` and merged. PR #162 removed 648 installable source lines and 186 test lines. This was the strongest deletion in the current wave, but it leaves many duplicate VMEC/Boozer modules and campaign objectives.

### 4.5 Provenance

Root `PROVENANCE.md` and `plan/baseline/gkx_1_8_2_gx_provenance.md` now record the conservative descendant boundary of the original GX-derived geometry port, the comparison revision, source hashes, and license notice. This is a good foundation.

Remaining provenance work:

- add short provenance notes to directly derived public or nontrivial retained functions;
- record exact upstream revision and path for every future translation;
- remove stale provenance rows when descendant code is deleted;
- distinguish software derivation from equations implemented from papers.

---

## 5. Review of source changed in the current wave

The table below covers every installable source area changed between 1.8.2 and the audited head. “Keep” means preserve the behavior, not necessarily the file.

| Changed path or family | Review | Required action |
| --- | --- | --- |
| `api/__init__.py` | advertised surface reduced, registry still broad | replace registry with a real small API; migration map outside runtime |
| `artifacts/io.py` | stable behavior, oversized mixed serializer | split only by schema ownership; converge into `gkx.io` |
| `artifacts/nonlinear_netcdf.py` | valuable canonical restart/output contract | retain one schema owner; remove duplicated serializers |
| `artifacts/plotting.py` | `gkx.plot` improved access; global style and many plot modes remain | replace with result-oriented plotting and local style contexts |
| `artifacts/spectral_layout.py` | useful shared spectral interpretation | merge with numerical spectral-layout owner |
| `cli.py` | aliases added, still a large argparse switchboard | reduce to six user commands and delegate to typed API |
| `diagnostics/zonal_validation.py` | optional dependency repaired | retain numerical metrics; move dataframe/report code out of core |
| `geometry/__init__.py` | canonical VMEX adapters exposed | retain a compact geometry facade |
| `geometry/booz_xform_bridge.py` | synthetic closure removed | delete the remaining bridge if only diagnostics remain; Boozer transform belongs upstream |
| `geometry/differentiable.py` | obsolete exports removed | retain only generic mapping and sensitivity contracts |
| `geometry/vmec_state_sensitivity.py` | large deletion, some compatibility remains | delete after exact VMEX adapter parity and downstream migration |
| `geometry/vmec_tensor_mapping.py` | now delegates more exact VMEX ownership | reduce to a thin adapter or remove |
| `operators/linear/moments.py` | duplicate moment logic reduced | merge into one velocity/field-moment owner |
| `operators/linear/rhs.py` | duplicate dispatch reduced | keep a small composition owner, not a second physics implementation |
| `operators/nonlinear/brackets.py` | fusion/performance cleanup | retain one mathematically specified bracket kernel |
| `operators/nonlinear/projection.py` | Hermitian projection consolidated | move next to spectral-grid ownership |
| `operators/nonlinear/rhs.py` | duplicate term assembly reduced | converge linear/nonlinear term ownership without copied field logic |
| `parallel/integrators.py` | small compatibility changes | retain only demonstrated independent-work and supported sharding routes |
| `parallel/velocity_drive.py` | duplicated drive logic reduced | merge global-index Hermite coefficients with the core term owner |
| `runtime.py` | convenience API added; remains high-fan-out facade | replace patchable globals and `Any` with typed orchestration |
| `solvers/linear/implicit.py` | small cleanup | retain one coupled implicit owner if useful |
| `solvers/linear/integrator_diagnostics.py` | shared step ownership improved | diagnostics must observe a solver, not implement one |
| `solvers/linear/integrators.py` | explicit algebra consolidated | keep as one public linear integration owner or merge into `solve/linear.py` |
| `solvers/linear/krylov_algorithms.py` | tolerance and algorithm fixes | keep concise matrix-free algorithms and residual certificates |
| `solvers/linear/parallel*.py` | duplicate calls reduced | merge supported slices; remove experimental variants without a product use |
| `solvers/nonlinear/diagnostics.py` | duplicated diagnostics removed | complete separation of diagnostics from time stepping |
| `solvers/time/explicit.py` | native owner retained | merge configuration and stepper policy into one explicit module |
| `solvers/time/explicit_diagnostics.py` | Rich branch and duplicate steps removed | eventually delete as a separate integrator |
| `solvers/time/explicit_steps.py` | now owns native step maps | retain as numerical kernel or merge into time module |
| `terms/__init__.py` | broad exports reduced | delete compatibility facade after migration |
| `terms/assembly.py` | centralization improved | define one immutable operator/cache assembly contract |
| `terms/fields.py` | fields consolidated | retain one field-equation owner with exact normalization tests |
| `workflows/linear.py` | small compatibility update | merge into typed `solve.linear` orchestration |
| `workflows/nonlinear.py` | prepared execution improved | merge into typed `solve.nonlinear`; no workflow-specific physics |
| `workflows/runtime/commands.py` | CLI aliases wired | delete after CLI delegates directly to public API |
| `workflows/runtime/config.py` | schema fields added | replace aliases with immutable public case models |
| `workflows/runtime/execution.py` | prepared route updates | merge with `PreparedSimulation` |
| `workflows/runtime/results.py` | aliases/schema updates | replace with actual public result types |
| `workflows/runtime/toml.py` | versioning and shorthand changes | retain one schema loader/migrator |
| `workflows/runtime/wout.py` | canonical adapter wiring | reduce to public geometry integration, no duplicated VMEC physics |

The current wave also added hundreds of test lines around compatibility aliases and retained tools. Future changes should prefer deletion plus one broader behavioral test over one test per wrapper.

---

## 6. Main gaps that still block GKX 3

### 6.1 Architecture

- 199 installable modules remain.
- Import ownership is still layered as `runtime -> workflows -> solvers -> operators -> terms`, with parallel and artifacts crossing those layers.
- The old checkout had at least three import cycles; the current tree must regenerate the graph and reduce it to zero.
- Large facades and report modules hide duplicate policy.
- Research-campaign admission and claim logic still appears in installable diagnostics/objective areas.
- `tools/` remains a second software project larger than the intended final core.

### 6.2 Public API

- The small advertised API is backed by historical aliases rather than final types.
- Root exports mix user actions with low-level solver-objective internals.
- No stable generic scan protocol exists.
- No typed prepared linear/nonlinear simulation object exists.
- Result types do not yet own all save, plot, inspect, convergence, and dataset behavior.
- Extension points for geometry, collisions, objectives, diagnostics, and transport coupling are not yet small and formal.

### 6.3 Numerical ownership

- Native explicit ownership is strong.
- Diffrax remains installed and tested as a second route.
- The useful role of the existing implicit route is not yet a clean product contract.
- Diagnostics, sampling, progress, restart, and integration remain partially duplicated.
- Experimental parallel pathways carry significant source and test cost without a production release requirement.

### 6.4 Scientific validation

- Aggregate coverage is high, but many retained modules miss their own target.
- Artifact and report tests are still mixed with solver-backed scientific evidence.
- Several physics lanes are validated only on named low-dimensional cases.
- Kinetic-electron, electromagnetic, and stellarator combinations need a coherent matrix rather than isolated checks.
- Nonlinear stopping and uncertainty policies require prospective, held-out validation.
- Finite-window nonlinear derivatives need ensemble and directional-prediction evidence, not only same-trajectory AD/finite-difference agreement.

### 6.5 User experience

- README is too long and carries multiple showcases, detailed media provenance, runtime transcripts, benchmark tables, and internal caveats before a concise first workflow.
- Documentation is a flat list that mixes tutorials, equations, API, internal architecture, release policy, manuscript figures, and research planning.
- Examples use inconsistent directory names and many `runtime_` prefixes.
- Python examples often depend on low-level internals and helper functions rather than the advertised API.
- Plotting lacks one stable visual and data contract.

### 6.6 Physics development

- Arbitrary-order production collision operators are not complete.
- The best quasilinear model has not been selected and made user-facing.
- The flagship high-dimensional nonlinear optimization has not been closed with held-out multi-surface evidence.
- Transport coupling and Pyrokinetics interoperability remain incomplete.
- Momentum transport and equilibrium shear remain future 3.x work.


---

## 7. Target product design

### 7.1 User model

A user should learn one model:

```python
import gkx

case = gkx.load("case.toml")
result = gkx.solve(case)
result.print_summary()
result.save("outputs/case.nc")
result.plot("outputs/figures")
```

A repeated or differentiated workflow should use one prepared object:

```python
simulation = gkx.prepare(case)
result = simulation.solve()
scan = simulation.scan("species[0].tprim", [2.0, 2.5, 3.0])
value, gradient = simulation.value_and_grad(objective, parameters)
```

The API should not require users to know runtime command dependencies, cache builders, term-conversion functions, report builders, or solver-private arrays.

### 7.2 Proposed top-level API

The final root surface should contain approximately 18 names:

```text
Case
Species
Grid
Geometry
Physics
Time
Output
LinearResult
NonlinearResult
ScanResult
PreparedSimulation
load
solve
scan
prepare
plot
inspect
validate
```

Advanced names belong in explicit subpackages:

- `gkx.geometry`: analytic, Miller, imported file, VMEX state/WOUT adapters;
- `gkx.physics`: collision and closure specifications;
- `gkx.optimize`: objectives, derivative policies, portfolios;
- `gkx.io`: schema and conversion functions;
- `gkx.integrations`: Pyrokinetics and transport adapters.

Do not top-level-export low-level RHS kernels, cache classes, validation reports, VMEX campaign functions, or private linear algebra.

### 7.3 Public types

#### `Case`

`Case` should be an immutable, validated PyTree-compatible dataclass. It owns:

- schema version;
- species;
- geometry request or in-memory geometry;
- grid;
- physics switches;
- collision and closure selection;
- time and solver policy;
- initialization;
- diagnostics;
- output policy;
- normalization.

Required methods:

```python
case.replace(...)
case.validate()
case.to_toml(path)
case.summary()
```

No case field may be interpreted differently by the CLI and Python API. Parsing aliases and migration belong in `gkx.io.config`, not in kernels.

#### `PreparedSimulation`

This is the compiled reusable execution object. It owns:

- validated static topology and array shapes;
- geometry arrays;
- operator/cache assembly;
- compiled value and gradient callables;
- persistent-cache metadata;
- device, precision, and sharding plan;
- memory and compilation estimates.

Required methods:

```python
solve(parameters=None, initial_state=None)
scan(parameter, values, *, parallel="auto")
value_and_grad(objective, parameters)
warmup()
estimate_memory()
summary()
```

`prepare` must support promoted linear and nonlinear cases. A prepared object must not silently change physics because a diagnostic, collision model, or integrator is selected.

#### Results

`LinearResult`, `NonlinearResult`, and `ScanResult` should be real stable classes, not aliases. They own:

- typed scalar diagnostics;
- array data and coordinates;
- convergence and resolution status;
- warning records;
- normalization;
- complete provenance;
- `save`, `plot`, `print_summary`, and `to_dataset` behavior.

A result should never describe an unsaturated or unresolved value as accepted. Rejected windows remain available with an explicit status and reason.

### 7.4 Extension protocols

Keep extension points small and structural:

```python
class GeometryProvider(Protocol):
    def build(self, request) -> FluxTubeGeometry: ...

class CollisionOperator(Protocol):
    def apply(self, state, context) -> Array: ...
    def invariants(self) -> CollisionInvariants: ...

class Objective(Protocol):
    def __call__(self, result_or_state, context) -> Array: ...

class Diagnostic(Protocol):
    def sample(self, state, fields, time, context) -> dict[str, Array]: ...
```

Protocols must not expose repository-specific report schemas. The public contracts should allow new physics or coupling modules without editing central dispatch tables.

---

## 8. Target source architecture

The following is a line and ownership budget, not a demand to create empty packages.

```text
src/gkx/
  __init__.py
  api.py
  case.py
  result.py
  cli.py
  _version.py

  physics/
    species.py
    equations.py
    fields.py
    collisions.py
    closures.py
    transport.py

  geometry/
    core.py
    analytic.py
    miller.py
    imported.py
    vmex.py

  numerics/
    grid.py
    velocity.py
    spectral.py
    operators.py
    explicit.py
    implicit.py
    eigensolver.py
    parallel.py

  solve/
    linear.py
    nonlinear.py
    prepared.py
    diagnostics.py
    convergence.py
    objectives.py

  optimize/
    derivatives.py
    stellarator.py
    portfolio.py

  io/
    config.py
    netcdf.py
    plotting.py
    provenance.py

  integrations/
    pyrokinetics.py
    transport.py
```

This layout contains fewer than 45 files. It may be adjusted when measured ownership demands it, but all alternatives must satisfy:

- zero import cycles;
- one owner for each equation and numerical algorithm;
- no `runtime`, `workflows`, `artifacts`, or `terms` package in the final topology;
- no source modules whose main purpose is building manuscript, release, or campaign reports;
- no private wrapper that forwards unchanged arguments to another module;
- no duplicated linear/nonlinear field equations;
- no separate “diagnostics integrator” implementing a second timestepper;
- no tool or test imports required by the installed package.

### 8.1 Deletion and merge map

| Current area | Target owner | Disposition |
| --- | --- | --- |
| `gkx.api` registry | root `api.py` | replace; keep a versioned migration table outside `__all__` |
| `gkx.artifacts` | `gkx.io` | merge schemas/plots; delete artifact governance |
| `gkx.benchmarking` | `benchmarks/` or tests | remove from package unless a reusable analysis function exists |
| `gkx.core` | `case`, `physics`, `numerics` | split by scientific ownership |
| `gkx.diagnostics` | `solve/diagnostics.py`, `solve/convergence.py`, `physics/transport.py` | remove reports and campaign gates |
| `gkx.geometry` | compact geometry package | retain generic contracts and adapters; delete duplicated equilibrium algebra |
| `gkx.objectives` | `solve/objectives.py`, `optimize/` | retain reusable objectives; remove campaign admission/report modules |
| `gkx.operators` + `gkx.terms` | `physics` + `numerics/operators.py` | one equation owner and one assembled operator owner |
| `gkx.parallel` | `numerics/parallel.py` | keep only supported independent work or measured sharding |
| `gkx.solvers` | `numerics` + `solve` | separate algorithms from orchestration |
| `gkx.workflows` + `runtime.py` | `solve`, `io`, `cli` | remove after public API migration |
| `tools/artifacts` | release workflow or external reproducibility package | delete most; retain at most one figure-regeneration command |
| `tools/campaigns` | ignored local workspace or separate research repository | remove from installed repo |
| `tools/comparison` | local-only scripts outside main or compact `scripts/compare.py` | no raw outputs in Git |
| `tools/profiling` | `scripts/profile.py` | one configurable profiler |
| `tools/release` | `scripts/check.py` plus CI | consolidate all release checks |

### 8.2 Architecture proof gates

Every architecture PR must report:

- source/test/tool file and line deltas;
- import graph and cycle count;
- top-level API count;
- largest ten modules and functions;
- duplicate normalized AST/function groups;
- package import time;
- wheel and sdist size;
- exact public behavior affected;
- numerical fingerprints before and after.

A file move without a net reduction in files, lines, cycles, public surface, or duplicated ownership does not count as progress.

---

## 9. Geometry ownership

### 9.1 Final boundary

VMEX owns:

- equilibrium state and solve;
- VMEC spectral geometry;
- live-state and standard-WOUT field-line evaluation;
- metric tensors, drifts, pressure/current contributions, and equal-arc mapping;
- closed mirror field-line construction.

`booz_xform_jax` owns:

- VMEC-to-Boozer transformation;
- Boozer Fourier spectra;
- `boozmn` I/O;
- Boozer-coordinate diagnostics.

GKX owns:

- the generic solver-ready flux-tube contract;
- normalization and finite-value checks;
- parallel-domain topology and twist/shift policy;
- interpolation onto the GKX parallel grid when necessary;
- consumption of geometry by gyrokinetic equations;
- thin adapters to VMEX state, VMEX WOUT, imported EIK, Miller, and analytic models.

Boozer coordinates are not required by the turbulence solver when a complete straight-field-line metric/drift mapping is already available. Optimization may use Boozer spectra as separate constraints.

### 9.2 Required deletion sequence

1. Freeze exact live-state versus WOUT versus retained imported-EIK parity on:
   - axisymmetric vacuum;
   - shaped finite-beta tokamak;
   - QA/QH/QI stellarators;
   - `LASYM=true` case;
   - closed periodic mirror.

   Blocker measured 2026-08-30: the mirror class cannot be frozen anywhere
   currently reachable. `gkx.geometry.from_vmex_mirror` needs
   `vmex.mirror.turbulence.gk_closed_fieldline_geometry`. The laptop has VMEX
   0.6.0, which has `vmex.mirror` but not that module; the office box (two
   RTX A4000, `~/stellarator_venv`, jax 0.6.2) has no `vmex.mirror` at all. The
   other four classes are unblocked. The same gap blocks regenerating the mirror
   README asset, so both wait on a VMEX version that is installed on neither
   machine.
2. Freeze AD/finite-difference geometry gradients for the live VMEX path.
3. Search downstream imports in GKX, VMEX, examples, and local scripts.
4. Redirect users to `from_vmex`, `from_vmex_wout`, or a generic complete mapping.
5. Delete remaining in-package VMEC/Boozer reconstruction modules and tests.
6. Retain the GX-derived imported-EIK/Miller path only if it provides a unique user capability. Otherwise delegate or reduce it to one compatibility adapter.
7. Update provenance and documentation in the same PR.

Candidate deletions include the remaining `vmec_boozer_*`, state-control, state-sensitivity, tensor/report, and duplicated field-line sampling modules. Do not delete solely by filename; inspect whether a unique standard-file or local-equilibrium capability remains.

### 9.3 Geometry proof tests

- analytic circular and Miller identities;
- metric positivity and determinant identities;
- `B . grad psi = 0` within discretization tolerance;
- field-line straightness and periodic/twist endpoint conditions;
- grad-B versus curvature drift pressure correction;
- finite-beta and current-dependent terms;
- equal-arc constant parallel derivative;
- parity under stellarator symmetry and `LASYM` support;
- live VMEX/WOUT equivalence;
- AD/finite-difference Taylor tests;
- resolution convergence of radial derivatives;
- closed-mirror periodicity and mirror-force consistency.

---

## 10. Equations, fields, and operator ownership

### 10.1 One model statement

Write the normalized delta-f electromagnetic gyrokinetic system once in `physics/equations.py` and once in the documentation. Every implemented term must map to:

- equation number;
- code owner;
- input switch;
- normalization;
- discrete representation;
- conservation/free-energy role;
- tests;
- reference.

The code should expose a term inventory generated from source metadata rather than manually maintained duplicate tables.

### 10.2 Field equations

`physics/fields.py` must be the only owner of quasineutrality, parallel Ampere, and perpendicular magnetic-field equations. Linear and nonlinear solvers call the same field owner. Required tests:

- electrostatic and electromagnetic limiting cases;
- adiabatic-electron `k_y=0` response;
- gauge and Hermitian reality contracts;
- species summation and normalization;
- dense versus matrix-free parity on small systems;
- residual certificates;
- JVP/VJP duality;
- float32 and float64 tolerances.

### 10.3 Linear and nonlinear operators

Use one immutable operator context assembled from `Case`, grid, geometry, species, and collision policy. The linear RHS and nonlinear RHS may be separate callables, but shared terms must not be copied.

The nonlinear bracket owner must prove:

- antisymmetry;
- zero bracket for constant fields;
- discrete particle conservation where applicable;
- discrete free-energy exchange without artificial production in the dissipation-free limit;
- exact Hermitian closure after dealiasing;
- compressed-real and full-complex parity;
- serial and supported sharded identity;
- observed spectral convergence on smooth manufactured fields.

---

## 11. Time integration and solver strategy

### 11.1 Promoted explicit path

Retain native RK2/RK3/RK4 with one step owner and one diagnostics observer. Required product behavior:

- fixed and adaptive timestep policies use the same RHS;
- CFL attribution is diagnostic and clearly labeled when heuristic;
- terminal times and saved states match exactly;
- restart continuation is equivalent to an uninterrupted run;
- diagnostics striding does not retain full device histories;
- reverse-mode finite-window differentiation follows the executed discrete map.

### 11.2 Stiff path decision

Do not implement a third new IMEX family. Evaluate the existing coupled implicit owner against the explicit path on the cases that motivate stiffness:

- kinetic-electron ITG/TEM;
- long-wavelength electrostatic response;
- electromagnetic kinetic-electron mode;
- high collision frequency with the production collision operator.

For each case compare:

- physical time reached per wall second;
- error relative to a converged reference;
- peak device and host memory;
- compile time and compiled executable count;
- solver residual and failure rate;
- gradient support and cost.

Decision gate:

- **promote** the coupled implicit path if it wins time-to-accuracy or enables a stable required case;
- **retain as expert optional** if it is scientifically useful but not generally faster;
- **remove from the base product** if it has no supported use after collision development;
- **remove Diffrax** once its migration-oracle role is complete and no unique promoted capability remains.

An IMEX label is not a goal. A smaller, faster explicit code is preferable when it solves the approved scope reliably.

### 11.3 Linear eigensolvers

Retain matrix-free dominant-branch methods with:

- residual certificates;
- dense parity on bounded cases;
- branch-continuation metrics;
- spectral-gap and near-degeneracy warnings;
- precision-aware tolerances;
- JVP/VJP and finite-difference agreement;
- cost independent of design dimension for reverse sensitivities.

Do not expose three synonymous eigensolver entry points.

---

## 12. Collision and closure program

### 12.1 Unified collision API

All collision models should share one selector, one normalization, one metadata record, and one operator protocol. A result must record:

- operator family and version;
- species-pair ordering;
- retained moments;
- finite-perpendicular-wavelength treatment;
- collision frequencies and normalization;
- implicit/explicit application;
- conservation and dissipation status.

### 12.2 Ordered milestones

#### C0: semantics and preflight

- unify `none`, model, original Sugama, improved Sugama, and Coulomb selection;
- fail before compilation for unsupported species/order/wavelength/integrator combinations;
- remove ambiguous “full Coulomb” language;
- state “linearized,” species scope, and retained-order scope everywhere.

#### C1: arbitrary-order drift-kinetic operators

- generate original Sugama, improved Sugama, and linearized Coulomb at arbitrary retained `(N_l, N_m)`;
- support general species pairs and mass/temperature ratios;
- avoid checked-in order-specific dense tables except tiny independent fixtures;
- cache static coefficient structures by physics signature.

#### C2: finite-`k_perp`, like-species runtime

- implement test-particle, field-particle, and polarization pieces;
- avoid materializing a dense moment matrix at every spatial point;
- use matrix-free, sparse, separable, or low-rank contractions;
- preserve exact invariant corrections;
- integrate through the selected stiff/explicit policy.

#### C3: finite-`k_perp`, multispecies runtime

- support ordered target/source pairs with distinct masses, temperatures, charges, and Larmor radii;
- conserve total particle number per species and total momentum/energy across pairs;
- validate electron-ion, ion-electron, impurity, and equal-species limits;
- batch species pairs without recompilation growth.

### 12.3 Collision proof matrix

Mathematical:

- density nullspace;
- total momentum and energy conservation;
- self-adjointness where the operator should have it;
- negative semidefiniteness and discrete H theorem;
- symmetry/reciprocity of species-pair coefficients;
- exact drift-kinetic limit as `k_perp rho -> 0`;
- expected gyro-diffusive finite-`k_perp` scaling;
- original/improved Sugama limiting relations.

Numerical and physical:

- coefficient recurrence and transform conditioning;
- arbitrary-order convergence;
- Spitzer-Harm conductivity through the actual runtime operator;
- multispecies temperature and momentum relaxation;
- zonal-flow residual and collisional damping;
- short-wavelength ITG stabilization by collisional FLR terms;
- TEM comparison among original Sugama, improved Sugama, and Coulomb;
- nonlinear heat-flux sensitivity to operator choice;
- CPU/GPU runtime and memory versus moment count.

### 12.4 Closures

Compare hard truncation, hypercollisions, outgoing Hermite-flux/reflectionless closure, and collision-based asymptotic closure. Promotion requires:

- reflected Hermite-flux measurement;
- recurrence-time scaling;
- linear dispersion error;
- nonlinear transport convergence;
- moment-tail decay;
- conservation and free-energy behavior;
- cost and robustness across collisionality.

Learned closures remain experimental until they preserve invariants, report uncertainty, and pass untouched out-of-domain cases.

---

## 13. Quasilinear model

### 13.1 Product tiers

- **Q1 ranking:** predicts ordering and useful optimization directions.
- **Q2 calibrated flux:** predicts quantitative flux within a declared domain and uncertainty band.
- **Q3 transport-coupled:** remains stable and useful inside profile evolution or steady-state transport iteration.

GKX should ship the highest tier that passes prospective gates. A model may be Q2 for electrostatic ion-scale ITG in specified tokamak and stellarator domains while remaining Q1 or unsupported for electromagnetic, electron-scale, pedestal, high-flow-shear, or other cases.

### 13.2 Candidate ingredients

Evaluate candidates built from:

- physical linear heat and particle flux weights;
- multi-`k_y` mode spectra and branch continuity;
- geometry-aware effective `k_perp`;
- growth rate and real frequency;
- eigenfunction structure;
- zonal-flow response or secondary-instability information;
- collisionality and species features;
- uncertainty and out-of-domain scores.

Do not assume that one scalar mixing-length constant is universal.

### 13.3 Model-selection protocol

1. Define the physical domain before fitting.
2. Split by equilibrium family, not random rows, so holdouts test transfer.
3. Keep stress cases visible.
4. Compare simple baselines before complex regression.
5. Score:
   - Spearman/rank performance;
   - signed and absolute flux error;
   - calibration and interval coverage;
   - monotonicity/limit consistency;
   - resolution robustness;
   - optimization-direction agreement;
   - transport-iteration stability;
   - cost.
6. Freeze a model card and coefficients with a versioned schema.
7. Reject or lower confidence outside the trained domain.
8. Expose the model through TOML, Python, NetCDF metadata, and result plotting.

The user-facing result must state whether it is a ranking proxy, a calibrated flux, or an out-of-domain estimate.

---

## 14. Differentiation and optimization

### 14.1 Linear derivatives

Preserve:

- implicit eigenvalue/eigenvector differentiation;
- matrix-free reverse mode;
- branch and conditioning diagnostics;
- derivatives through VMEX geometry where supported;
- parameter-count scaling measurements.

Every promoted derivative requires Taylor-remainder tests, JVP/VJP duality, finite-difference step ladders, and near-degenerate failure behavior.

### 14.2 Nonlinear finite-window derivative

The current method computes the exact reverse derivative of a finite discrete trajectory at a detached saturated initial state. It is not the derivative of the infinite-time invariant turbulent measure.

Required evidence before use as a production optimization direction:

- finite-difference step ladders, not one step size;
- multiple independently saturated initial states;
- separated time windows;
- gradient covariance and pairwise direction cosine;
- dependence on window length and autocorrelation time;
- held-out finite perturbations along predicted directions;
- comparison with SPSA and ensemble finite differences at matched GPU cost;
- successful line search or trust-region decrease;
- final long replicated transport audit.

Investigate shadowing, ensemble, or response methods only as bounded research branches with prospective rollback gates. Do not add a permanent algorithm family until it improves estimation of the stationary objective.

### 14.3 Flagship stellarator workflow

The target campaign must include:

- 50-200 independent geometry controls;
- multiple flux surfaces;
- multiple field-line labels;
- multiple `k_y` values;
- aspect ratio, iota, equilibrium quality, and QA/QH/QI constraints as appropriate;
- declared training and held-out surfaces/field lines;
- uncertainty-aware short-window gradients;
- comparison with SPSA and finite differences at matched compute budget;
- long nonlinear baseline/candidate ensembles;
- resolution, timestep, moment, spectral-tail, and stationarity gates;
- one local independent-code audit where practical, converted into self-contained GKX regression/proof evidence.

Campaign admission and report generation belong outside the installable core. The core should expose reusable objectives, sampling portfolios, statistics, and result schemas.

---

## 15. Coupling and interoperability

### 15.1 VMEX

- VMEX owns equilibrium and exact field-line geometry.
- GKX consumes `from_vmex` and `from_vmex_wout` mappings.
- Optimization composes both packages in memory.
- Do not duplicate equilibrium, Boozer, or drift geometry in GKX.

### 15.2 Pyrokinetics

Add a maintained GKX plugin for:

- reading/writing GKX TOML;
- converting normalized species and local geometry;
- reading linear and nonlinear NetCDF results;
- exposing eigenvalues, eigenfunctions, fields, fluxes, spectra, and time coordinates;
- round-trip tests without depending on another code executable.

Pyrokinetics already standardizes several gyrokinetic input/output formats and normalizations. GKX adoption will be easier if users can compare and analyze it through that ecosystem.

### 15.3 Transport coupling

Define a small transport adapter that takes radial profiles and returns:

- particle, heat, and momentum fluxes when supported;
- uncertainty;
- convergence and domain status;
- gradients/Jacobians when trustworthy;
- cache and warm-start metadata.

Support independent radial batches before adding complicated coupling. Test a mock transport loop for stability and schema behavior. Real TGYRO/T3D/Trinity-style coupling remains an integration layer, not a dependency of the core package.

### 15.4 Local external comparison policy

A local comparator run must record:

- repository and commit;
- compiler/build flags;
- hardware and precision;
- complete input and normalization map;
- resolution and timestep;
- residual/convergence status;
- postprocessing version;
- quantitative comparison and uncertainty.

Raw external output stays outside Git. The permanent GKX test should use an analytic result, a published scalar/curve with provenance, an independently implemented formula, a manufactured solution, or a compact derived fixture whose meaning does not depend on future external-code behavior.

---

## 16. JAX and performance contract

### 16.1 Compilation and topology

- JIT the outer simulation step/scan, not hundreds of tiny wrappers.
- Keep array topology and shapes static inside compiled loops.
- Build functions once; do not create new jitted callables inside iterations.
- Use a persistent compilation cache for CLI and repeated design workflows.
- Include JAX/jaxlib version, device type/count/topology, XLA flags, precision, and static signature in performance records.
- Explain cache misses during development and gate unintended recompilation.

### 16.2 Memory

- Keep diagnostics strided and host-resident where possible.
- Do not materialize complete state histories unless explicitly requested.
- Use buffer donation only when lifetime tests show that the input is not reused.
- Measure device peak memory separately from Python host allocations.
- Track live memory after synchronization and retained cache memory.
- Fail before execution when a predictable state/table allocation exceeds a configured device budget.

### 16.3 Array layout and kernels

- Choose layout from profiles, not aesthetic preferences.
- Keep transform axes contiguous where practical.
- Avoid degenerate general matrix products for single-digit moment contractions.
- Fuse elementwise physics when it reduces memory traffic, but add explicit barriers when fusion creates strided rereads.
- Pin precision on invariant-carrying contractions that must not lower to TF32.
- Preserve bitwise or tolerance-bounded identities around every kernel rewrite.

### 16.4 Parallel policy

Production 3.0 requires:

- serial CPU;
- single NVIDIA GPU;
- independent `k_y`, parameter, seed, surface, and field-line batches where useful.

State sharding may remain experimental unless it demonstrates:

- numerical identity;
- lower time-to-solution or memory extension on realistic problems;
- bounded communication;
- differentiation support where claimed;
- a simple user contract.

Do not keep five parallel routes for possible future hardware.

### 16.5 Benchmark method

Every benchmark separates:

- import and setup;
- host-to-device transfer;
- first compile plus execution;
- warm execution after `block_until_ready()`;
- output transfer and serialization;
- end-to-end user time;
- peak host and device memory.

Comparisons use matched precision and matched scientific error. Record medians and spread after warmup. Microbenchmarks supplement, but do not replace, representative end-to-end cases.


---

## 17. Test and verification design

### 17.1 Interpretation of “proof tests”

A finite test suite cannot formally prove every physical claim. GKX shall use the strongest available evidence for each layer:

- exact algebraic identities and invariants where a mathematical proof can be encoded;
- analytic and manufactured solutions for code verification;
- observed-order tests for discretizations;
- independent implementations for delicate formulas;
- literature-anchored benchmarks for model behavior;
- statistically designed ensembles for turbulent quantities;
- regression tests only after the underlying result is independently justified.

A frozen output without a mathematical, physical, or numerical reason is weak evidence and should not dominate the suite.

### 17.2 Evidence levels

| Level | Evidence | Examples |
| --- | --- | --- |
| E0 | API/schema contract | types, shapes, errors, TOML and NetCDF round trips |
| E1 | exact mathematics | recurrence identities, symmetry, nullspaces, conservation |
| E2 | numerical verification | manufactured solutions, order, conditioning, restart equivalence |
| E3 | analytic physics | Landau damping, dispersion relations, zonal residuals, conductivity |
| E4 | literature benchmark | Cyclone, W7-X/HSX, TEM/ETG/KBM, published collision curves |
| E5 | independent local code comparison | GX, GENE, stella, GS2, CGYRO; diagnostic only |
| E6 | nonlinear statistical evidence | stationarity, autocorrelation, seeds, resolution and timestep |
| E7 | performance evidence | synchronized runtime, memory, compilation, scaling at matched error |

Every promoted feature must have at least one E1/E2 test and the appropriate E3-E7 tests. Code coverage alone is E0.

### 17.3 Target test topology

Target 24-28 Python files, never more than 30:

```text
tests/
  conftest.py
  test_api_schema.py
  test_io_restart.py
  test_geometry_analytic.py
  test_geometry_vmex.py
  test_velocity_basis.py
  test_fields.py
  test_linear_operator.py
  test_nonlinear_operator.py
  test_collisions.py
  test_closures.py
  test_integrators.py
  test_eigensolvers.py
  test_linear_solve.py
  test_nonlinear_solve.py
  test_diagnostics_statistics.py
  test_autodiff.py
  test_parallel.py
  test_cli_examples.py

  physics/
    test_streaming_landau.py
    test_zonal_gam.py
    test_itg_tem_etg.py
    test_electromagnetic.py
    test_stellarator.py
    test_collision_physics.py
    test_nonlinear_transport.py
    test_quasilinear.py
    test_optimization.py

  test_release.py
```

Combine related cases through parameterization and fixtures. Do not create one test file per script, report, artifact, or bug.

### 17.4 Mathematical proof-oriented tests

#### Basis and transforms

- Hermite and Laguerre orthogonality under the implemented quadrature;
- recurrence relations and normalization;
- forward/inverse transform identity;
- parity and reality conditions;
- truncation-boundary terms;
- Parseval/free-energy consistency;
- condition numbers over promoted orders.

#### Spectral grid and bracket

- Fourier derivative on exact modes;
- dealiasing support and zeroing policy;
- Hermitian projection idempotence;
- twist-and-shift index and continuous-phase identities;
- bracket antisymmetry and constant-field nullspace;
- discrete conservation/free-energy exchange;
- compressed-real/full-complex parity.

#### Fields and operators

- quasineutrality and Ampere residuals;
- species-sum and normalization identities;
- electrostatic/electromagnetic limits;
- collisionless and zero-drive limits;
- term-by-term linear operator parity against independent small dense assembly;
- adjoint identities `<u, Lv> = <L^*u, v>` where applicable.

#### Collisions

Use the matrix and physics tests listed in Section 12. These are release blockers, not optional slow checks.

### 17.5 Numerical verification

Use exact and manufactured solutions to verify:

- temporal order of RK2/RK3/RK4 and any retained implicit method;
- parallel streaming and mirror coupling;
- drift and diamagnetic terms;
- source/damping terms;
- nonlinear bracket in multiple dimensions;
- geometry interpolation and radial derivatives;
- collision application and implicit solve;
- diagnostic quadrature and flux moments.

For each order test:

- use at least three refinement levels;
- fit observed order with uncertainty;
- reject pre-asymptotic or roundoff-dominated points;
- state the norm;
- compare to the formal order with a prospective tolerance.

### 17.6 Physics matrix

The retained release matrix should include at minimum:

| Physics | Core cases |
| --- | --- |
| Free streaming/phase mixing | slab Hermite cascade, recurrence, closure effect |
| Electrostatic waves | Landau damping and simple dispersion roots |
| Zonal response | Rosenbluth-Hinton tokamak, stellarator response/damping |
| ITG | Cyclone adiabatic and kinetic electron, Miller shaping |
| TEM/ETG | kinetic-electron trapped mode and electron-scale branch |
| Electromagnetic | KAW and KBM; microtearing when model support is complete |
| Geometry | circular, Miller, W7-X, HSX, QA/QH/QI, finite beta, `LASYM` |
| Collisions | conductivity, relaxation, ZF damping, ITG/TEM operator comparisons |
| Nonlinear | tokamak ITG and at least two stellarator configurations |
| Quasilinear | training/holdout families and domain failures |
| Optimization | derivative direction, constrained line search, held-out nonlinear audit |

Each case records inputs, normalization, convergence ladder, accepted range, source, and claim boundary.

### 17.7 Nonlinear statistical gates

A nonlinear value is accepted only when:

- spin-up is excluded by a prospective rule;
- the averaging interval is long relative to the integrated autocorrelation time;
- corrected relative SEM is below the case threshold;
- adjacent batches agree within their uncertainty;
- heat flux and relevant free-energy diagnostics show bounded trend;
- spectral tails pass necessary resolution screens;
- matched timestep, perpendicular, parallel, and velocity-space rungs agree;
- replicate seeds are consistent or their spread is included in uncertainty.

A causal stopping policy must be frozen on training traces and evaluated without retuning on held-out traces. A favorable post-hoc suffix is evidence about the final mean, not validation of a stopping algorithm.

### 17.8 Coverage and mutation policy

- Measure branch coverage.
- Require 100% coverage of public error branches and schema migrations.
- Require at least 95% branch coverage for every retained promoted module.
- Remove dead or obsolete code before writing tests for it.
- Apply mutation testing or targeted fault injection to high-risk kernels: field solve, bracket, collisions, twist/shift, restart, and derivatives.
- A test must fail when the intended sign, normalization, neighbor index, invariant-restoring term, or timestep stage is perturbed.
- Avoid snapshot tests of prose or large JSON reports unless the schema itself is the contract.

### 17.9 Test tiers

- **Tier 0, under 60 s CPU:** API, algebra, small manufactured systems.
- **Tier 1, under 10 min CPU:** integration, examples, analytic physics, coverage.
- **Tier 2, scheduled CPU/GPU:** representative linear/nonlinear, convergence, gradients.
- **Tier 3, manual campaign:** long turbulence ensembles, external comparisons, optimization.

Only Tier 0 and Tier 1 should block every ordinary PR. Tier 2 blocks release and relevant physics changes. Tier 3 produces reviewed evidence and compact self-contained gates.

---

## 18. Documentation redesign

### 18.1 Information architecture

Use Sphinx with MyST Markdown for narrative pages, MathJax, autodoc, BibTeX citations, copy buttons, and a modern accessible theme such as PyData Sphinx Theme. Keep dependencies deliberate; do not add notebooks or gallery frameworks unless they reduce maintenance.

Organize by user need:

```text
docs/
  index.md
  getting_started/
    install.md
    first_linear_run.md
    first_nonlinear_run.md
    first_stellarator_run.md
  tutorials/
    kinetic_electrons.md
    electromagnetic.md
    collisions.md
    quasilinear.md
    autodiff.md
    optimization.md
  how_to/
    choose_resolution.md
    use_toml.md
    restart.md
    run_gpu.md
    run_scans.md
    diagnose_convergence.md
    use_vmex.md
    couple_transport.md
  explanation/
    gyrokinetic_model.md
    normalization.md
    hermite_laguerre.md
    geometry.md
    fields.md
    collisions.md
    closures.md
    nonlinear_statistics.md
    differentiability.md
  reference/
    inputs.md
    outputs.md
    api.md
    cli.md
    equations_to_code.md
    validation_matrix.md
    performance.md
    limitations.md
    provenance.md
    citations.md
```

Internal release policy, roadmap, campaign logs, manuscript figures, and architecture migration do not belong in the public documentation navigation. Keep them under `plan/` or GitHub project metadata.

### 18.2 Writing rules

- Lead with the concrete user or physics point.
- Use active voice and exact names, dates, commands, equations, and measured values.
- Remove generic importance language, binary slogans, faux-insight setups, and repetitive summaries.
- Do not claim universal support from one benchmark.
- Define every symbol near first use.
- Tie each equation to normalization, code owner, and test.
- Keep docstrings concise; put derivations in documentation, not in 80-line source comments.
- Verify all links and citations in CI, with an explicit allowlist for sites that block automated link checks.

The maintainer reviewed the 1.8.2 README on 2026-08-30 and named the specific
patterns to remove. They are listed here as a checklist because they recur:

| Pattern | Example flagged | Rule |
| --- | --- | --- |
| Meta-commentary about the artifact itself | "It is deliberately short and says so" | Show the thing; do not narrate the choice to show it. |
| Packaging trivia | "The wheel pulls in CPU JAX, SciPy, matplotlib, NetCDF4..." | State the Python floor and link the JAX install guide. Nothing else. |
| Cutesy headings | "Stop when the answer stops changing" | Headings name the feature: "Stopping at saturation". |
| Prose where a table belongs | the saturation paragraph, the cost paragraph | If it enumerates, it is a table. |
| Self-justifying em-dash clauses | "the same data twice, because a flux-tube movie that only shows..." | Caption states what is shown, not why it is worth showing. |
| Parenthetical hedges | "(Transcript abridged; the per-chunk ETA is whatever the host can sustain...)" | Abridge silently, or do not abridge. |
| Caveat paragraphs longer than the claim | the CPU/GPU speedup span, the parallel status table | One sentence of scope, then a link. |
| A figure or movie with no physical content | the rotating racetrack loop | Every asset must carry information the still text cannot. |

### 18.3 Documentation completeness gates

A capability is not complete until docs contain:

- purpose and scope;
- equations and assumptions;
- inputs and defaults;
- outputs and units;
- numerical method;
- convergence guidance;
- limitations and failure modes;
- runnable TOML and Python examples;
- plots from a validated case;
- API reference;
- primary sources.

All code snippets used as tutorials should execute in CI or be imported from tested example files.

### 18.4 README target

The maintainer's ruling on 2026-08-30: "i dont care about specific readme number
of lines, just about the content". The line count below is therefore an
indication of density, not a gate, and no PR should be judged against it. What
is binding is the ordering and the content list:

1. badges;
2. one-sentence identity and scope;
3. one strong image;
4. installation;
5. one CLI run;
6. one Python run;
7. supported capabilities and explicit limitations;
8. links to examples, documentation, validation, citation, and contributing.

Move detailed benchmark tables, movie encoding, machine profiles, claim inventories, and long transcripts into documentation. The README should help a new user run GKX before explaining the development history.

Ordering is a decision, not a preference. The maintainer's ranking on
2026-08-30: "more important than these proses and videos, are actually how to
install gkx, examples of using it, api and toml, that should be the number one
priority of gkx. possibly the flux tube movie at the top to captivate people's
attention, but then installation and usage is more important than some other
results". The performance-versus-GX panel moves near the top because "users
really want to see that", and is re-rendered two rows by one column so each
panel is legible at README width.

The closed-mirror section is refactored to one short paragraph. All four mirror
assets leave the README; they stay tracked because `docs/geometry.rst` uses
them. The rotating-racetrack loop is removed: animating the camera around a
closed circuit tells the reader nothing and confuses what the geometry is.

Two findings from that refactor, recorded because they outlive the README pass:

- Open-ended mirrors are structurally inadmissible, not merely unimplemented.
  The parallel derivative is a periodic FFT (`operators/linear/streaming.py`
  `grad_z_periodic`) and the only admitted boundaries are `periodic` and
  `linked` (`operators/linear/cache_builder.py`, `solvers/nonlinear/
  state_integration.py`). The racetrack closure exists because periodicity is
  required. A README figure showing a GKX flux tube on an open mirror would be
  a false capability claim, so it must not be built.
- The shipped hybrid's `|B|` is a two-level step -- flat on each straight leg,
  varying only through the bends -- so the recorded `mirror_ratio = 1.7785` is
  the ratio between two legs, not between a well minimum and two localized
  throats. The tracked panel is additionally centered on the high-field leg, so
  it reads as a barrier rather than a well. Whether that framing is correct is
  an open review item; the numbers stand, the words around them may not.
  Regenerating the asset needs a machine with a newer VMEX: the installed 0.6.0
  lacks `vmex.mirror.turbulence`, so both the builder and `from_vmex_mirror`
  fail at import on the laptop.

---

## 19. Example redesign

### 19.1 Canonical gallery

Keep 10-12 numbered example groups:

```text
examples/
  01_linear_tokamak/
    case.toml
    run.py
  02_linear_stellarator/
    case.toml
    run.py
  03_nonlinear_tokamak/
    case.toml
    run.py
  04_nonlinear_stellarator/
    case.toml
    run.py
  05_kinetic_electrons/
    case.toml
    run.py
  06_electromagnetic/
    case.toml
    run.py
  07_collisions/
    case.toml
    run.py
  08_quasilinear/
    case.toml
    run.py
  09_autodiff/
    case.toml
    run.py
  10_vmex_optimization/
    run.py
  11_parallel_scan/
    case.toml
    run.py
  12_restart_and_analysis/
    case.toml
    run.py
```

Additional validation inputs belong under `benchmarks/cases/`, not the user gallery.

### 19.2 Python style

Every canonical Python example shall:

- have no `argparse`;
- have no `main()` function;
- have no `if __name__ == "__main__"` guard;
- define editable input parameters near the top;
- import only public APIs;
- construct or load geometry explicitly;
- construct a case or solver;
- run the calculation;
- print a concise summary;
- save machine-readable results;
- make and save polished plots;
- state expected runtime and device assumptions in the module docstring;
- remain readable from top to bottom without helper indirection unless a helper explains a real repeated concept.

Example template:

```python
"""Linear Cyclone ITG scan; seconds on CPU, faster after JAX compilation."""

from pathlib import Path

import gkx

OUTPUT = Path("outputs/linear_tokamak")
KY = [0.1, 0.2, 0.3, 0.4, 0.5]

case = gkx.load("case.toml")
simulation = gkx.prepare(case)
result = simulation.scan("ky", KY)

result.print_summary()
result.save(OUTPUT / "scan.nc")
result.plot(OUTPUT / "figures")
```

Tests should execute every canonical example at a bounded smoke resolution and verify its outputs. Long production settings may be stored in a paired `case_full.toml` outside ordinary CI.

### 19.3 TOML policy

- Remove `runtime_` prefixes from canonical filenames.
- Every key appears in the input reference with type, default, units, valid values, and physics meaning.
- Keep short tutorial TOMLs; provide complete annotated reference TOMLs separately.
- `gkx validate case.toml` checks schema, geometry availability, memory estimate, timestep policy, collision compatibility, and output path before compilation.
- Schema migration errors name the old key, new key, and supported release window.

---

## 20. Plotting and visual evidence

### 20.1 Public plotting contract

All result classes support:

```python
result.plot(output_directory)
gkx.plot("result.nc", output_directory)
```

A plotting function should accept an existing `Axes` or return a `Figure` without mutating global `matplotlib.rcParams`. Use `matplotlib.rc_context` and one documented style module.

### 20.2 Standard figures

Linear:

- growth rate and frequency versus `k_y` with convergence/error markers;
- eigenfunction amplitude and phase along the field line;
- field-channel and species contributions;
- fit interval and residual quality.

Nonlinear:

- flux time series with accepted/rejected averaging window and uncertainty;
- heat/particle/momentum spectra;
- field and free-energy spectra;
- real-space perpendicular snapshot;
- physical field-line/flux-tube view;
- convergence and spectral-tail panel;
- compact run summary with warnings.

Collisions:

- invariant residuals;
- eigenvalue/dissipation spectrum;
- moment convergence;
- conductivity/relaxation/ITG/TEM/ZF comparison.

Optimization:

- objective and constraints versus iteration;
- gradient uncertainty/direction agreement;
- baseline and candidate geometry;
- held-out transport comparison with error bars;
- surface/field-line coverage map.

### 20.3 Publication-quality rules

- save vector PDF/SVG for line art and high-resolution PNG/WebP previews;
- include units, normalization, legends, panel labels, and uncertainty;
- use an accessible palette and distinguish series by line/marker as well as color;
- avoid tiny fonts, cropped labels, excessive whitespace, and rasterized text;
- make axes limits/data transforms explicit;
- show failed or unresolved cases rather than dropping them;
- store machine-readable data and a reproduction command beside each retained documentation figure;
- keep tracked previews compact; place larger movies or campaign bundles in release assets.

Add image-comparison tests only for stable layout-critical figures. Prefer structural tests of labels, series, scales, warnings, and saved formats over brittle pixel snapshots.

---

## 21. Packaging, dependencies, and repository hygiene

### 21.1 Core dependency review

Audit every base dependency by tracing imports from a normal linear and nonlinear run.

Likely final base:

- `jax`;
- `numpy`;
- `scipy` only for required host algorithms;
- one NetCDF library;
- `matplotlib`;
- `rich` for CLI presentation if adopted;
- `solvax` only for algorithms actually exercised by the stable product.

Candidates for removal or relocation:

- explicit `jaxlib` dependency when normal JAX packaging suffices;
- Diffrax after integrator migration;
- Equinox if only small PyTree helpers use it;
- `booz_xform_jax` from base installation after VMEX/imported geometry ownership is clean;
- `tqdm` if Rich owns progress;
- pandas outside validation extras;
- Pillow outside media-building extras.

Use bare dependency names by default. Add a lower bound only when GKX requires an API or correctness fix absent in older versions. Avoid speculative upper bounds. Maintain exact CPU and NVIDIA constraints files for reproducibility, separate from package metadata.

### 21.2 Environment matrix

CI and release testing:

- Python 3.11 minimum supported stack;
- latest supported Python/JAX stack;
- Linux CPU full Tier 0/1 suite;
- scheduled NVIDIA GPU Tier 2 matrix;
- wheel and sdist clean installs;
- documentation build and link check;
- optional integration environments for VMEX, Pyrokinetics, and validation tools.

### 21.3 Repository size

- keep a normal clone below 20 MB when practical;
- no raw NetCDF campaign data in Git;
- no profiler traces, logs, local environments, or generated full-resolution figures;
- compact numerical fixtures must have provenance and a clear test owner;
- use release assets for movies and bounded reproduction bundles;
- run size gates on Git history additions, not only working-tree size.

### 21.4 Developer commands

Consolidate to at most eight commands under `scripts/`:

```text
check.py          lint, type, architecture, size, docs, release
inventory.py      files, lines, imports, cycles, API, duplicates
benchmark.py      representative CPU/GPU benchmark matrix
profile.py        configurable JAX trace/profile runner
validate.py       scientific validation matrix
figures.py        reviewed documentation/README figures
compare.py        local external-code comparison protocol
release.py        build, smoke install, metadata, tag preflight
```

Each command must be configuration-driven. Delete one-off generators and report-specific scripts after migrating their useful behavior.

---

## 22. Source comments, docstrings, typing, and provenance

### 22.1 Docstrings

Require:

- 100% docstrings for public modules, classes, methods, and functions;
- concise docstrings for private functions that implement nontrivial physics, mathematics, numerics, normalization, or array layout;
- parameters, returns, shapes, units/normalization, differentiability, and failure behavior where relevant;
- equation and source references for scientific kernels.

Do not enforce long docstrings on obvious one-line private helpers. The metric is whether a maintainer can understand the contract without reverse engineering call sites.

### 22.2 Comments

Comments should explain:

- why an algorithm or layout is used;
- a subtle sign, normalization, parity, or boundary convention;
- why fusion is blocked or precision is pinned;
- the invariant or equation preserved;
- a known limitation and its gate.

Delete comments that restate syntax, narrate history already in Git, or preserve obsolete campaign status.

### 22.3 Typing

- remove `Any` from public user contracts;
- use PyTree-compatible immutable dataclasses;
- type geometry, case, result, objective, and collision protocols;
- keep array shape information in docstrings and optional typing helpers without adding a large runtime dependency;
- make MyPy or Pyright pass on the full installable package with narrowly justified external-library ignores.

### 22.4 Provenance

For directly adapted GX code, use a short source note:

```python
"""Construct Miller metric coefficients.

Adapted from GX `geometry_modules/miller/gx_geo.py` at revision `<sha>`.
See `PROVENANCE.md` for the full mapping and license.
"""
```

The root ledger owns full history, hashes, license, and original contributors. Do not insert a historical essay into every function. New ports without an exact upstream revision and source path are not accepted.

---

## 23. Phased execution program

A phase closes only when every exit gate passes on `main` after merge.

### Phase H0: current-head handoff and rebaseline

**Purpose:** make this plan and the current audited revision the unambiguous ground truth.

Tasks:

- replace root `plan.md` with this file;
- move the old static plan to `plan/archive/plan_pre_2026-08-30.md`;
- keep `plan/log.md` as the historical append-only log;
- regenerate exact source/test/tool/docs/example counts at `e19336dc...`;
- update architecture targets to the approved 45/45k, 30/35k, <=30 API limits;
- regenerate current import graph, cycles, API inventory, coverage deficits, largest modules, and duplicate groups;
- add a compact PR ledger for #1-#162, linking the existing detailed audit rather than duplicating it;
- obtain a green post-merge `main` CI run;
- verify branch protection contexts.

Exit gates:

- exact current metrics committed;
- no stale planning-branch instructions;
- architecture manifest matches approved targets;
- `main` green after merge;
- this file identifies one active next PR.

### Phase A: true architecture and API contraction

#### A1: public data model

- implement real immutable `Case`, `Species`, grid/geometry/physics/time/output types;
- implement real result classes;
- preserve TOML/NetCDF schema compatibility through explicit adapters;
- remove aliases once migrated.

#### A2: prepared simulation and orchestration

- implement typed `PreparedSimulation` for linear and nonlinear solves;
- move cache/compile/device policy into it;
- make CLI, Python solve, scans, optimization, and examples call the same API;
- eliminate patchable global dependency bundles.

#### A3: root API contraction

- move advanced functions into `gkx.geometry`, `gkx.physics`, `gkx.optimize`, and `gkx.integrations`;
- replace the 350-target lazy registry with the final advertised surface;
- provide one release of explicit import migration errors where justified.

#### A4: source package consolidation

- merge `runtime`, `workflows`, `artifacts`, `terms`, and solver wrappers into target owners;
- delete campaign/release report builders from package;
- reduce import cycles to zero;
- reduce source to at most 70 files and 65,000 lines as an intermediate gate.

Phase-A exit:

- real public types and prepared object;
- no hidden broad API;
- no import cycles;
- <=70 source files and <=65,000 lines;
- all numerical fingerprints and user workflows preserved or deliberately migrated.

### Phase B: geometry, integration, and dependency deletion

Tasks:

- complete VMEX live/WOUT/EIK parity and delete duplicated VMEC/Boozer code;
- decide the unique retained role of imported GX-derived geometry;
- close the stiff-path decision using time-to-accuracy evidence;
- remove Diffrax if no unique promoted role remains;
- remove unused Equinox/booz_xform_jax/tqdm/jaxlib dependencies as justified;
- consolidate I/O and plotting;
- reach <=55 source files and <=52,000 lines.

Exit gates:

- exact geometry ownership;
- one explicit owner and at most one useful stiff owner;
- dependency clean-install matrix;
- no behavior change hidden by delegation;
- CPU and NVIDIA performance no worse at matched error.

### Phase C: test-suite consolidation and proof matrix

Tasks:

- implement the target 24-28 file structure;
- convert wrapper/artifact tests into broad behavior tests;
- add missing branch coverage for retained modules;
- implement algebraic, manufactured-solution, order, and fault-injection gates;
- keep local external comparisons outside CI;
- delete tool-test mirrors.

Exit gates:

- <=30 files and <=35,000 lines;
- >=95% package statement and retained-module branch coverage;
- all public errors covered;
- mutation/fault tests detect representative sign/index/normalization defects;
- Tier 0 and Tier 1 runtime within budgets.

### Phase D: user experience, documentation, examples, and plots

Tasks:

- rewrite README;
- implement Diataxis-style docs;
- rebuild inputs/outputs/API/equations/reference;
- replace examples with the canonical gallery and paired TOMLs;
- implement result-owned plots and publication formats;
- remove old generated data and obsolete docs.

Exit gates:

- fresh Python 3.11 user completes install and first linear/nonlinear run from docs;
- every canonical example executes in CI at smoke resolution;
- no example imports private APIs or uses argparse/main guards;
- strict docs build and link policy pass;
- tracked repository remains below size target.

Gate status, measured 2026-09-01:

- *fresh user first run*: enforced for both halves. The wheel smoke test
  proved only that the wheel imports; it now runs bare ``gkx`` from the
  installed wheel in an empty directory and requires the five artifacts
  quickstart.rst names (10 s), then the documented nonlinear command (38 s).
  The third documented command, ``run-runtime-linear`` on the Cyclone deck,
  measured 970 s -- sixteen minutes of multicore work sitting in the same
  quickstart block as the ten-second demo. It stays off the pull-request lane,
  and the quickstart should stop presenting it as a first command.
- *every canonical example executes*: 16 of 36 execute at smoke resolution, up
  from 4. The other 16 are registered against the specific thing that blocks
  them (an untracked ``*.eik.nc``, a ``wout_*.nc`` from a full VMEC solve,
  eight devices), and a static guard fails if an example is neither executed
  nor skipped-with-a-reason. Running them found two bugs that the mocked
  coverage could not: the parallelization deck never ran at all, and
  linear_rhs_demo integrates an identically zero state because its seed index
  is out of bounds on an ``Nx == 1`` grid and JAX drops the scatter silently.
- *no private-API imports, no argparse/main guards*: half met. Zero examples
  import a private name and zero use argparse. Eleven still carry a ``__main__``
  guard; the registry pins the entrypoint shape so a module-body example cannot
  grow one, but converting the remaining eleven is open.
- *strict docs build and link policy*: enforced. ``sphinx -W`` was already
  gated; linkcheck now runs nightly. Its first pass found 27 broken links of
  which 23 were publisher 403s rather than rot, so the gate fails only on
  404/410. The four real ones were three unregistered DOIs and a link to an
  unpublished branch, all fixed.
- *repository below size target*: met, 19.9 MB tracked against a 50 MB ceiling.

### Phase E: core physics validation

Tasks:

- close systematic electrostatic/electromagnetic and adiabatic/kinetic-electron matrix;
- close geometry family matrix;
- close nonlinear statistical/convergence policy;
- document unsupported combinations;
- keep equilibrium `E x B` shear deferred unless reopened.

Exit gates:

- E1-E6 evidence for every promoted core feature;
- no README capability claim lacks an indexed validation gate;
- unresolved and negative results remain visible.

### Phase F: scalable collisions and closures

Implement C0-C3 and the closure program from Section 12.

Exit gates:

- arbitrary-order drift-kinetic and finite-`k_perp` runtime;
- multispecies conservation and H-theorem gates;
- conductivity, relaxation, ZF, ITG, TEM, and nonlinear comparisons;
- practical memory/runtime scaling on NVIDIA;
- documented TOML/Python selection.

### Phase G: quasilinear model and transport interfaces

Tasks:

- freeze model domain and data split;
- evaluate candidates;
- promote best passing Q1/Q2/Q3 tier;
- add model card, uncertainty, OOD behavior, and user API;
- add Pyrokinetics and transport adapters.

Exit gates:

- prospective untouched holdouts;
- explicit limitations;
- stable transport mock/coupled loop;
- round-trip interoperability tests.

### Phase H: nonlinear derivatives and flagship optimization

Tasks:

- ensemble and directional validation of finite-window gradients;
- matched-cost SPSA/finite-difference/AD comparison;
- 50-200 controls, multiple surfaces/field lines/`k_y`;
- constrained optimization;
- long held-out nonlinear audit.

Exit gates:

- gradient usefulness quantified with uncertainty;
- final candidate passes all physics and statistical gates;
- campaign code remains outside installable core;
- complete reproducibility manifest without committing large raw outputs.

### Phase I: GKX 3.0 release

Final gates:

- <=45 source files and <=45,000 lines;
- <=30 test files and <=35,000 lines;
- <=30 root names;
- zero import cycles;
- >=95% aggregate and retained-module branch coverage;
- clean Python 3.11 wheel/sdist and latest stack;
- CPU/NVIDIA validation and performance matrices;
- rewritten docs/examples/plots;
- PyPI and conda-forge readiness;
- citation, provenance, governance, changelog, and compatibility policy;
- green post-merge `main`.

---

## 24. Immediate pull-request queue

Work in this order unless a defect blocks users.

### PR H0-1: replace plan and rebaseline current `main`

Files limited to planning, architecture manifests, generated inventory summaries, and branch/CI policy. No solver change.

Acceptance:

- exact counts and current coverage deficits;
- complete current PR ledger;
- updated approved targets;
- old plan archived;
- green post-merge `main`.

### PR A1-1: import graph and deletion map

Generate and review:

- module dependency graph and cycles;
- public name downstream usage in GKX, VMEX, and canonical examples;
- reachability from `gkx.solve`, CLI, examples, and tests;
- dead modules and forwarding wrappers;
- campaign/report modules still installed;
- concrete file-by-file move/delete map to <=70 files.

Delete only clearly dead wrappers in this PR. No large moves.

### PR A1-2: real public case and result types

- add immutable public types;
- adapt current loaders/results internally;
- preserve numerical values and schemas;
- add concise API/schema tests;
- begin removing aliases.

### PR A1-3: real `PreparedSimulation`

- linear and nonlinear support;
- typed solve/scan/value-and-grad;
- compilation and cache metadata;
- public examples use it;
- remove patchable runtime dependency bundles that become unnecessary.

### PR A2-1: CLI and workflow consolidation

- six clear commands: `run`, `scan`, `estimate`, `plot`, `inspect`, `validate`;
- old commands emit migration messages for one release;
- CLI delegates to public API;
- remove duplicated flag and command orchestration.

### PR B1-1: remaining geometry deletion

- freeze parity;
- remove obsolete VMEC/Boozer modules and objective/report consumers;
- update provenance;
- show substantial source/test deletion.

### PR B2-1: integrator/dependency decision

- finish matched explicit/existing-implicit evidence;
- remove Diffrax and duplicate source if the gate selects native ownership;
- otherwise document one precise retained stiff role;
- update TOML, docs, examples, and constraints.

### PR C1-1 onward: test consolidation by domain

Merge one domain at a time, preserving detection power while reducing files and lines. Do not postpone all deletion to a final mega-PR.

The acceptance check is a diff of the *collected pytest node-ID sets* before and
after, not a comparison of totals. Four domains have now been merged and the
hazards found are worth stating, because most of them keep the suite green:

- **Decorators vanish.** Emitting a function with `ast.get_source_segment`
  starts at `def`, dropping every decorator and collapsing each `parametrize`
  family to one test. Caught by the ID diff.
- **Module constants collide.** Two files defining the same top-level name leave
  the later value winning for the whole merged module, so tests run against the
  wrong number while still passing. Rename per origin block; never delete a
  definition. Caught by an explicit collision check, not by the ID diff.
- **`pytestmark` widens silently.** A module-level marker applies to every test
  in the merged file. A float64 `skipif` from one file would have extended to
  unrelated gates, and a widened skip turns green, not red. Convert module
  markers to per-test decorators on exactly their original tests.
- **Import hoisting breaks `sys.path` bootstraps.** Where a module-level
  `sys.path.insert` must precede an import of a local helper, hoisting imports
  above statements breaks collection.
- **Docstrings carry provenance.** Absorbed module docstrings cite papers,
  equations, and measured values; preserve them verbatim at the origin marker.
- **Live code loads tests by path.** `tests/unit/parallel/test_parallel_core.py`
  reads a test module by file path. Grep deleted basenames across `.py`, config,
  workflows, and docs -- not only the test tree.

Ratchet the manifest centrally when several consolidation branches are open.
Each measures against the same `main` and independently computes the same new
baseline, which is wrong for every branch after the first.

### PR D1-1: README restructure

Execute §18.4. Ship the reordering, the prose cuts, the feature table, the
re-rendered two-row performance panel, and the refactored mirror section in one
PR so the README is never half-converted on `main`. The mirror replacement asset
and the saturation figure may land in the same PR or immediately after it, but
the rotating racetrack loop is removed either way.

### PR D1-2: documentation prose pass

Apply the §18.2 checklist to `docs/*.rst`. Cut rhetoric only: no technical fact,
number, citation, equation, or cross-reference may be lost. `sphinx -b html -W`
stays clean.

---

## 25. Agent operating contract

### 25.1 Branch and review

- Branch from current `main`.
- Use a descriptive branch name.
- Keep commits small and reversible.
- Never merge to `main`.
- Do not rewrite shared history.
- `rogeriojorge` reviews and merges.

### 25.2 Change discipline

A normal PR should change one of:

- product/API contract;
- one scientific owner;
- one numerical owner;
- one test domain;
- one documentation/example domain;
- one performance bottleneck;
- one physics capability.

Do not combine broad file movement, new physics, performance tuning, generated figures, and campaign results.

### 25.3 Required pre-PR evidence

Always run:

```console
python scripts/inventory.py
python scripts/check.py
python -m pytest <focused tests>
sphinx-build -W -b html docs docs/_build/html
python -m build
```

Until those scripts are consolidated, use the current equivalent release/architecture commands documented in the repository.

For numerical changes, also record:

- exact case and command;
- precision and hardware;
- before/after value and tolerance;
- convergence or residual;
- cold/warm runtime and memory if performance may change;
- derivative check if the path is differentiable.

### 25.4 Rollback rules

Rollback or split the work when:

- source/test lines grow without a measured capability;
- a new owner duplicates an old path;
- a method does not beat its prospective accuracy/performance gate;
- a scientific result changes without a derivation and independent check;
- a test passes only by weakening a literature or invariant tolerance;
- a public API needs private objects to be useful;
- a geometry adapter reconstructs equations owned by VMEX;
- an external-code discrepancy is “fixed” by copying its number without resolving normalization and convergence;
- documentation claims more than the indexed evidence supports.

### 25.5 Work-log template

Append to `plan/log.md`:

```markdown
## YYYY-MM-DD - <task and branch>

Baseline:
- GKX SHA:
- companion SHAs:
- source/test/tool files and lines:
- relevant existing gate:

Scope:
- intended change:
- non-goals:
- prospective acceptance and rollback criteria:

Changes:
- files/functions removed, merged, or added:
- public/schema behavior:

Evidence:
- focused tests:
- physics/mathematics/numerics gates:
- CPU/NVIDIA measurements:
- values, tolerances, residuals, uncertainty:

Outcome:
- accepted, rejected, or partial:
- remaining blocker:
- next task:
```

Do not paste full logs, stack traces, or generated tables into the work log. Store them in ignored local output or bounded CI artifacts and record the command and digest.

---

## 25b. Source simplification, measured

The test-suite analysis asked what the lines actually assert rather than how
many there are. The same question was put to `src/gkx` on 2026-08-31 at
`c23f6f94`. It is recorded here because the answer changes what "simplify"
should mean: there is nothing to delete, and a great deal to *flatten*.

### 25b.1 Measured shape

193 files, 88,305 lines, 2,929 functions, 375 classes. Docstrings and comments
are 5.6 per cent of the file, which is low for research code and consistent
with the rule that derivations live in documentation.

Functions classified by what their body does:

| Kind | Functions | Lines | Share |
| --- | ---: | ---: | ---: |
| numerical kernel (`jnp`/`lax`/fft) | 706 | 24,974 | 34.7% |
| other logic | 1,080 | 25,730 | 35.7% |
| validation and error construction | 295 | 7,865 | 10.9% |
| thin wrapper (<=3 statements, returns a call) | 347 | 6,315 | 8.8% |
| pure delegation (single return of a call) | 381 | 4,633 | 6.4% |
| serialization and IO | 105 | 2,507 | 3.5% |
| stub or docstring-only | 15 | 21 | 0.0% |

### 25b.2 What the measurement rules out

- **There is no dead code.** Of 3,304 definitions in `src`, zero appear only at
  their own definition site. Nothing can be removed for free, so simplification
  cannot come from deletion.
- **Only about a third of the package is numerical kernel.** The physics is
  24,974 lines; the rest is structure around it.

### 25b.3 What the measurement points at

- **27.2 per cent of the package -- 1,379 functions, 24,027 lines -- is
  indirection.** That is the union of thin wrappers and helpers with exactly one
  call site and at most 40 lines. A 318-line kernel called once is good
  decomposition; a 12-line helper called once is a hop that hides the code from
  its reader.
- **The file-size cap manufactures some of it.**
  `tools/package_architecture_manifest.toml` sets `default_max_lines = 1000`,
  and six modules sit at 986 to 1000 lines. A hard per-file cap does not reduce
  complexity, it displaces complexity into helper modules and wrappers. The cap
  should become a cohesion rule, or be raised and gated on complexity, before
  any inlining pass -- otherwise flattening a file simply pushes it over the
  ceiling and the indirection comes back.
- **Twenty functions, 400 lines, are referenced only by tests.** Among them
  `streaming_contribution`, `exb_nonlinear_contribution`,
  `bessel_laguerre_kernels`. Each is either public API that should be
  advertised and documented, or a helper that should be folded into its caller.
  Existing only to be tested is neither.
- **Five modules are pure re-export facades**, 238 lines, with no definitions.

### 25b.4 Ordered program, and the gate it must pass

1. Replace the flat 1,000-line cap with a cohesion rule, or raise it and gate on
   complexity. Nothing below can be done honestly while the cap rewards hiding.
   **Done.** `[cohesion_policy]` in the architecture manifest now measures two
   things and ratchets both downward: the number of modules imported by exactly
   one other module (42 at `c23f6f94`, holding 18,361 lines, which is the
   fingerprint of cap-driven splitting), and the number of modules with at least
   six top-level definitions whose internal-reference ratio is below 0.30 (8).
   `default_max_lines` moves 1000 -> 1200 and is demoted to a runaway tripwire.
   The single-consumer arm counts only edges where the consumer imports at
   least six names, so the gate cannot push anyone into dissolving a genuine
   interface to improve its score.
   Merging a single-consumer module back into its only caller now improves the
   score instead of breaching a limit. Both arms were verified to fire by
   tightening each baseline by one and confirming the checker raises.
2. ~~Collapse pure delegation, 381 functions and 4,633 lines.~~ **Withdrawn on
   inspection.** The heuristic that produced 4,633 lines counts named
   constructors and jit aliases as waste. `_bracket_evidence_config` converts
   one config into another and `_matched_transport_context` builds a context
   from a payload; inlining either would make its caller worse, not better.
   Applying the strict test -- a forwarder passing its own parameters straight
   through, unchanged -- leaves 22 functions and 318 lines, and several of those
   are legitimate (`assemble_rhs_cached_jit` is a jit alias; `solve_fields` and
   `solve_fields_species_shard` are two entry points onto one implementation).
   There is no function-level refactor worth doing here, and the original figure
   should not be quoted again.
3. Merge modules that were split rather than encapsulated. Interface width is
   the discriminator, not consumer count: `geometry/imported_miller.py` has one
   consumer that imports exactly one name, which is 941 lines behind a real
   entry point and must be left alone, while
   `workflows/runtime/initial_conditions.py` has one consumer that reaches for
   ten of its names, which is one module cut in two. Five modules are both
   deeply coupled and small enough to rejoin under the 1,200-line tripwire.
4. Consolidate validation and error construction, 295 functions and 7,865
   lines, behind shared validators. The message text is repeated far more than
   the logic is.
5. Resolve the twenty test-only functions in each direction deliberately.
6. Collapse the five facade modules into explicit `__init__` exports.

The feature-preservation gate, and it is not optional: the advertised public API
in `gkx.api.__all__` and the lazy `_EXPORT_TARGETS` registry may not shrink
except by a decision recorded in this plan; the full test suite keeps its
collected node-ID set; and the physics gates are unchanged. Simplification that
cannot show those three is refactoring on trust, which is what this section
exists to avoid.

## 25c. Flat source layout, measured

Section 25b asked what the lines do. This section asks what the *directories*
do, prompted by the observation that `src/gkx` carries twenty of them, some
holding a single file. Measured at `c23f6f94` on 2026-08-31.

### 25c.1 What the structure actually is

193 files in 20 directories, 88,988 lines. Several directories exist to hold
almost nothing: `solvers/__init__.py` is one line and the package holds only
subpackages; `operators/` has 50 lines of its own beside its subpackages;
`benchmarking/` holds one real file of 933 lines; `core/` holds two;
`utils/` holds two, named `callbacks` and `compilation_cache`, under a
directory name that says nothing about either.

### 25c.2 Three hypotheses tested, one confirmed

- **"There are experimental lanes nobody uses."** Largely false. Walking the
  import graph from `gkx.api`, `gkx.cli`, and every module named in the lazy
  `_EXPORT_TARGETS` registry reaches 178 of 193 modules. Only 15 modules and
  2,436 lines are unreachable. The package is not full of abandoned work.
- **"`objectives/` is experimental."** False as stated, but it is lopsided.
  It is 10,918 lines across 21 modules, and exactly four names from all of it
  are advertised in `gkx.api.__all__`, three from `core.py` and one from
  `vmec_transport.py`. Seven modules totalling 3,667 lines export nothing at
  all, and the `vmec_boozer_*` cluster alone is 2,355 of those. Every one of
  them has between one and four internal consumers, so they are deep machinery
  rather than dead code. The problem is proportion, not abandonment.
- **"`data/` is test-only ballast."** False, and the way it was nearly got
  wrong is worth recording. A literal grep for each filename found no consumer
  for `finite_wavelength_coulomb.{json,npz}` or their `_18` variants -- four
  files, 112 kB -- so they looked like dead payload shipped in the wheel.
  Deleting them broke three tests. `operators/linear/collision_tables.py`
  builds the names at run time from `_FINITE_WAVELENGTH_STEM` and a moment
  count of 8 or 18, so no literal ever appears in the source. Those tables are
  the `coulomb_finite_kperp` operator, one of the five shipped collision models
  and a headline capability. They were restored.
  **A filename that is assembled at run time is invisible to a grep, so
  "nothing references this" is a hypothesis to be tested by deletion and a test
  run, never a conclusion on its own.** All twelve `data/` files are live.

### 25c.3 The flat layout, and what it costs

Grouping every current module by its directory and allowing 1,200 lines per
file gives **88 flat modules** in place of 193 files in 20 directories. Nothing
about that requires subdirectories.

The tension has to be stated because it constrains the order of work:

| Lines per file | Flat modules needed |
| ---: | ---: |
| 800 | 112 |
| 1,000 | 89 |
| 1,200 | 75 |
| 2,000 | 45 |

Section 2.4 asks for at most 45 installable files and at most 45,000 lines.
Those two are consistent with each other at about 1,000 lines per file. They
are *not* consistent with today's 88,988 lines: 45 files now would mean modules
of about 1,980 lines each, which trades twenty directories for forty-five
unreadable files and is not a simplification. **File flattening and line
reduction are the same problem, and flattening must not be used to claim the
file target while the line count is untouched.**

### 25c.4 Revised program

1. Leave `data/` alone. All twelve files are live; see above for the check that
   proves it and the mistake that nearly removed four of them.
2. Remove the directories that hold nothing: `solvers/`, `operators/`, and
   `benchmarking/` as containers, folding their direct content into the module
   that uses it.
3. Rename by what the code is, not where it sits. `utils` becomes the two
   things it holds; `workflows` and `artifacts` are named for the reader rather
   than the author, and `artifacts` in particular is output plumbing that no
   user of the solver calls.
4. Flatten the remaining packages into single modules per concept, targeting 88
   files at 1,200 lines, and only then pursue 45 as the line count falls.
5. Rebalance `objectives`: 10,918 lines behind four advertised names is the
   largest proportion problem in the package. Decide per module whether it is
   public capability that should be advertised, or machinery that belongs
   inside its one consumer.

The gate from section 25b applies unchanged: the advertised API and lazy
registry may not shrink except by a recorded decision, the collected test
node-ID set is unchanged, and the physics gates are untouched. Moving code
between files must not move anything out of the package.
## 25d. Architecture, tested against the dependency graph

Sections 25b and 25c measured lines and directories. This one measures the
*connections*: a weighted import graph over 182 modules, 667 edges and 1,644
imported names, built at `f9e5e97a` on 2026-08-31. Several candidate
rearrangements were tested against it before proposing any.

### 25d.1 The layering is already sound

Assigning each package a layer (configuration, geometry, terms and operators,
solvers, diagnostics, objectives, artifacts, workflows, entry points) and
counting edges that run downhill gives **17 violating edges out of 667**. The
package is not tangled. Any proposal that reorganises everything is solving a
problem the code does not have, and the earlier instinct to flatten wholesale
should be tempered by that.

### 25d.2 The one structural inversion worth fixing

`RuntimeConfig`, aliased as the public `Case`, lives in
`workflows/runtime/config.py`, the deepest layer, and is imported by
`geometry/vmec_eik.py` and `geometry/miller_eik.py`, among the shallowest.
That is the worst inversion in the graph, and it is a symptom rather than an
accident: **GKX has two parallel configuration systems.**

| | Module | Holds |
| --- | --- | --- |
| physics configs | `gkx/config.py` | `GridConfig`, `TimeConfig`, `GeometryConfig`, `ModelConfig`, `CycloneBaseCase` |
| deck configs | `gkx/config.py` | `RuntimeSpeciesConfig`, `RuntimePhysicsConfig`, ..., `RuntimeConfig` alias `Case` |

The split is mostly clean -- 20 modules under `workflows/runtime` use the deck
family, 23 elsewhere use the physics family -- but the deck family leaks into
**10 modules outside its own package**, and geometry is one of them. The public
entry type a user touches first therefore sits below everything, and pulling it
upward drags the runtime layer with it.

### 25d.3 What the graph says the pieces are

The highest fan-in module is `operators/linear/params.py`: 528 lines, from
which 99 names are imported. That is the real core type module.

**The clustering claim first written here was wrong, and the method was the
reason.** Weighted label propagation was run once, with one seed, and its output
reported as measurement. Repeating it across twenty seeds gives a largest
cluster ranging from 14 modules and 9,299 lines to 75 modules and 34,977 lines:
the algorithm is not reproducible at this size and a single run says almost
nothing. A 300-seed consensus, scored by co-association, gives the durable
answer.

What is actually there is a 39-module, 18,027-line community around the linear
operator and linear solver -- "assemble a linear operator and advance or solve
it". It has **zero import cycles** and is a clean twelve-deep DAG. The
stellarator objectives are **not** part of it: they never co-clustered in 300 of
300 runs, and form their own 17-module, 9,257-line community. The single seed
had merged the two across a seam of seven name-imports.

`params.py`'s fan-in is the reason that community coheres, not a defect. Its 100
imports resolve to only fifteen distinct names -- `LinearParams` thirty-eight
times, `LinearTerms` eighteen -- which is shared vocabulary rather than a
grab-bag. It is not a cut vertex either: removing it leaves the rest connected.
Splitting it would be actively harmful.

Two earlier readings were wrong and are corrected here. `artifacts/spectral_layout.py`
looked misplaced on fan-in alone; its 50 imported names come from only three
consumers, so it is a cohesive cluster rather than a stray. And the
`objectives` package is not experimental: the cluster analysis puts its
stellarator code with the linear solver, which is where it belongs.

### 25d.4 Revised program, in priority order

1. **Lift the case type.** Move `RuntimeConfig` and its `Case` alias to the
   configuration layer, so geometry and the rest stop importing downward from
   `workflows.runtime`. This removes the deepest inversion and is the single
   highest-value structural change available.
2. ~~Decide whether two configuration systems are wanted.~~ **Answered, and the
   premise was wrong.** There are not two systems. `RuntimeConfig` *composes* the
   physics configs -- `grid: GridConfig`, `time: TimeConfig`,
   `geometry: GeometryConfig`, `init: InitializationConfig` are four of its
   thirteen fields -- and the two families share only six field names out of
   about a hundred each, which are the composition points themselves. There is
   no duplicated surface to remove, and the translation layer this section
   assumed does not exist: composition means there is nothing to translate.
   Merging them would push deck concerns such as output paths and quasilinear
   settings into the solver-facing model, and would turn a change of file format
   into a change of physics code. The boundary is load-bearing and stays.
   The nearby simplification that section suggested -- consolidating the nine
   deck-only `Runtime*Config` classes -- was measured and rejected too. Every one
   maps 1:1 onto a TOML table on both paths: `toml.py` holds a literal
   `(section_name, constructor)` table keyed by the `RuntimeConfig` field name,
   and `to_dict` plus `deck_text` render one table per class. Merging any two
   renames a user-visible section and breaks `Case.to_toml` round-tripping
   through `gkx.load`. Even the thinnest, `RuntimeExpertConfig` at 13 lines, is
   a documented section used by two tracked benchmark decks. Three others carry
   `__post_init__` validation and are not passive field bags. **No consolidation
   is safe here; the nine stay.**

   One real duplication was found and deliberately left: `to_dict` returning
   `asdict(self)` is repeated verbatim in seventeen classes. A shared mixin would
   have to cross the physics/deck boundary that this section just established as
   load-bearing, so it is recorded rather than done.
3. **Do not split the linear community.** It is one concept and the data says
   so. The payoff there is not decomposition but five edge fixes totalling seven
   imports, after which it is a strictly layered DAG that documents itself:
   move the `collision_operator_from_config` factory out of the type module into
   `operators/linear/collision_factory.py`; promote two preconditioner builders
   from `solvers.linear.implicit` into a `preconditioners` module; promote
   `_linear_native_step` to a public stepping entry point; and narrow
   `objectives.core`, which reaches into `solvers.linear.krylov_algorithms` for
   the private `_build_shift_invert_precond`. That last one is the whole seam
   joining the objectives community to the linear one.
4. Continue flattening only where a directory is a container. Directory count
   fell 20 -> 13 and the remaining ones hold real clusters.
5. Fuse only where the consumer imports six or more names, and expect chains:
   fusing `stellarator_tables` revealed `stellarator_reduced` at fourteen names,
   which revealed `stellarator_contracts` at six. A module can look
   double-consumed only because both consumers are halves of one split module.

### 25d.5 Encapsulation is the next measurable problem

Promoting `_build_shift_invert_precond` to
`build_shift_invert_preconditioner` closed the seam between the objectives and
linear communities, and the count behind it is worth recording: **216 imports
reach across package boundaries for a private name**, dunders excluded. The
most-reached-into modules are `artifacts/spectral_layout.py` (25),
`solvers_time_explicit_steps` (20), `operators/linear/params.py` (15) and
`solvers_linear_krylov_algorithms` (15).

This is a different problem from layering and from cohesion, and it is the one
that most directly obstructs the plan's other goals. A module whose private
names are imported by four other packages cannot be refactored without breaking
them, which is why so many fusions in this wave turned out to be blocked by a
seam rather than by the physics. The honest programme is to promote the names
that are genuinely part of a contract, as was done here, and to move the rest
inside the one caller that needs them -- not to add a gate first, because a
ratchet on 216 would freeze the current shape rather than improve it.

### 25d.6 Fusion hazards, from four completed fusions

None of these is visible to a grep, and each was caught by a checker or a test:

- a facade and its implementation colliding by name, so the later definition
  silently wins and the facade's own logic never runs;
- a constant used in a **default argument**, which evaluates at definition time,
  so an appended body raises `NameError`;
- a module docstring carrying a **scope claim** -- the reduced stellarator model
  is a fitted feature map with an ODE envelope, not gyrokinetics -- which fusion
  would have deleted;
- precision guards keyed by file name, one of which failed with "the guard is
  vacuous" because it was written to detect its own hollowing-out.

## 25e. Cold-start cost, measured

The performance work starts from what is already recorded in
`docs/performance.rst`: about 97 per cent of a run is time stepping, and within
one step about 59 per cent is data movement, 31 per cent the FFTs themselves,
and under 10 per cent physics arithmetic. That is the *warm* picture. Cold runs
were measured separately on 2026-08-31 and have a different bottleneck.

### 25e.1 The cache build is 278 compilations, not a computation

Cold setup at a deliberately tiny grid, 16x16x8 with `Nl=2, Nm=4`, on CPU:

| stage | time | share |
| --- | ---: | ---: |
| `build_linear_cache` | 2,619 ms | 74.6% |
| `import jax` | 618 ms | 17.6% |
| `build_spectral_grid` | 262 ms | 7.5% |
| everything else | 14 ms | 0.4% |

At that grid the arithmetic is trivial, so the time is not computation, and a
second call proves it: **2,373 ms first, 10 ms second**. Changing the grid to
32x32x16 re-pays it in full at 2,166 ms, so the cost is per distinct shape.

Counting compilation events during one build gives the mechanism: **278 separate
XLA compilations**, led by 56 `jit(multiply)`, 36 `jit(broadcast_in_dim)`, 22
`jit(add)`, 18 each of `subtract`, `true_divide` and `convert_element_type`.
There is no `jax.jit` anywhere in that path, so every primitive is being
compiled on its own under op-by-op dispatch. This is the textbook cost of
unjitted `jnp` in a setup path, and it is paid before a single time step.

### 25e.2 Why the obvious fix does not apply

Wrapping the build in `jax.jit` fails twice, and both failures are informative.
`SpectralGrid` is not hashable, so it cannot be a static argument; and passing
it as a traced pytree raises `ConcretizationTypeError`, because the build
deliberately calls `float()` on geometry values to decide whether a twist-shift
cache is resolvable. Those are *structural* decisions made from concrete values,
not arithmetic, and they are the reason the module already carries an
`_is_tracer` helper.

So the fix is not one decorator. The candidates are to compute the one-shot
setup arrays with `numpy`, which removes XLA from the path entirely but must not
break the differentiable geometry route that reaches this build through
`objectives/core.py`; or to split the structural decisions from the array
construction and jit only the latter.

**That question is now answered by measurement, and it removes the constraint.**
Instrumenting `build_linear_cache` to record whether its geometry and parameters
arrive as tracers, then running the autodiff objective gradient tests -- the very
path the concern was about -- gives **traced = 0, concrete = 42**. The cache is
never built under a trace, not even when a gradient is being taken. The
differentiable geometry route gets its derivatives through the implicit
eigensolve VJP rather than by differentiating the cache construction.

So nothing in the differentiable path depends on those 278 primitive
compilations, and the `numpy` route is available: compute the one-shot arrays
with `numpy` and convert once at the boundary. The gate on that change is the
autodiff objective suite, which must still produce identical gradients, plus the
physics gates -- the point being that a value computed in `numpy` and handed to
the solver is the same value, and the only thing lost is 278 XLA compilations
nobody wanted.

### 25e.3 The FFTs are losing their thread pool, and the obvious fix is worse

Research into XLA:CPU turned up a mechanism worth verifying: an FFT thunk is
handed the intra-op thread pool only under some conditions, and an FFT fed by an
elementwise fusion appears to lose it. GKX multiplies by `i*kx` and `i*ky`
immediately before every `irfft2` in the bracket, so if true every transform in
the step is affected.

Verified on this machine at GKX's own shapes, `(2,4,8,49,96,48)` complex64:

| form | time |
| --- | ---: |
| `irfft2` of a jit **parameter** | 52.3 ms |
| `irfft2` of a **fusion result**, same values | 83.0 ms |

So the penalty is real and is about **1.6x**, against 5.4x reported on proxy
code elsewhere. That is very likely part of what the warm profile attributes to
"data movement": stalled wall time around the transforms rather than bytes moved.

**The obvious fix makes it worse.** Splitting the multiply and the transform into
two `jit` calls, so the transform is fed by a parameter, measures 89.5 ms against
66.3 ms for the fused form: materialising the 113 MB intermediate costs more than
the threading recovers. An earlier version of this measurement was wrong in the
other direction because the `jit` wrappers were constructed inside the timing
loop and retraced on every call; that error is recorded here because the
corrected numbers reverse its conclusion.

Any real fix therefore has to move the multiply to the other side of an existing
transform, not insert a new materialisation. The structural candidate is packing
the two derivative components into one complex array and taking a single c2c
inverse instead of a stacked pair, recovering the two real fields as real and
imaginary parts -- halving the transform count rather than rearranging it. That
was then measured, on GKX's own shapes:

| form | time | temp memory |
| --- | ---: | ---: |
| today: stack two components, one batched `irfft2` | 83.1 ms | 228.9 MB |
| packed: one c2c `ifft2`, components as real and imaginary parts | **58.0 ms** | 226.5 MB |

**That measurement was invalid and the technique does not apply here.** Checking
the identity rather than the timing settles it: recovering two real fields from
one complex inverse transform requires the **full** complex spectra of both. On
half spectra it is simply wrong -- maximum error 2.38 against a field of order
one, where the ordinary `irfft2` of the same half spectrum recovers the field to
4.8e-7.

The 1.43x came from comparing a stacked two-component `irfft2` against a
single-component `ifft2`: half the work, and the wrong answer. Correctness was
labelled unverified at the time, which was not enough -- the comparison was not
like-for-like, so the timing did not mean anything either.

The deeper reason is structural. Packing two real fields into one complex
transform is how a code that works in **full** complex spectra buys back the
factor of two that reality gives it. GKX already takes that factor by using
`rfft`, so there is nothing left to pack: one complex transform of a given size
costs about what two real transforms of that size cost. gyaradax report a gain
because their transforms are c2c throughout. **Do not implement this.**

### 25e.4 The array layout is the largest measured lever

The state is laid out `(species, l, m, ky, kx, z)` and the perpendicular
transforms run over axes `(-2, -3)`, so the Fourier axes are **not** innermost.
XLA transforms only innermost axes, so `jnp.fft` inserts a `moveaxis` before and
after every call. The profile attributes 7.9 per cent of step time to
`transpose_copy_fusion` before counting what was absorbed into larger kernels,
and finds that 18 of 25 transposes in the module come from three lines.

Measured directly, same element count and same transform:

| layout | time | temp memory |
| --- | ---: | ---: |
| current, `(..., ky, kx, z)` over axes `(-2,-3)` | 38.1 ms | 114.4 MB |
| proposed, `(..., z, ky, kx)` over axes `(-2,-1)` | **18.0 ms** | **57.8 MB** |

2.1x faster at half the temporary memory **on the transform alone**. That is
where the microbenchmark ends and the verification begins, because a research
thread had already reported the same isolated win evaporating inside a full
bracket.

It does evaporate. Measured on a bracket that reproduces the real op sequence --
two stacked derivative transforms, the product, the forward transform, the mask
and the Hermitian completion:

| layout | bracket time | temp memory |
| --- | ---: | ---: |
| current, `(..., ky, kx, z)` | 175.4 ms | 344.5 MB |
| proposed, `(..., z, kx, ky)` | 146.6 ms | 344.5 MB |

**1.20x, and the memory saving disappears entirely.** The transposes the layout
removes are a small part of a bracket dominated by the completion and the
transforms themselves, and the temporaries are set by the real-space product,
which the layout does not touch.

A second thing surfaced while building that comparison, and it matters more than
the timing. The change is **not a permutation**: `rfft2` halves the *last*
transformed axis, so moving `z` inward without care moves the half-spectrum from
`ky` to `kx`. The honest proposal is `(..., z, kx, ky)`, keeping `ky` last.

So the recommendation is now the opposite of what this section first recorded.
1.20x with no memory gain does not justify changing the state contract, which
the artifact writers, the restart format and the parallel decomposition all
depend on. **Do not do the layout change for performance.** If it is ever done
it should be for a different reason, and this measurement should be redone
end to end first.

### 25e.5 The 41.9 per cent has no contained fix

Two attempts were made to reduce the completion cheaply and both failed, which
is worth recording because the function is the single largest item in the
profile.

The index gather that reverses `kx` was replaced with a slice, a reverse and a
concatenate -- the change that gave 1.35x in `streaming.py`. In
`_complete_hermitian_ky` it gives nothing: 4.31 ms against 4.35 ms at
`(4,8,49,96,48)`, with byte-identical output. XLA lowers both the same way here
because the concatenate dominates and the gather fuses into it.

So the cost is the materialisation, not the indexing, and the materialisation is
intrinsic to the present representation: the bracket computes in half-ky because
it uses a real FFT, and it must return full-ky because the state is full-ky. The
only fix is to stop the round trip, which means carrying the state in half-ky
through the linear RHS. That is a state-contract change of the same class as the
layout change, and it must be justified end to end rather than on the profile
share -- the layout change looked like 2.1x in isolation and was 1.20x in a real
bracket.

The projector in `operators/nonlinear/projection.py` is a separate user of the
same helper, reached from the RK4 in `objectives/core.py` rather than from the
production step, and its truncate-then-re-expand is symmetry *enforcement*
rather than a representation change.

### 25e.6 A negative result worth not repeating

`_complete_hermitian_ky` rebuilds the full ky spectrum from the half spectrum
with a slice, a conjugate, a reverse, an index gather and a concatenate. The
gather has an algebraically identical `roll` formulation, and the two were
benchmarked on CPU at two grid sizes: identical outputs, and the roll is 4 to 9
per cent faster, which is inside the noise. **The gather is not the bottleneck
and XLA lowers it fine.** Recorded so the experiment is not repeated.

## 26. Risk register

| Risk | Consequence | Control |
| --- | --- | --- |
| File-count target creates giant modules | unreadable core | line ceilings, import graph, one-owner reviews |
| Aggregate coverage hides weak modules | false confidence | per-module branch coverage and mutation tests |
| Test consolidation loses detection power | silent regressions | fault injection and before/after mutation samples |
| API aliases persist indefinitely | hidden 1.x architecture | dated removal schedule and downstream search |
| External-code values become frozen truth | brittle or wrong tests | convert comparisons into independent/analytic evidence |
| JAX recompilation dominates workflows | poor usability | prepared objects, static topology, cache-miss gates |
| GPU tuning harms CPU or precision | device divergence | CPU/NVIDIA parity and matched-precision benchmarks |
| New IMEX effort repeats rejected work | code growth and delay | existing-owner-first, prospective time-to-accuracy gate |
| Geometry duplication returns | inconsistent drifts/gradients | VMEX ownership and adapter-only GKX policy |
| Quasilinear model is oversold | misleading transport | model tiers, domain card, uncertainty, OOD refusal |
| Nonlinear AD is treated as stationary sensitivity | invalid optimization claims | ensemble/directional/held-out gates |
| Long campaign tools remain installed | repository bloat | separate local/research workflow from package |
| Documentation becomes a research log | unusable user docs | Diataxis navigation; plans stay under `plan/` |
| Publication-style plots lack reproducibility | attractive but unauditable results | machine-readable sidecar, command, SHA, case metadata |
| Dependency floor becomes untrue | install failures | minimum/latest clean environments |
| Main is merged with red/cancelled CI | uncertain release state | required post-merge green run and branch protection |

---

## 27. Local workspace for the next agent

Recommended sibling repositories:

```text
workspace/
  GKX/
  VMEX/
  SOLVAX/
  booz_xform_jax/
  GX/
  stella/
  GENE/          # when access/build permits
  GS2/
  Pyrokinetics/
  SIMSOPT/
```

Record every revision before comparison. Use separate virtual environments when dependency stacks conflict. Do not edit companion repositories as part of a GKX refactor unless a specific missing upstream API blocks an approved GKX capability. When a companion change is required:

1. make the smallest upstream change;
2. add its own tests and documentation there;
3. record the exact dependency in the GKX work log;
4. avoid vendoring or copying the implementation into GKX.

External simulations and large outputs remain in ignored workspace directories. Store compact hashes, commands, and derived acceptance values in GKX only after review.

---

## 28. Reference set for implementation

Use primary sources and official documentation. Update `plan/references.md` when a source changes a model or acceptance gate.

### Gyrokinetic model and codes

1. Mandell et al., “GX: a GPU-native gyrokinetic turbulence code for tokamak and stellarator design,” *Journal of Plasma Physics* 90, 905900402 (2024), doi:10.1017/S0022377824000631.
2. Barnes, Parra, and Landreman, “stella: An operator-split, implicit-explicit delta-f gyrokinetic code for general magnetic field configurations,” *Journal of Computational Physics* 391, 365-380 (2019), doi:10.1016/j.jcp.2019.01.025.
3. Kim et al., “Optimization of nonlinear turbulence in stellarators,” *Journal of Plasma Physics* 90, 905900210 (2024), doi:10.1017/S0022377824000369.
4. Acton et al., “Optimisation of gyrokinetic microstability using adjoint methods,” *Journal of Plasma Physics* 90, 905900406 (2024), doi:10.1017/S0022377824000709.
5. Galletti, Volkmann, and Brandstetter, “gyaradax: Local Gyrokinetics JAX Code,” arXiv:2604.06085 (2026).
6. Artigues, Merlo, and Jenko, “iGENE: A Differentiable Flux-Tube Gyrokinetic Code in TensorFlow,” arXiv:2605.03086 (2026).
7. GENE-X official project and the 2026 stellarator extension, doi:10.1016/j.cpc.2026.110138.
8. Pyrokinetics official documentation and current supported-code interface.

### Collisions and closures

9. Frei et al., “Development of Advanced Linearized Gyrokinetic Collision Operators Using a Moment Approach,” arXiv:2104.11480.
10. Frei, Ernst, and Ricci, “Numerical Implementation of the Improved Sugama Collision Operator Using a Moment Approach,” arXiv:2202.06293.
11. Frei, Hoffmann, and Ricci, “Local Gyrokinetic Collisional Theory of the Ion-Temperature Gradient Mode,” arXiv:2201.02860.
12. Jorge, Frei, and Ricci, “Nonlinear Gyrokinetic Coulomb Collision Operator,” arXiv:1906.03252; use as a full-f/out-of-scope and algorithmic reference.
13. Abel et al., “Linearized model Fokker-Planck collision operators for gyrokinetic simulations,” arXiv:0808.1300.

### Verification and research software

14. Salari and Knupp, *Code Verification by the Method of Manufactured Solutions*, Sandia report SAND2000-1444, doi:10.2172/759450.
15. Oberkampf and Roy, *Verification, Validation, and Uncertainty Quantification in Scientific Computing*, 2nd ed., especially exact and manufactured solutions (2025).
16. FAIR4RS Working Group, “FAIR Principles for Research Software,” doi:10.15497/RDA00068.
17. Diataxis documentation framework, https://diataxis.fr/.
18. SIMSOPT documentation and JOSS paper for modular objective/optimization design.
19. Peter Yang, `no-ai-slop`, for concrete prose review patterns; apply as an editing checklist, not a scientific source.

### JAX

20. JAX official documentation: benchmarking, persistent compilation cache, buffer donation, explicit sharding, and `shard_map`.

---

## 29. Definition of success

GKX 3 succeeds when a new tokamak or stellarator user can install it, understand its model limits, run a checked and visibly progressing linear or nonlinear case, inspect convergence and uncertainty, save/restart/plot the result, and couple the same typed objects to scans or optimization without learning the internal solver tree.

For maintainers, success means that every physical term and numerical algorithm has one owner, one derivation, one public selection path, and proof-oriented tests; source, tests, and developer tools meet the hard budgets; and a failed experiment can be removed cleanly without leaving a permanent family of wrappers, reports, and compatibility tests.

For scientific use, success means that supported claims are tied to equations, invariants, convergence, literature, independent checks, statistical uncertainty, and measured CPU/NVIDIA performance. Unsupported regimes are named directly.

**Active next task:** Phase H0, PR H0-1 — replace the plan, regenerate exact current-head inventories, correct architecture targets, complete the PR ledger, and obtain a green post-merge `main` workflow. No solver or physics change belongs in that pull request.
