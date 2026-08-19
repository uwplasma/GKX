# GKX Research-Grade Plan

Working plan for turning GKX into a research-grade, differentiable, fast flux-tube
gyrokinetic code that designs and assesses stellarators in vacuum and at finite beta.
Baseline: branch `feat/bounded-memory-nonlinear-adjoint` @ `7cf5e6d1` (PR #48), after the
2026-08 hands-on assessment (five code surveys + ~20 local runs; findings referenced
throughout as "assessment").

---

## North-star goals

1. **One-command UX**: `gkx wout_XXX.nc` runs a sensible adiabatic-ITG nonlinear
   simulation on any VMEC/VMEX equilibrium, uses every local CPU/GPU automatically,
   stops itself when the saturated heat-flux mean is converged, and leaves behind a
   restartable NetCDF plus a complete, polished set of publication plots.
2. **Cross-code validated**: GKX matches GX (latest) and stella on a tokamak +
   multi-stellarator benchmark matrix spanning vacuum/finite beta, ES/EM,
   ITG/TEM/ETG/KBM, adiabatic/kinetic electrons, single/multi-species — flux tube only,
   with every reference regenerable by a script in this repo.
3. **Differentiable everywhere**: fast, FD-verified gradients for linear, quasilinear,
   and nonlinear objectives — including twist-shift boundaries, custom collision
   operators, electromagnetic and multi-species cases — and nonlinear heat-flux-window
   derivatives with respect to VMEC boundary coefficients through vmex.
4. **Fast with strong scaling**: competitive with GX per device on CPU and GPU, with
   multi-device sharding of a single nonlinear run (GX-style multi-GPU) and low
   gradient runtime/memory overhead.
5. **Slim and testable**: deliberate code only (no scaffolds, testbeds, proxies,
   experimental lanes), shrinking LOC/files/folders, with CI ≥ 95% coverage,
   literature-anchored, finishing in under 30 minutes.

## Guiding principles

- **Repository discipline.** All commits, pushes, and PRs are authored by
  `rogeriojorge` (the logged-in git identity) — never by Claude Code, codex, or any
  other tool identity, and no tool co-author trailers. Tools prepare changes;
  rogeriojorge commits them.
- **One gold standard per model.** Linear, quasilinear, and nonlinear each get one
  default solver/method that is accurate and fast in almost all cases; at most one or
  two alternates retained for documented edge cases. Everything else is deleted.
  **Decided:** the linear/quasilinear gold standard is the certified **matrix-free
  eigensolver** (implicit-VJP path) as default, with the IVP fit as the single
  fallback when certification gates reject a mode.
- **CPU and GPU are both first-class.** Every benchmark, parity table, scaling
  artifact, and gradient test carries a CPU row and a GPU row. The office machine
  (ssh, 2× RTX A4000) is the standing GPU host: GX/stella reference runs, GKX GPU
  runs, profiling, and the trial-study matrices all execute there; laptops cover the
  CPU rows.
- **A claim without a regenerating script is not a claim.** Every parity/validation
  number must be reproducible by a tracked tool from tracked (or scripted-download)
  references. No frozen literals.
- **Deleting code is progress.** Prefer fewer files, fewer folders, fewer lines —
  including tests. Consolidate rather than accumulate.
- **Examples read like vmex examples**: imports at top, explicit input parameters, build
  from the public API, run with verbose prints, then print/save/plot. No argparse in
  examples.
- **Physics honesty stays**: keep the fail-closed gate culture, but gates must guard
  runnable evidence, not frozen artifacts.

---

## Phase 0 — Hotfixes and hygiene (unblocks everything)

Small, independent, high-value fixes. Each lands with a regression test.

- [ ] **0.1 Fix the krylov eigensolver path** (assessment defect #1).
  `scan-runtime-linear` with `[scan] solver="krylov"` on
  `examples/linear/axisymmetric/cyclone.toml` returns damped/non-monotonic γ(ky) while
  `--solver time` is physical. Diagnose mode selection in
  `src/gkx/solvers/linear/krylov.py` (`dominant_eigenpair`: branch selection, shift
  strategy, growth-gap gates) against the time-solver answer at every shipped example.
  **Decision made:** the certified matrix-free eigensolver becomes the linear gold
  standard and default once fixed (IVP fit as the single fallback when certification
  gates reject a mode); demote or delete the remaining solver options.
- [ ] **0.2 Overflow-blind growth-rate fit** (defect #2). Any non-finite or
  overflowing trajectory (|signal| beyond ~1e30) must fail the fit loudly with a
  diagnosis (likely CFL/integrator instability), never report confident γ/ω. Add a CFL
  estimate + warning at startup for explicit/IMEX methods.
- [ ] **0.3 Robust IVP growth-rate estimates** (defect #3). The auto fit window can
  select 1.6 time units and estimates scatter ±10% with horizon. Fit on a
  stationarity-tested window of the instantaneous γ(t) trace, report a fit uncertainty,
  and warn when γ·t_max < ~5 (under-resolved growth). Verify against the GX Cyclone
  probe (γ=0.1018, ω=0.2868 at ky=0.3, Nl=16, Nm=48).
- [ ] **0.4 Fix `gkx --plot` crash on ky-scan outputs** (defect #5): wire the existing
  `scan_comparison_figure` into `plot_saved_output` for `kind="linear_scan"`.
- [ ] **0.5 Example/docs drift** (defect #8): add `[scan].ky` to
  `examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml`; remove or implement
  advertised-but-missing flags (`--outdir` in `docs/inputs.rst:303`, `--title` in
  `docs/outputs.rst:130`); fix duplicated `gkx` line in quickstart.
- [ ] **0.6 Declare dependency floors**: JAX minimum in `pyproject.toml` (the dense
  eigenvector-derivative path needs `lax_linalg.eig(..., enable_eigvec_derivs=True)`;
  11 tests fail on jax 0.9.2), plus `booz_xform_jax` as a real dependency (VMEC path is
  unusable without it). Same class of bug PR #41 fixed for Python.
- [ ] **0.7 Dev environments**: create a dedicated venv with the newest JAX
  (`.venv-jax-latest`) for the eigenvector-derivative path and GPU work, and keep the
  floor-version venv for compatibility testing. Document both in `docs/testing.rst`.
  Local editable install (`pip install -e .`) documented as required for examples.

**Exit gate:** all shipped example configs produce physical results with the default
solver; no known way to get a silent wrong number from the CLI.

---

## Phase 0.5 — Open PRs: assess, merge, and branch triage

Five PRs are open. Each gets a full assessment with the same rigor as the #48
assessment — run the code locally (CPU) and on the office GPUs, verify every claim in
the PR body against the diff, check CI, and only then merge. All merges are performed
by rogeriojorge. Proposed order (dependencies first, smallest risk first; rebase each
on `main` after the previous merge):

| Order | PR | Branch | Why this slot |
|---|---|---|---|
| 1 | [#46](https://github.com/uwplasma/GKX/pull/46) | `fix/auto-fit-window` | Directly addresses assessment defect #3 (fit-window scatter); assess whether it subsumes item 0.3 or 0.3 finishes on top of it. |
| 2 | [#47](https://github.com/uwplasma/GKX/pull/47) | `fix/residual-gate-precision-floor` | Gate correctness for the eigensolver certification — prerequisite for making the matrix-free path the default (0.1). |
| 3 | [#44](https://github.com/uwplasma/GKX/pull/44) | `fix/tf32-precision-audit` | Precision policy pinning; needed before any new GPU parity/scaling numbers are trusted (Phases 2 and 4). Verify on the office A4000s, not just CPU. |
| 4 | [#48](https://github.com/uwplasma/GKX/pull/48) | `feat/bounded-memory-nonlinear-adjoint` | This branch; merge after Phase 0 items and the assessment's pre-merge list (restore gradient-evidence generators, JAX floor, linked-boundary + multi-species/EM window tests). |
| 5 | [#45](https://github.com/uwplasma/GKX/pull/45) | `bench/gx-parity-matrix` | Largest and most valuable: six-case GX linear parity + `--isolate-shapes` two-device methodology (route overhead ≈ 1.0, scaling 1.92–2.01×, exact identity). Re-validate its GPU rows on the office machine **after** #44 lands and **after** the GX re-baseline (2.1) so the parity matrix is measured against latest GX once, not twice. Feeds 2.1 and 4.1 directly. |

Assessment checklist per PR: (a) does it still apply cleanly on current `main`;
(b) do its claims reproduce on CPU **and** GPU; (c) does it add net lines that Phase 5
would delete (if so, trim before merge); (d) does it change any tracked artifact
without the regenerating script (if so, block until scripted); (e) CI green after
rebase.

Branch triage (slim-repo discipline): the remaining `origin/*` branches
(`chore/pencil-fft-helpers`, `docs/readme-inline-movies`, `feat/dimits-linear-threshold`,
`feat/nonlinear-parallel-routing`, `feat/timestep-cost-diagnostics`, `fix/adaptive-chunk-host-diagnostics`,
`fix/gpu-numerics-tolerances`, `fix/gradient-parameter-units`, `fix/linked-mixed-species-hermite`,
`fix/objective-velocity-space-closure`, `fix/python-version-portability`,
`fix/sharded-reduce-observable-96`, `fix/shift-preconditioner-*`,
`perf/device-z-production-granularity`, `perf/observable-fused-transform`,
`perf/pencil-bracket-overhead`) are audited once: branches whose PRs already merged are
deleted; live work either gets a PR into this queue or is closed with a note. No
long-lived orphan branches.

**Exit gate:** zero open PRs older than the current phase; every remote branch is
either `main`, an active PR, or deleted.

---

## Phase 1 — CLI & UX: `gkx wout_XXX` end to end

### 1.1 Zero-config runs from an equilibrium file

- [ ] `gkx wout_XXX.nc` — CLI sniffs positional args: a NetCDF with VMEC/VMEX wout
  signature (attributes/variables, e.g. `xm`/`xn`/`rmnc`) is treated as an equilibrium;
  a TOML is treated as an input deck. `gkx input.toml wout_XXX.nc` uses that input deck
  with that wout (overriding `[geometry].vmec_file`). Optional explicit flags
  `--vmec/--vmex FILE` do the same; no flag is required.
- [ ] `examples/common_input.toml` — the default deck used for these runs: adiabatic-
  electron ITG parameters likely to run everywhere (a/LT=3, a/Ln=1, nonlinear on,
  hyperdissipation on, `boundary="fix aspect"`, sane grid/velocity resolution, default
  `torflux≈0.64`, `alpha=0`, auto device use, auto-stop on). The CLI copies the deck it
  ran next to the outputs for reproducibility (pattern already exists in the demo).
- [ ] `--linear` switch: same deck but a ky scan (default ky list), producing γ(ky),
  ω(ky) plots + eigenfunctions.
- [ ] Resolution/time overrides stay available (`--Nl --Nm --dt --steps ...`).

### 1.2 Auto-stop at converged saturation (new default)

GX does **not** auto-stop (fixed `nstep`/`t_max`; averaging is post hoc) — confirm while
on the office machine (§2.1), then make this a GKX differentiator:

- [ ] Implement on-the-fly stationarity detection on the heat-flux trace (and Φ² as a
  guard): rolling window mean with Sokal integrated-autocorrelation SEM (estimator
  already exists in `tools/campaigns/heat_flux_autocorrelation.py` — promote it into
  `src/gkx/diagnostics/`), stop when the windowed mean is stationary and its relative
  SEM is below target (default ~5%), with a spin-up exclusion and a hard `t_max` cap.
- [ ] Option name (**decided**): `[time] run_to = "saturation"` (alternative:
  `"t_max"`) with knobs `saturation_rel_sem`, `saturation_min_window`; CLI
  `--until-saturated / --no-until-saturated`. **Default: on** for nonlinear runs;
  `t_max` becomes the safety cap, not the target. Works with diffrax and fixed-step
  paths (chunked integration = natural check points; reuse the movie tool's chunking
  pattern).
- [ ] README + docs section describing the criterion, its knobs, and how the reported
  window/SEM appear in outputs and plots.

### 1.3 Plotting: automatic, complete, polished

- [ ] After **every** CLI run, produce the full figure set (fast, non-blocking failure):
  - nonlinear: Q(t) and Γ(t) with the measured averaging window shaded + mean±SEM
    annotated, W_φ(t), Q(ky) and Q(kx) spectra, Φ²(ky), Φ²(kx,ky) heatmap,
    zonal vs nonzonal Φ², x–y snapshot at outboard midplane, 3-D flux-tube snapshot,
    per-species fluxes when multi-species.
  - linear scan: γ(ky), ω(ky) (+ reference overlay when available), eigenfunctions.
  - single linear: existing panel (|φ|² fit + eigenfunction).
- [ ] Promote the movie renderer out of `tools/artifacts/build_turbulence_movie.py` into
  `src/gkx/artifacts/snapshots.py` (`potential_real_space`, `render_frame`,
  `_field_line_tube`, `_torus_wireframe`) so CLI, movie tool, and docs share one
  implementation.
- [ ] `--movie`: fast preset (small grid or `snapshot_stride` on the main run), mp4 via
  ffmpeg when present, PNG frames fallback — matching the README look.
- [ ] Add `[output] snapshot_stride` writing time-resolved `PhiXY` (bounded memory) so
  movies/snapshots can be made **from a finished `.out.nc`** rather than re-running.
- [ ] `gkx --plot FILE` works on: GKX `.out.nc`, GKX sidecar summaries (all kinds,
  including `linear_scan`), and **GX output NetCDFs** (add a reader shim — field names
  from GX's diagnostics module; useful for cross-code comparisons anyway).
- [ ] QA pass on every figure: render each in CI at small size, check axes/labels/
  legends/finiteness (extend the existing figure-style tests), and visually review once.

**Exit gate:** on a fresh machine (with deps), `gkx wout_LandremanPaul2021_QA_lowres.nc`
runs to converged saturation on all local devices and writes the full plot set +
restartable NetCDF without any other input. README shows exactly this.

---

## Phase 2 — Cross-code validation: GX (latest) + stella, vacuum → finite beta

### 2.1 GX re-baseline (office machine, ssh)

- [ ] Update the office GX checkout to latest `main`; rebuild; record commit hash. Diff
  GX's release notes/source since the frozen reference revision `bc2fe552` for physics
  or normalization changes that affect tracked parity tables.
- [ ] Script the entire GX reference generation (input decks live in repo;
  `tools/benchmarks/run_gx_references.sh` on the office host) so references are
  regenerable; store GX outputs (or their windowed statistics + hashes) in-repo or in a
  DOI'd archive. Kills the "frozen artifact" reproducibility debt.
- [ ] Re-run the existing 6-case linear parity matrix + 5-case nonlinear window matrix
  against latest GX; update tables; investigate any drift. Every case runs three ways:
  GX on the office GPUs, GKX on the office GPUs, GKX on CPU — so parity and CPU/GPU
  self-consistency are measured from the same script.

### 2.2 Finite-beta geometries from vmex

- [ ] Generate with vmex and add to `examples/vmec/`: QA finite-beta (pressure only),
  QA finite-beta (pressure + current), and at least one QH/QI finite-beta equilibrium
  (inputs in repo; wouts gitignored + regenerated by `generate_wouts.sh`).
- [ ] Extend the VMEC→flux-tube geometry path to finite beta: close the
  "finite-beta pressure corrections and broad-equilibrium drift gates remain open"
  items in `src/gkx/geometry/vmec_boozer_core.py`; validate geometry arrays
  (|B|, curvature/grad-B drifts, gds*, jacobian) against GX's geometry module for the
  same wout/surface/α — this is a pure-geometry parity test, cheap and decisive.
- [ ] Run GKX vs GX linear + nonlinear on these equilibria (ES first, then EM at the
  equilibrium beta). Where differences exceed gates, dive into GX source
  (its geometry, field solve, and time-stepping) until the difference is *understood*
  and documented — resolution, normalization, or physics — not just measured.

### 2.3 stella as a second independent code

- [ ] `git clone` stella (stella-gk); build on the office machine; study source to
  extract: evolved equations/normalizations, velocity grid (it is mixed
  pitch-angle/energy — document the mapping to Hermite–Laguerre resolutions), geometry
  conventions, input namelist, output NetCDF layout.
- [ ] Write `tools/benchmarks/` converters: GKX-case → stella input deck (flux tube),
  stella output → the common windowed-statistics format used by the parity tools.
- [ ] Reproduce first: a published stella W7-X ITG case (linear + nonlinear heat flux)
  as the external literature anchor — the strongest public stellarator claim available.

### 2.4 The benchmark matrix (flux tube only)

Three codes (GKX, GX, stella) on a matrix; each cell = tracked case config + windowed
statistics + regenerating script. Start with rows that exist, add columns left→right:

| Geometry | vacuum/β | ES/EM | modes | electrons | species |
|---|---|---|---|---|---|
| CBC tokamak (s-α, Miller) | — | ES | ITG, ETG, TEM | adiabatic + kinetic | 1–2 |
| CBC tokamak | β scan | EM | KBM | kinetic | 2 |
| W7-X | vacuum | ES | ITG, TEM | adiabatic + kinetic | 1–2 |
| HSX | vacuum | ES | ITG | adiabatic | 1 |
| QA (LandremanPaul) | vacuum + finite β | ES + EM | ITG, KBM | adiabatic + kinetic | 2 |
| QH / QI (vmex) | finite β | ES + EM | ITG, TEM | kinetic | 2 |

- [ ] Close the known physics gaps in the process (they are rows of this matrix):
  - **Kinetic electrons / TEM**: current parity is unusable (γ errors up to 35×, ω sign
    flips, TEM branch anti-correlated). Debug the electron response/field solve on the
    simplest failing case with matched resolutions against both GX and stella;
    fix `tprim_e` handling foot-guns while there.
  - **KBM**: close the 20% linear discrepancy vs GX (currently the accepted-branch
    outlier); β scan around the documented case; then nonlinear KBM.
  - **Finite-beta stellarator EM**: after 2.2 geometry parity, one GENE-3D-style
    heavy-electron EM stellarator benchmark or stella EM case.
- [ ] Collision operators (GKX's differentiator): add one collisional ITG growth-rate
  and one collisional zonal-damping cross-code case (stella has Dougherty/Fokker–Planck
  options) so the operator hierarchy is exercised beyond operator algebra.

**Exit gate:** verification matrix shows three-code agreement (or understood,
documented differences) for every promoted row, with GKX CPU and GPU rows agreeing
within the pinned precision policy; no Open row among: kinetic-e TEM, KBM, finite-beta
QA; all references regenerable by script.

---

## Phase 3 — Autodiff: complete, fast, boundary-to-flux

### 3.1 Coverage (all cases differentiate)

- [ ] **Twist-shift boundaries**: `build_linear_cache` raises "traced magnetic shear is
  not supported with twist-shift boundaries" under `jit`/`grad`, so linked BCs — the
  standard sheared flux tube — cannot be differentiated. Fix by making the twist-shift
  policy static (shear is a geometry constant for a given case): resolve linked indices
  outside the trace and thread them as static aux data; keep a loud error only for
  genuinely traced shear (shear-as-design-variable, which stays unsupported for now).
  Regression: gradient test on CBC with `boundary="linked"`.
- [ ] **Custom collision operators** in the differentiable window:
  `nonlinear_heat_flux_window` must accept `collision_operator` (same physics as the
  forward run — today it silently drops it); gradient test w.r.t. collisionality.
- [ ] **Electromagnetic + multi-species gradients**: extend the window gradient tests to
  a 6-D (Ns) state and finite-β (`apar`/`bpar` wired but never tested under `grad`);
  CPU/GPU gradient agreement test at one shared size.
- [ ] **Quasilinear objective**: FD-verified gradient test of the full QL 6-vector via
  the implicit eigenpair path on at least one stellarator geometry (needs the newer-JAX
  venv from 0.7 until the floor is raised).

### 3.2 Boundary-coefficient derivatives with vmex

- [ ] Harden the `QA_optimization.py` chain (vmex implicit equilibrium derivative ∘ GKX
  window adjoint) into a tested API: `gkx.objectives` function taking a vmex problem +
  window config → value & gradient w.r.t. RBC/ZBS. FD-check at production window size.
- [ ] Wire the **linear (γ)** and **quasilinear** objectives through the same
  boundary chain (today they stop at VMEC internal state), so all three objective
  families differentiate w.r.t. boundary coefficients.
- [ ] Extend to finite-beta equilibria once 2.2 lands (vmex supports pressure/current).

### 3.3 Performance & evidence

- [ ] Targets: gradient ≤ 3× forward runtime (measured 2.1–2.7× on CPU — keep it, and
  measure the same ratio on the office GPUs), memory O(√N) block checkpointing (keep),
  compile time of the grad path ≤ 2× forward compile. Timing/memory artifacts carry
  CPU and GPU rows.
- [ ] Restore the deleted evidence generators (adjoint-vs-FD ladder across window
  lengths, checkpointing memory profile, divergence-knee measurement) as slim tools in
  `tools/benchmarks/` — the current headline numbers exist only as literals in a figure
  builder. Add a runtime guard/warning when a requested window exceeds the measured
  divergence knee (QA_optimization's 1024 sits one step below it).

**Exit gate:** one gradient test matrix (boundary/params × linear/QL/NL × ES/EM ×
linked/periodic × collisions on/off) green in CI, with timing/memory artifacts
regenerated by scripts.

---

## Phase 4 — Parallelization: one run, all devices, strong scaling

### 4.1 Study & design first (write `docs/parallelization_design.md`)

- [ ] Read GX's multi-GPU implementation in its source (how it decomposes: which axes,
  what collectives, NCCL usage, diagnostics handling) — we know it scales; steal the
  shape of the solution.
- [ ] Survey JAX distributed docs (`shard_map`, `jax.make_mesh`, `jax.distributed`,
  pallas where relevant) and how GENE/stella decompose (species/velocity first, then
  perpendicular) for contrast.
- [ ] Trial-study matrix on real hardware (logical-CPU mesh 2/4/8 + the two office
  A4000s, `--isolate-shapes` methodology from `bench/gx-parity-matrix` — without it,
  route overhead is misreported): candidate decompositions ky, kx, z, species, Hermite,
  species×Hermite, and hybrid (perp FFT axes replicated, velocity sharded). Record
  step-time, collective volume, identity error. Pick **one** production decomposition;
  document why; delete the losing lanes (Phase 5 depends on this).

Known starting points from the assessment: fused pencil-bracket route reaches ~2.0×/2
devices with exact serial identity (unmerged `bench/gx-parity-matrix`); the 118× scalar
observable overhead must be folded into the fused RHS (accumulate diagnostics while the
bracket is live); whole-state `pjit` and two-GPU Hermite sharding are dead ends as
measured (0.59×, 0.03×).

### 4.2 Implementation

- [ ] Promote the winning decomposition to the production integrator (full operator,
  not the reduced bracket: streaming/mirror/curvature/collisions/species included),
  gated by the existing identity ladder (state, RHS, conservation, transport window).
- [ ] Fused diagnostics reduction on-device (removes the 118× overhead), Neumaier
  compensation kept.
- [ ] **Auto-mesh on run start**: `gkx input.toml` detects visible devices (GPUs, else
  CPU cores via `xla_force_host_platform_device_count`), builds the mesh, shards the
  nonlinear state; `[parallel] auto = true` default with manual override. Print what it
  chose ("using 2 GPUs, ky-sharded, 2.0× measured step speedup at this size").
- [ ] Multi-host hooks (`jax.distributed.initialize` + SLURM env detection) designed but
  optional — single-node strong scaling is the deliverable.
- [ ] Strong-scaling artifact regenerated by script (1/2/4/8 logical CPU, 1/2 GPU) and
  tracked; scaling gate in CI (identity + minimum speedup on 4 logical CPUs).

### 4.3 Single-device speed (continuous)

- [ ] Kernel/XLA-level profiling on CPU and GPU: HLO dumps, `jax.profiler` traces to
  TensorBoard/perfetto, per-op cost attribution (reshape/broadcast/transpose debt:
  1545 reshapes/1822 broadcasts recorded on the GPU HLO), FFT layout tuning,
  operator fusion passes. Compare kernel-by-kernel with GX on matched Cyclone/W7-X
  cases to find where GX's nonlinear step wins.
- [ ] Cold-start: persistent compilation cache on by default
  (`JAX_COMPILATION_CACHE_DIR` under `.cache/gkx/`), trim `build_linear_cache`
  (gyro-Bessel + Laguerre caches dominate), target < 10 s to first step on CBC.
- [ ] Mixed precision policy audited (tf32 branches exist): x64 where physics needs it
  (Landau roots, conservation checks), f32 defaults elsewhere, documented.

**Exit gate:** `gkx common_input + wout` on a 2-GPU box runs ≥ 1.8× faster than
single-GPU at production size with bitwise-gated identity; CPU 8-core ≥ 5×; warm
nonlinear step within 1.5× of GX on matched cases, and faster where collisions are on.

---

## Phase 5 — Slim codebase, examples, CI

### 5.1 Deletion list (shrink LOC/files/folders)

- [ ] Remove the reduced analytic stellarator surrogate
  (`src/gkx/objectives/stellarator_reduced.py` — self-declared "slated for removal")
  and its example/test constellation once the real linear/QL example scripts (5.2)
  land.
- [ ] Delete the losing parallel lanes after 4.1 picks the production decomposition
  (velocity/species shard_map experiments, whole-state pjit, `device_z` reduced
  operator if not the winner) — `src/gkx/parallel/` currently carries five lanes with
  202 tests; keep production + one diagnostic route.
- [ ] Collapse the report/manifest machinery: the `nonlinear_gradient_evidence` /
  campaign-status/readiness/admission JSON-report stack (many files in
  `src/gkx/objectives/` and `src/gkx/diagnostics/`) shrinks to the few gates that guard
  runnable evidence.
- [ ] Merge single-use figure builders in `tools/artifacts/` into a small number of
  parameterized tools; delete builders whose only content is frozen literals (replaced
  by 3.3 regenerators).
- [ ] Fold `run-runtime-linear`/`scan-runtime-linear`/`run-runtime-nonlinear` into
  `gkx run` (auto-detect, as `run` already does) — keep the old names as hidden aliases
  for one release.
- [ ] Test consolidation: dedup the geometry/objectives coverage-test files, remove
  re-export contract tests made moot by the deletions, and fix the marker mess —
  default `pytest` must run the core physics tests (today
  `tests/unit/linear/test_linear.py` collects **zero** tests under the default
  `-m "not integration"`).
- Track the metric: LOC and file count reported in CI; goal is net-negative per phase.

### 5.2 Examples in vmex style

Pattern for every example (mirror uwplasma/vmex examples): imports at top → explicit
input parameters → build simulation objects from the public API → run with verbose
prints → print/save/plot results. No argparse.

- [ ] Rewrite existing `examples/` to this pattern (they are currently TOML + CLI
  invocations or argparse scripts).
- [ ] `examples/optimization/linear_growth_optimization.py` — boundary-coefficient
  optimization of γ (via 3.2 chain), QA_optimization workflow: mode ladder,
  seed-normalized weights, verbose stage prints, before/after plots.
- [ ] `examples/optimization/quasilinear_flux_optimization.py` — same workflow, QL
  objective (with the saturation-rule caveat printed honestly).
- [ ] Keep `QA_optimization.py` as the nonlinear flagship; add the divergence-knee
  guard; finite-beta variant once 2.2/3.2 land.
- [ ] `examples/common_input.toml` (from 1.1) documented as *the* starting deck.

### 5.3 CI: ≥ 95% coverage, < 30 min

- [ ] Coverage gate at 95% (codecov config exists) measured on `src/gkx` after the 5.1
  deletions (smaller denominator makes this honest, not cosmetic).
- [ ] Restructure to three tiers:
  1. **PR CI (< 30 min)**: unit + physics anchors (Landau, collision matrices,
     conservation, closure coefficients, gradient matrix from 3.1 at tiny sizes,
     identity gates) sharded wide, compilation cache warmed between shards, x64 on for
     the physics-anchor shard.
  2. **Nightly**: benchmark matrix regeneration checks (small members), scaling gate,
     GPU lane if a runner is available.
  3. **Weekly/manual**: full cross-code parity regeneration on the office machine.
- [ ] Every physics test cites its anchor (paper equation, exact root, or cross-code
  case id) in the docstring — literature-anchored, future-proof.
- [ ] Kill CI time sinks: measure per-test cost, cache `build_linear_cache` fixtures,
  drop redundant parameterizations; the 24-shard wide-coverage job becomes the PR lane.

**Exit gate:** `pytest` (default) exercises the physics; CI wall time < 30 min at
≥ 95% coverage; repo LOC and file count strictly below today's.

---

## Sequencing & dependencies

```
Phase 0   (days)  ────────────────────────────► hotfixes; unblocks defaults
Phase 0.5 (days–1 wk) PR queue #46→#47→#44→#48→#45; branch triage
Phase 1   (1–2 wk) CLI/auto-stop/plots  ◄─ needs 0.1–0.4
Phase 2   (continuous, 4–8 wk) GX rebaseline → stella → finite-beta → physics gaps
Phase 3   (2–3 wk) AD coverage ◄─ 0.6/0.7 (envs); boundary chain ◄─ vmex; finite-β ◄─ 2.2
Phase 4   (3–6 wk) study → implement → auto-mesh ◄─ profiling infra; feeds 1.1 defaults
Phase 5   (continuous) deletions ◄─ 4.1 decision + 5.2 examples; CI restructure last
```

Parallel tracks that never block each other: (2) validation, (3) autodiff, (4) performance.
(1) ships first for usability; (5) runs as a standing discipline with a deletion pass at
the end of every phase.

## Immediate next actions (this week)

1. Phase 0.5: assess PR #46 (it may subsume the 0.3 fit fix) and #47; queue merges in
   order — merges and pushes by rogeriojorge.
2. 0.1 krylov diagnosis → matrix-free eigensolver becomes default (blocks 1.x).
3. 0.2 + 0.3 (whatever #46 leaves) + 0.4 + 0.5 quick fixes — one PR.
4. 0.6 + 0.7 dependency floors + venvs — one PR.
5. ssh office: update GX to latest, record hash, confirm GX has no auto-stop, start the
   scripted reference re-run on the GPUs (2.1); clone + build stella (2.3); re-validate
   PR #44/#45 GPU claims there.
6. Draft `examples/common_input.toml` + wout-sniffing CLI (1.1) behind a small PR.
7. Write `docs/parallelization_design.md` skeleton and launch the 4.1 trial matrix on
   the office GPUs.

## Definition of done (research-grade)

- `gkx wout.nc` on a laptop and on a 2-GPU box: converged saturated Q with stated
  uncertainty, full plot set, restartable output — no other input.
- Three-code benchmark matrix green (or documented-understood) including kinetic
  electrons, TEM, KBM, finite-beta QA/QH; every number regenerable by a script, with
  CPU and GPU rows for every GKX entry.
- All history authored by rogeriojorge; no open PRs older than the active phase.
- Gradient test matrix green (boundary→flux included), ≤ 3× forward cost, O(√N) memory.
- Strong scaling on local devices with identity gates; warm step competitive with GX.
- ≥ 95% coverage in < 30 min CI; codebase smaller than today in LOC, files, and folders.
