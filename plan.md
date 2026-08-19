# GKX Research-Grade Plan & Work Log

**This file is both the plan and the append-only work log.** It is written so that any
contributor — human, Claude, codex, or another agent — can pick up the work from
scratch with no other context: every decision, artifact, patch, host, repo, and open
question is recorded here or in `plan/` (same PR). Companion material:

- `plan/patches/` — ready and partial patches from completed/interrupted work (apply with
  `git apply plan/patches/<name>.patch` onto the stated base commit).
- `plan/notes/` — full study reports (stella, parallelization design, finite-beta,
  cross-code runs, GX office build makefile, wave status snapshot).
- `plan/decks/` — finite-beta VMEC input decks (destined for `examples/vmec/`).

## How to use this file (log protocol)

1. **Before starting work**: read §Context, §Decisions, §Work board; claim an item by
   appending a log entry (`CLAIM <item-id>`) and setting its board status to
   `in-progress (<who>)`. One item per claim; unclaimed partial work is fair game.
2. **After finishing (or stopping)**: append a log entry with what was done, numbers,
   file:line references, patch/branch names; update the board row. Never rewrite or
   delete old log entries — corrections are new entries.
3. **Log entry format**: `### YYYY-MM-DD HH:MM <who> — <item-id or TOPIC>` followed by
   short prose. `<who>` = initials or agent tag (e.g. `RJ`, `agent/krylov`).
4. **Authorship rule (hard)**: all git commits, pushes, and PRs are authored by
   `rogeriojorge` — never by Claude/codex tool identities and never with tool co-author
   trailers. Agents prepare diffs/patches; rogeriojorge commits.
5. Keep this file ordered: §Log is append-only at the bottom; the board reflects the
   latest truth; plan sections are edited in place only to reflect *decisions*, each
   logged.

---

## Environment & assets (verified 2026-08-18)

| Asset | Where | State |
|---|---|---|
| GKX repo | `/Users/rogeriojorge/local/GKX`, github.com/uwplasma/GKX | main checkout on `feat/bounded-memory-nonlinear-adjoint` @ 7cf5e6d1 (PR #48) |
| Work worktrees | `/Users/rogeriojorge/local/GKX-worktrees/{pr44,pr45,pr46,pr47,krylov,phase0,woutux,autostop,fitrobust,plots,planpr}` | krylov/phase0/woutux/autostop/fitrobust/plots hold **uncommitted** implementations (patches also in `plan/patches/`) |
| Laptop | macOS arm64 (Darwin 23.4), MacPorts python 3.11.14, **jax 0.9.2 CPU** | system `gkx` in site-packages is STALE — use `PYTHONPATH=<checkout>/src` or the venv |
| New-JAX venv | `/Users/rogeriojorge/.venvs/gkx-jax-latest` | jax/jaxlib **0.10.2**, gkx editable → main checkout, all 151 eig-derivative tests pass with `JAX_ENABLE_X64=true` |
| Office GPU host | `ssh office` = pop-os, 36 cores, **2× RTX A4000** (free), CUDA 11.5 + gcc-10.4 | GX re-cloned+rebuilt: `/home/rjorge/GX` @ **3865a537** (2026-08-11; ref bc2fe552 in history). Build: `make -j36 GK_SYSTEM=office` with untracked `Makefiles/Makefile.office` (copy in `plan/notes/gx_Makefile.office`). Flag decision pending: upstream now uses `-prec-sqrt=true`; office build kept `-use_fast_math` w/o it |
| stella | `/Users/rogeriojorge/local/stella` @ 4cdc5fcd (v0.5.1-240, 2023, **ES-only**; master 2b8e269f is 185 ahead w/ EM + new namelists) | binary `build_cmake/stella` rebuilt+validated (CBC ky=0.5 ω=0.18703+0.09703i bit-for-bit); run `/opt/local/bin/mpirun -np 4`. Build recipe in `plan/notes/stella_study.md` §2 |
| vmex | pip 0.5.0 + repo `/Users/rogeriojorge/local/vmex` | `vmex` CLI works; `from vmex import optimize` works |
| Finite-beta wouts | generated in session scratchpad (regenerate: `vmex plan/decks/input.LandremanPaul2021_QA_beta2*`, ~1 min each) | ⟨β⟩=1.91%/1.89%, converged fsq<1e-12; properties in `plan/notes/finite_beta_findings.md` |
| QHS wout | `examples/vmec/wout_NuhrenbergZille_1988_QHS.nc` (gitignored) | generated via `vmex input.NuhrenbergZille_1988_QHS` (455 s) |
| Session scratchpad (VOLATILE — dies with session) | `/private/tmp/claude-501/-Users-rogeriojorge-local/1e858e4f-...` | everything needed already copied into `plan/` |

Open PRs: [#44](https://github.com/uwplasma/GKX/pull/44) tf32 audit ·
[#45](https://github.com/uwplasma/GKX/pull/45) gx-parity-matrix ·
[#46](https://github.com/uwplasma/GKX/pull/46) auto-fit-window ·
[#47](https://github.com/uwplasma/GKX/pull/47) residual-gate floor ·
[#48](https://github.com/uwplasma/GKX/pull/48) bounded-memory nonlinear adjoint (the feat branch).

---

## North-star goals

1. **One-command UX**: `gkx wout_XXX.nc` runs a sensible adiabatic-ITG nonlinear
   simulation on any VMEC/VMEX equilibrium, uses every local CPU/GPU automatically,
   stops itself at a converged saturated heat-flux mean (`run_to="saturation"`), and
   leaves a restartable NetCDF plus a complete polished plot set. `gkx input.toml
   wout.nc` (either order) uses a specific deck; `--vmec/--vmex` optional aliases;
   `--linear` gives a ky scan with γ/ω plots; `--movie` a README-style movie;
   `gkx --plot <output>` re-plots existing GKX (or GX) outputs.
2. **Cross-code validated** against GX (latest) and stella on a flux-tube matrix:
   tokamak + several stellarators × vacuum/finite-β × ES/EM × ITG/TEM/ETG/KBM ×
   adiabatic/kinetic electrons × 1–2 species; every number regenerable by a script.
3. **Differentiable everywhere**: linear/quasilinear/nonlinear gradients — including
   twist-shift BCs, custom collision operators, EM, multi-species — and nonlinear
   heat-flux-window derivatives w.r.t. VMEC boundary coefficients through vmex.
4. **Fast with strong scaling**: GX-competitive per device (CPU & GPU); multi-device
   sharding of one nonlinear run; gradients ≤3× forward, O(√N) memory.
5. **Slim**: deliberate code only; LOC/files/folders strictly decreasing; CI ≥95%
   coverage in <30 min, literature-anchored.

## Decisions (settled — do not relitigate without a log entry)

- **Linear/QL gold standard = certified matrix-free eigensolver** (`method="adaptive"`,
  residual-gated, fail-closed), now wired as the generic runtime default by the krylov
  fix (`plan/patches/krylov_fix.patch`). IVP time-fit is the fallback.
- **Auto-stop**: `[time] run_to = "saturation"` (default for nonlinear) with
  `saturation_rel_sem` (0.05), `saturation_min_window`; CLI `--until-saturated`;
  `t_max` = hard cap. GX verified to have NO equivalent (fixed nstep + `.stop` file).
- **Parallel decomposition = GX-parity (species, hermite) 2-D `shard_map` mesh**;
  (ky,kx), Laguerre, z replicated. Full analysis in `plan/notes/parallelization_design.md`
  (§tradeoffs: ky/kx rejected — FFT all-to-all; z rejected — 4 z-nonlocal operators;
  Laguerre dead — dense full-l transform in grid mode). Diagnostics fuse into the scan
  carry (the measured 118× lesson). Sanity: halo/psum/`jax.grad` through shard_map
  bitwise-exact on 2 logical CPUs at jax 0.9.2.
- **JAX floor = 0.10.1** (`enable_eigvec_derivs` introduced there; **0.11 does not
  exist** — a `>=0.11` pin bricks installs). `jaxlib>=0.10.1`, `booz_xform_jax`
  becomes a declared dependency. Already in `plan/patches/phase0_bundle.patch`.
- **All commits authored by rogeriojorge** (see log protocol §4).

## Key findings a fresh agent must know (evidence in plan/notes/ and §Log)

1. **Normalization convention (OPEN — resolve first for any cross-code work).**
   GKX source uses `vth=sqrt(T/m)`, `rho=sqrt(T*m)/|q|` (`src/gkx/operators/linear/
   params.py:264-265`) — the GENE-like c_s family — while GX and stella use
   `vth=sqrt(2T/m)` (GS2 family). Empirically `ky_stella = sqrt(2)·ky_gkx`; after the
   remap, GKX↔stella CBC agreement is 0.1–3% at mid/high ky (`plan/notes/
   stella_vs_gkx_runs.md`). BUT repo docs claim GX-identical conventions, and the
   tracked GX parity tables claim sub-% agreement *without* an explicit remap —
   determine whether the parity tooling/normalization contracts (`[normalization]
   contract="cyclone"`) already fold in the factor, and write ONE definitive
   conventions doc (docs/normalization.rst) with a conversion table. All existing
   "GX ref γ=0.1018 at ky=0.3" comparisons in this file inherit this caveat.
2. **Time-solver fit bias (OPEN defect).** The IVP fit γ reads ~30% below the
   certified/dense eigenvalue on the collisional s-α cyclone lane at two resolutions
   (fit 0.070 @ t_max=80 vs dense 0.1027, reduced res), yet the stella agent's
   collisionless-Miller fits agreed with stella to 0.1–3% and converged upward with
   horizon. Both can't be universal: suspect collisional-lane specific fit pathology
   (signal choice/window/transient mixture). The certified eigensolver is the
   reference; root-cause the fit on the collisional lane (fitrobust item continues this).
3. **Krylov defect root cause** (fixed): generic runtime `solver="krylov"` dispatched
   to an ungated propagator branch (2×24-dim Arnoldi on ONE imex2 step) returning
   Rayleigh quotients at relative residual **0.9993**. Dense ground truth: ITG branch
   isolated, λ=+0.1027+0.1686j (reduced res). Fix + regression test in
   `plan/patches/krylov_fix.patch`.
4. **Finite-beta**: runtime wout path already carries pressure physics (verified
   cvdrift≠gbdrift with correct sign/magnitude on the new β=2% wouts); ONLY the
   differentiable vmex-state bridge (`src/gkx/geometry/vmec_boozer_core.py`) drops
   pressure — 4 precise changes listed in `plan/notes/finite_beta_findings.md`.
5. **GX multi-GPU architecture** (for phase 4): 1 rank=1 GPU; decomposes species→
   Hermite only; NCCL comms + m±1/m±2 halo (`src/moments.cu:~350-440`); field solve
   allreduce on m0 ranks (`src/solver.cu:130`). GX upstream drift since reference:
   bpar-CFL fix, g0 gyroaverage fix, FLR precision, `[Wspectra]` group REMOVED,
   pyvmec sign fix → re-baseline before trusting old parity numbers.
6. **Default pytest deselects the core physics tests** (`-m "not integration"` +
   integration-marked `tests/unit/linear/test_linear.py` etc. → 0 collected). CI's
   wide shards override. Fix in Phase 5.

---

## Work board

Status: `ready-to-commit` = finished patch in plan/patches, needs review+commit by RJ ·
`partial` = interrupted mid-work, partial patch + remaining steps listed ·
`todo` = not started · `done` = merged/landed.

| id | item | status | artifact | base |
|---|---|---|---|---|
| P0.5-44 | Assess PR #44 | **done — verdict: merge as-is** (2 nits, GPU checklist in §Log 08-18-a) | — | — |
| P0.5-46 | Assess PR #46 | **done — verdict: merge as-is** (0.3 remainder scoped) | — | — |
| P0.5-47 | Assess PR #47 | **done — verdict: merge as-is, before krylov fix** (2 caveats to ack) | — | — |
| P0.5-45 | Assess PR #45 | **partial** — all checks done except Cyclone repro; comparison-tool fix verified correct; verdict pending | worktree `pr45` | 0f860069 |
| 0.1 | krylov fix + adaptive default | **PR open: [#52](https://github.com/uwplasma/GKX/pull/52)** branch `fix/krylov-certified-default` | `plan/patches/krylov_fix.patch` (+165/−4, incl. regression test) | rebased onto main |
| 0.2/0.4/0.5/0.6 | overflow guard, --plot scan, drift fixes, dep floors | **PR open: [#50](https://github.com/uwplasma/GKX/pull/50)** branch `fix/phase0-robustness` (89 tests green) | `plan/patches/phase0_bundle.patch` | rebased onto main |
| 1.1 | `gkx wout.nc` UX + common_input.toml | **PR open: [#51](https://github.com/uwplasma/GKX/pull/51)** branch `feat/wout-cli` (42 CLI tests green) | `plan/patches/wout_ux.patch` | rebased onto main |
| 1.2 | auto-stop run_to="saturation" | **partial** (+559/−48; implementation ~complete, agent died at verification: byte-identical t_max check + saturation integration run + docs) | `plan/patches/autostop_partial.patch`, worktree `autostop` | 7cf5e6d1 |
| 0.3 | fit robustness (stationary windows, γ±stderr, warnings) | **partial** (+380/−7; agent died updating CLI print sites; must then verify vs certified eigenvalue per finding #2) | `plan/patches/fitrobust_partial.patch`, worktree `fitrobust` | 7cf5e6d1 |
| 1.3a | plot library (snapshots module, flux/spectra figures) | **partial** (+1266/−128; figures render, agent died fixing x-y title/colorbar collision, then final QA + tests rerun) | `plan/patches/plots_partial.patch`, worktree `plots` | 7cf5e6d1 |
| 2.1 | GX office re-baseline | **done** (rebuilt @3865a537; reference re-runs still todo → 2.1b) | `plan/notes/gx_Makefile.office` | — |
| 2.2 | finite-beta equilibria + geometry audit | **done (start)**; path-B fix → item 3.2b | `plan/decks/*`, `plan/notes/finite_beta_findings.md` | — |
| 2.3 | stella build + study | **done** | `plan/notes/stella_study.md` | — |
| 2.4-r1 | stella↔GKX CBC rung | **partial** — scans done, √2 convention found; final table + certified-krylov cross-check unfinished | `plan/notes/stella_vs_gkx_runs.md` + `make_comparison.py` | — |
| 4.1 | parallelization design | **done** | `plan/notes/parallelization_design.md` | — |
| 0.7 | latest-JAX venv | **done** | `~/.venvs/gkx-jax-latest` | — |

## PR queue (merge order; all merges by RJ)

1. **#46** → 2. **#47** → 3. **#44** (all assessed merge-as-is; rebase each on main after
   the previous; on #47 acknowledge: f32 floor is 119× looser on the always-complex64
   runtime path; GMRES inner-solve slack collapses at the floor — optional one-liner
   `1e2*eps` inner floor).
2. **#48** after adding: the two assessment pre-merge items still outstanding
   (restore deleted gradient-evidence generators; linked-boundary + multi-species/EM
   window-gradient tests — see §Phase 3).
3. **#45** last, re-validated on office GPUs after #44 and after 2.1b re-baseline
   (finish P0.5-45 first).
4. **New PRs from the board** (each = apply patch onto fresh branch off main after the
   queue lands, rebase, run stated tests, commit as RJ):
   `fix/krylov-certified-default` (0.1) → `fix/phase0-robustness` (0.2/0.4/0.5/0.6) →
   `feat/wout-cli` (1.1) → `feat/run-to-saturation` (1.2, finish first) →
   `fix/fit-stationarity` (0.3, finish first) → `feat/plot-library` (1.3a, finish first).
   Conflict notes: krylov vs #47 = one intentional duplicate helper block
   (`certifiable_residual_tolerance`) + krylov.py imports — keep either. fitrobust vs
   phase0 = same file `workflows/runtime/diagnostics.py`, different functions.
   plots vs phase0 = plotting.py (phase0 adds a `plot_saved_output` branch; plots patch
   is append-only) — apply phase0 first.

## Remaining steps for the partial items (exact)

**1.2 autostop** (worktree has full implementation): (a) rebuild pristine base copy,
verify `run_to="t_max"` 20-step run byte-identical to base; (b) run the shrunk cyclone
saturation case (32×32×16 Nl2 Nm4 dt=0.05 t_max=200 rel_sem 0.15) — confirm stop
before t_max with reported window/mean±SEM; (c) docs/inputs.rst [time] table + README
"Run to saturation" para; (d) unit tests for stopping logic already written — run them;
(e) regenerate patch.

**0.3 fitrobust**: (a) finish the two CLI print sites (γ ± stderr display); (b) run its
unit tests + the cyclone scatter matrix (t_max=10 stride 2/10, t≈30) comparing old vs
new window method; (c) **validate against certified eigenvalue** (γ=0.1027 reduced-res
dense; 0.0889 f32 full-res certified) not just self-consistency; investigate/report the
collisional-lane fit bias (finding #2); (d) regenerate patch.

**1.3a plots**: (a) shorten x-y snapshot default title (colorbar offset-text collision);
(b) rerun its plotting tests + movie-tool one-frame check; (c) QA all PNGs again;
(d) regenerate patch.

**P0.5-45**: rerun the PR's Cyclone parity tooling (t=150 protocol) — was running when
killed; then verdict + the office-GPU re-measure list. Note finding #1 (√2): explicitly
check whether the parity tooling remaps conventions.

**2.4-r1 stella rung**: finish `make_comparison.py` table (stella scan ky=0.2–0.7 vs
GKX √2-remapped batch scan vs GX tracked values); add one certified-krylov point via
worktree `krylov` src; deliver comparison.csv + writeup into `plan/notes/`.

## Phase plan (full detail)

### Phase 1 remainder — CLI & UX
- 1.3b: wire plot library into CLI (auto-plot after every run; extend `--plot` to read
  GX NetCDF via a reader shim), after 1.3a + 1.1 land. Auto-plot set: Q(t)/Γ(t) with
  shaded window + mean±SEM, Q(ky)/Q(kx), Φ²(ky)/Φ²(kx,ky), zonal-vs-nonzonal, x-y
  snapshot, 3-D tube, per-species fluxes; linear: γ(ky)/ω(ky) + eigenfunctions.
- 1.4: `--movie` fast preset (chunked capture; `[output] snapshot_stride` for
  time-resolved PhiXY in NetCDF so movies come from finished runs).
- 1.5: `[parallel] auto=true` default once Phase 4 lands.

### Phase 2 — validation
- 2.1b: scripted GX reference re-runs on office (`tools/benchmarks/run_gx_references.sh`
  to write; decide `-prec-sqrt` flag; regenerate the 6-case linear + 5-case nonlinear
  windows vs GKX with explicit convention handling per finding #1).
- 2.4 ladder (stella): r1 CBC (finish) → r2 RH/zonal residual (stella `RH` regression
  vs GKX `benchmarks/runtime_miller_zonal_response.toml`) → r3 W7-X linear s=0.49 using
  stella's bundled `wout_w7x_standardConfig.nc` + `input.geometry` overwrite hook to
  force IDENTICAL geometry → r4 nonlinear CBC heat flux (stella `.fluxes` vs GKX
  window) → r5 kinetic electrons (stella nspec=2, me/mD=2.7e-4) → r6 EM (build stella
  master on office). Velocity mapping: nmu≥Nl exact; nvpa≈2–3×Nm; `flux_norm=false`.
- 2.5 physics gaps: kinetic-e/TEM parity debug (simplest failing case, matched
  resolutions, three codes); KBM 20%; finite-β EM stellarator after 3.2b.
- 2.6: reference data tracked or DOI'd; parity regenerable in one command.

### Phase 3 — autodiff completeness
- 3.1: twist-shift traced-shear fix (make twist policy static: resolve linked indices
  outside trace — `src/gkx/operators/linear/cache_builder.py:243`); custom
  collision_operator + EM (apar/bpar) + multi-species gradient tests for
  `nonlinear_heat_flux_window`; CPU/GPU gradient parity test.
- 3.2a: boundary-chain API hardening (vmex ∘ GKX window); 3.2b: finite-β pressure in
  vmec_boozer_core (4 changes in finite_beta_findings.md) with the two new wouts as
  A-vs-B parity fixtures; then wire linear/QL objectives through the boundary chain.
- 3.3: restore deleted gradient-evidence generators (adjoint-vs-FD ladder, memory
  profile, divergence knee — deleted in 612e1311/a7b41968; recover via `git show`),
  add window>knee runtime warning (QA_optimization WINDOW_STEPS=1024 sits at the edge).

### Phase 4 — parallelization (design DONE, see notes)
- 4.2: implement (species,hermite) shard_map on the full operator per design §5;
  fused in-carry diagnostics; identity ladder; auto-mesh; then office 2-GPU benchmark
  matrix (--isolate-shapes) with gates ≥1.8×/2-GPU, identity exact-or-≤1e-12.
- 4.3: kernel/XLA profiling vs GX (HLO census: 1545 reshapes/1822 broadcasts recorded);
  compile-cache on by default; cold start <10 s.

### Phase 5 — slim + CI (standing)
- Deletion list: reduced stellarator surrogate + its constellation; losing parallel
  lanes after 4.2; report/manifest machinery not guarding runnable evidence;
  single-use figure builders; CLI subcommand collapse into `gkx run`.
- Examples → vmex style (imports/params/API/verbose/plot; no argparse);
  linear + QL optimization examples mirroring QA_optimization.py.
- CI: fix marker deselection (finding #6); ≥95% coverage <30 min; 3 tiers
  (PR / nightly / weekly-office); every physics test cites its anchor.
- Track LOC + file count in CI; net-negative per phase.

---

## Log (append-only, newest last)

### 2026-08-17 RJ+agents — ASSESSMENT
Full hands-on assessment of `feat/bounded-memory-nonlinear-adjoint` (report PR #48
context): physics core validated locally (demo γ=0.0900/ω=0.2898; t400 cyclone
saturates Q=9.6±1.0; HSX VMEC chain end-to-end 72 s; AD vs FD 2.7e-10 linear /
7.7e-12 nonlinear window, grad 2.1–2.7× fwd). Defects found: krylov scan garbage,
overflow-blind fits, ±10% fit-horizon scatter, --plot linear_scan crash, no JAX floor,
twist-shift AD break, deleted gradient-evidence tools. CI green, PR #48 mergeable.

### 2026-08-18 RJ — PLAN v1
plan.md v1 written on the feat branch (superseded by this file). Decisions: matrix-free
eigensolver default; run_to="saturation"; authorship rule; PR queue #46→#47→#44→#48→#45.

### 2026-08-18 RJ+agents — WAVE 1+2 (10+6 agents)
- P0.5: #44/#46/#47 assessed merge-as-is (details in board + notes); #45 partial.
- 0.1 krylov root-caused & fixed (patch ready; collateral: time-fit low bias, finding #2).
- 0.2/0.4/0.5/0.6 bundle ready; 1.1 wout UX ready (real QHS smoke pass).
- 2.1 office GX re-cloned/rebuilt @3865a537 (was DELETED; 56-commit drift documented);
  no auto-stop in GX; multi-GPU arch mapped (finding #5).
- 2.3 stella rebuilt+validated; full study in notes. 2.4-r1 partial: **√2 convention
  finding (#1)**; after remap 0.1–3% agreement mid/high ky.
- 2.2 finite-beta: two β≈1.9% QA wouts (decks in plan/decks); runtime path has correct
  pressure drifts; vmex-state bridge does not (4-change list in notes).
- 4.1 parallelization design decided: (species,hermite) shard_map (notes).
- 0.7 venv: jax floor truth = **0.10.1** (0.11 nonexistent).
- 15:0x session limit killed 5 agents mid-flight (autostop, fitrobust, plots,
  stella-r1, pr45) — partial patches captured into plan/patches; remaining steps
  itemized in §Remaining steps. Plan v2 (this file) created on branch
  `plan/research-grade-roadmap` as its own PR.

### 2026-08-18 RJ — PR MECHANICS: three ready patches → PRs #50/#51/#52
All three ready patches were verified to apply cleanly on `main` (not just on the
feat branch), so each became an independent PR off main rather than a stacked one:
- **[#50](https://github.com/uwplasma/GKX/pull/50)** `fix/phase0-robustness` — items
  0.2/0.4/0.5/0.6. 89 tests green (test_cli.py + test_runtime_helpers.py, x64).
- **[#51](https://github.com/uwplasma/GKX/pull/51)** `feat/wout-cli` — item 1.1.
  42 CLI tests green; symlink single-sourcing of the default deck confirmed in place.
- **[#52](https://github.com/uwplasma/GKX/pull/52)** `fix/krylov-certified-default` —
  item 0.1. Re-verified the fix on a main-based branch: cyclone ky=0.3 gives
  **γ=0.088930, ω=0.280209** (was γ=−0.113 garbage), matching the diagnosis exactly.
Worktrees for these: `GKX-worktrees/{applytest,woutpr,krylovpr}`.
Merge order unchanged: #46 → #47 → #44 → #48 → #45, then #50 → #52 → #51.
(#52 conflicts with #47 only in the intentional duplicate `certifiable_residual_tolerance`
block; #50 touches `plot_saved_output` in plotting.py so it must land before the
plot-library PR, whose diff is append-only.)

### 2026-08-18 RJ — OFFICE GPU UNBLOCKED
`ssh office` confirmed available for GPU + GX work. Two agents dispatched:
- **office-gkx-setup**: stand GKX up on the A4000s (clone/pull, jax[cuda] venv at
  `~/.venvs/gkx-gpu`, GPU visibility), then a CPU-vs-GPU parity table against the
  laptop reference values (demo γ=0.089982/ω=0.289838; cyclone time-solver scan
  γ=[0.0168,0.0362,0.0632,0.0575,0.0244]; nonlinear-short Wg=4.06441e-4,
  Wphi=8.41601e-6, heat_flux(t=5)=3.4246564609; krylov-fixed ky=0.3 γ=0.088930),
  plus warm-step and cold-compile timings. Fetches PR branches #50/#51/#52 for
  GPU testing. This CPU/GPU agreement check is a standing plan gate.
- **gx-rebaseline (2.1b)**: run the rebuilt GX @3865a537 on the tracked Cyclone
  linear probe (tracked value from GX @bc2fe552: γ=0.101814, ω=0.286777) to measure
  whether the 56-commit upstream drift invalidates the frozen parity tables, extend
  to the other tracked linear cases where decks exist, and **settle finding #1
  (normalization)** by reading BOTH codes' definitions and reporting the exact
  ky/γ/ω conversions with file:line evidence. Outputs archived on office under
  `~/gx_rebaseline_20260818/` so later runs can diff.
Note for future GX work: the `[Wspectra]` input group was removed upstream, so old
GX decks carrying it may need editing before they run.
