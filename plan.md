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
- **JAX floor = 0.10.1** (`enable_eigvec_derivs` first shipped there; the in-code comment
  claiming 0.11 was wrong for the floor). NOTE 2026-08-18: jax **0.11.1 does now exist**
  for py312 and is what the office GPU box runs — the `>=0.10.1` floor remains correct and
  safe; only the earlier "0.11 does not exist" remark is superseded. `jaxlib>=0.10.1`, `booz_xform_jax`
  becomes a declared dependency. Already in `plan/patches/phase0_bundle.patch`.
- **All commits authored by rogeriojorge** (see log protocol §4).

## Key findings a fresh agent must know (evidence in plan/notes/ and §Log)

1. **Normalization convention — FULLY RESOLVED. Both earlier claims were right.**
   **GX is NOT in the GS2/stella family — GX shares GKX's convention.** GX
   `src/parameters.cu:1336-1338` defines `vt=sqrt(temp/mass)`, `tz=temp/z`,
   `rho2=temp*mass/(z*z)` — a line-for-line match of GKX `params.py:264-266`. GX halves
   the GS2-convention drifts on ingest (`src/geometry.cu:537,795`), so its stored geometry
   is the halved form and its netCDF has `gbdrift(0)=0.36=1/rmaj` not `2/rmaj`. GKX's
   importer reads GX grouped output verbatim and applies 0.5 only to root-level
   GS2-convention `.eik.nc` (`src/gkx/geometry/flux_tube.py:433-452`) — correct bridge on
   each branch. ky grids agree to 6e-8. `contract="cyclone"` is a pure no-op.
   **=> GKX <-> GX: NO remap (why the tracked parity tables are legitimately sub-%).**
   **=> GKX <-> stella: sqrt(2) remap (stella genuinely is GS2-family).**
   **=> GX <-> stella: sqrt(2) remap.**
   (An earlier working assumption that "GX and stella share conventions" was WRONG and is
   retracted; it briefly propagated into agent prompts.)
   Details of the stella side:
   **The conversion is one factor sqrt(2), in OPPOSITE senses on wavenumber and rates:**
   `ky_stella = sqrt(2)*ky_gkx` and `(gamma,omega)_stella[v_th/a] = (gamma,omega)_gkx[c_s/a]/sqrt(2)`
   (GKX: `c_s=sqrt(T/m)`, `rho_s=sqrt(Tm)/|q|`; stella/GX: `v_th=sqrt(2T/m)`).
   Verified three ways (`plan/notes/stella_vs_gkx_rung1.md`): ratio constant to ~1% of
   sqrt(2) while gamma varies x2.68 and omega x4.64; five alternative mappings falsified
   (next-best 28.6% error vs 1.2%); and confirmed fit-free via the certified eigenpair
   (R_gamma=1.42850, R_omega=1.42897, mutually consistent to 0.03%).
   **omega is the discriminating channel** for any convention audit — dropping the ky
   remap costs only ~30% in gamma (it is flat near peak) but 53.7% in omega.
   STILL OPEN for GX: the rung-1 lane (Miller, collisionless) is not like-for-like with
   the tracked `cyclone_mismatch_table.csv` lane (s-alpha, hypercollisional), so it
   cannot say whether the GX parity tooling folds in the sqrt(2). Needs a like-for-like
   s-alpha collisional GKX-vs-GX run (the office gx-rebaseline agent is on it).
   ORIGINAL NOTE:
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
2. **Time-solver fit bias — CLOSED. There was never a fit-machinery bias.**
   Root cause: horizon. The shipped `cyclone.toml` runs to `t_max=10` = **0.93 e-foldings**,
   inside the ringing transient (instantaneous γ climbs monotonically −0.42 → +0.102 through
   the whole run). At `t_max=80` (7.4 e-foldings) EVERY combination of window method and mode
   extraction agrees with the certified eigenvalue to **<0.6%**. The error is not even
   one-signed: too-short fits land 34% low, the old auto window could land 22% high.
   Secondary real effect: `cyclone.toml` and `cyclone_coulomb_collisions.toml` are the ONLY
   example decks pinning `mode_method="z_index"` (all others use `"project"`); a single
   z-slice is contaminated by a weak near-degenerate branch (γ scatters ±25% across z at
   t=30, late-time std 0.0046 vs 0.0003 for `project`). That, not collisions, is why the
   collisionless-Miller lane looked clean — the lane was mislabeled.
   Ruled out with direct evidence: signal choice (adiabatic electrons ⇒ n_i ∝ φ ⇒
   bit-identical fits) and collisions (nothing collisional touches the fitted quantity).
   SUPERSEDED NOTE:
   On the collisionless Miller CBC lane the time fit matches the certified eigensolve to
   **+0.30% in gamma** (0.13902 vs 0.138607 c_s/a) and errs the OPPOSITE way from the
   collisional lane, so the fit machinery is NOT generally broken. Hunt the pathology in
   the collisional s-alpha cyclone lane specifically (candidates: end-damping /
   kz-hypercollisions acting on the fitted signal, collisional transient in the window,
   z_index mode selection under s-alpha, phi-vs-density signal choice).
   ORIGINAL NOTE: The IVP fit γ reads ~30% below the
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
| 1.2 | auto-stop run_to="saturation" | **PR open: [#54](https://github.com/uwplasma/GKX/pull/54)** `feat/run-to-saturation` (129 tests green on main; stops at t=128/200) | patch `autostop.patch` | rebased onto main |
| 0.3 | fit robustness (stationary windows, γ±stderr, warnings) | **PR open: [#56](https://github.com/uwplasma/GKX/pull/56)** `fix/fit-stationarity` (109+279 tests green) | patch `fitrobust.patch` | rebased onto main |
| 1.3a | plot library (snapshots module, flux/spectra figures) | **PR open: [#53](https://github.com/uwplasma/GKX/pull/53)** `feat/plot-library` (38 tests green; 3 real-data collisions fixed) | patch `plots_lib.patch` | rebased onto main |
| 2.1 | GX office re-baseline | **done** (rebuilt @3865a537; reference re-runs still todo → 2.1b) | `plan/notes/gx_Makefile.office` | — |
| 2.2 | finite-beta equilibria + geometry audit | **done (start)**; path-B fix → item 3.2b | `plan/decks/*`, `plan/notes/finite_beta_findings.md` | — |
| 2.3 | stella build + study | **done** | `plan/notes/stella_study.md` | — |
| 2.4-r1 | stella↔GKX CBC rung | **done — PASSES** (\|Δγ\|≤2.8%, \|Δω\|≤1.9%, 6 ky points); √2 conversion triple-verified | `plan/notes/stella_vs_gkx_rung1.md` | — |
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

### 2026-08-18 agent/stella-r1 — 2.4-r1 CBC RUNG PASSES + conversion solved
GKX vs stella, collisionless Miller CBC, six ky points: **|Δγ| ≤ 2.8%, |Δω| ≤ 1.9%**
(≤0.5%/1.7% at every point with an adequate fit horizon). Rung 1 passes. The exact
sqrt(2) conversion is now established and triple-verified — see finding #1 above.
Certified-vs-fit on this lane: fit is +0.30% in γ vs the certified eigenvalue, which
**re-scopes finding #2** to the collisional s-α lane only (relayed to the fitrobust agent).
Certification note: the runtime builds complex64 states, so the certification tolerance
floors at 1.19e-4 (1000·eps) — a real ceiling on how tightly this path can certify.

**NEW DEFECT (0.3-adjacent): fixed-`t_max` scans under-report low-ky growth, one-sided
downward.** The controlling parameter is the e-folding count N = γ·t_max: N≈2–3 → 9–14%
low; N≈5–6 → 1.3–5.7%; N≈6–10 → ≤0.5%. **Size t_max by γ·t_max ≳ 7, not by wall clock.**
This should become the under-resolved warning threshold (supersedes the γ·t<5 guess) and
likely explains part of the measured horizon scatter.

**Rung 2 (Rosenbluth–Hinton) is scoped and ready** — `plan/notes/stella_vs_gkx_rung1.md`
§rung2. Both endpoints exist (stella `tests/regression/linear/RH/RH.in`, which already
uses the exact rung-1 CBC Miller geometry, and GKX `benchmarks/runtime_miller_zonal_response.toml`).
Three blockers: (i) the GKX zonal benchmark is Merlo Case III geometry, not stella's RH
CBC — needs the rung-1 Miller block substituted; (ii) **live trap in the repo**: kx carries
the same sqrt(2), so the tracked toml's `kx=0.05` is a *different physical kx* than stella's
`akx_min=0.05` (GKX equivalent is 0.0353553); (iii) stella needs `write_phi_vs_time=.true.`.
Why rung 2 is valuable: the RH residual is a pure ratio and therefore **convention-independent**
— immune to the sqrt(2) question entirely — with an analytic anchor ≈0.119 for the matched
geometry (ε=0.18, q=1.4). GAM frequency/damping are rates and convert as /sqrt(2).

Artifacts: `plan/notes/stella_vs_gkx_rung1.md`; regeneration scripts (`make_comparison.py`
rebuilds the table from logs without re-running, `verify_units.py`) in the session
scratchpad `stella_vs_gkx/`.

### 2026-08-18 agent/pr45 — NORMALIZATION RESOLVED + GX-parity fit protocol
Finding #1 is closed (see above): GX shares GKX's `vt=sqrt(T/m)` convention — verified in
GX source, in the drift halving on ingest, in the netCDF drift values, in GKX's importer
branching, in the ky grids (6e-8), and by `contract="cyclone"` being a no-op. So no remap
is needed GKX<->GX, and the sqrt(2) applies only against stella (and GX<->stella).

**The GX-parity fit protocol is now documented** and is the best-performing fit recipe in
the repo (sub-0.1% agreement) — a strong candidate default for item 0.3:
fixed window `[0.7*t_end, t_end]` with auto_window OFF; signal = complex phi at midplane
`z=Nz//2+1`; TWO OLS fits — `log|phi|` -> gamma and the UNWRAPPED PHASE -> -omega;
Nl=16/Nm=48; imex2 at dt=0.002; float64; plus a **half-integration-time probe** that
re-fits at `t_end/2` and gates a 5% "settled" flag. That probe is a ready-made stationarity
self-check. Relayed to the fitrobust agent, with the hypothesis that the collisional-lane
bias may partly come from `fit_signal="auto"` selecting density rather than midplane phi.
PR #45's own Cyclone reproduction (t=150, 75000 steps + half-time probe, ~82 min under
contention) was still integrating at the time of writing; verdict pending on that number.

### 2026-08-18 agent/autostop — 1.2 DONE, and a SEM convention bug fixed
Auto-stop works: the shrunk cyclone case (32×32×16, Nl2/Nm4, rel_sem 0.15) **stopped at
t=128 of t_max=200** (36% of the horizon saved) in ~85 s, reporting
`window=[64.05,128.0] heat_flux=35.92+/-3.85 rel_sem=0.107 tau_ac=4.24`, stationary,
window span 63.95 vs min_window 42.37 (=10·τ_ac). `run_to="t_max"` is **byte-identical**
to base on the diagnostics CSV (summary differs only in wall-clock fields); the default
saturation path at 20 steps is also byte-identical and correctly reports
"no measurable window".

**Real bug caught while finishing**: the stop criterion used `n_eff = n/(1+2τ/dt)`, a
convention that `gkx/diagnostics/analysis.py::_correlated_sample_stats` explicitly
documents as REJECTED — it double-counts the zero-lag term, returns n/2 for independent
samples, and overestimates the SEM by 22% at zero correlation. A run would therefore have
stopped on a SEM disagreeing with the SEM the post-hoc transport-window gates report for
the same window. Switched to the validated `n_eff = min(n, n·dt/(2τ))`.

Also: flipping the default routed tiny stub runs into the chunk loop and broke 4 existing
tests; fixed at the source (the stop condition returns None when the entire step budget is
below the minimum sample count — such a run can never reach a decision, so chunking it is
pure overhead) rather than by patching the tests. 303 tests green across the four suites;
ruff clean.

Known follow-up (deliberately not done here): `sokal_autocorrelation_time` in
`transport_windows.py` and `integrated_autocorrelation_time` in `analysis.py` are now
near-duplicate FFT estimators in sibling modules. Consolidating touches a numerically
validated gate path and inverts module layering — belongs in the Phase 5 slimming pass.

### 2026-08-18 RJ — SIX WORK PRs NOW OPEN
Implementation PRs opened off `main`, all authored by rogeriojorge, all with green local
tests: **[#50](https://github.com/uwplasma/GKX/pull/50)** phase-0 robustness (89) ·
**[#51](https://github.com/uwplasma/GKX/pull/51)** wout CLI (42) ·
**[#52](https://github.com/uwplasma/GKX/pull/52)** krylov certified default (γ=0.088930
re-verified on a main-based branch) · **[#53](https://github.com/uwplasma/GKX/pull/53)**
plot library (38) · **[#54](https://github.com/uwplasma/GKX/pull/54)** run-to-saturation
(129). Plus **[#49](https://github.com/uwplasma/GKX/pull/49)** = this plan.

**Suggested merge order** (pre-existing queue first, then new work):
`#46 → #47 → #44 → #48 → #45`, then `#50 → #52 → #53 → #54 → #51`.
Conflict notes: #52 duplicates #47's `certifiable_residual_tolerance` helper (keep either);
#53 is append-only in `plotting.py` and #50 adds the `plot_saved_output` linear_scan branch
there, so **#50 before #53**; #54 touches `docs/inputs.rst`, `cli.py` and `config.py`, which
#50 and #51 also touch — expect small textual conflicts, no semantic ones.
Worktrees backing these: `GKX-worktrees/{applytest,woutpr,krylovpr,plotspr,autostoppr}`.

### 2026-08-18 agent/gx-rebaseline — 2.1b: THE TRACKED GX REFERENCE IS MID-TRANSIENT
Run on office with the rebuilt GX @3865a537 using **GX's own shipped decks** (found at
`/home/rjorge/GX/benchmarks/linear/`, each with a `*_correct.out.nc` regression reference;
none carries `[Wspectra]`, so all ran unedited; **no HSX GX deck exists in either repo**).

**Two results, same binary:**
1. **The 56-commit drift is benign.** Re-running at `t_max=10` lands on the same step
   (2145, t=10.00213, dt=4.663e-3) that `capability_matrix.toml:9` records, giving
   γ=0.101840 / ω=0.286760 vs tracked 0.101814 / 0.286777 — **+0.026% / −0.006%**.
2. **But t=10 is mid-transient.** The same deck run to its own `t_max=150` settles at
   **γ=0.093049, ω=0.281991**, matching GX's shipped `_correct.out.nc` (0.093018/0.281990)
   to 0.03% and PR #45's `gamma_reference` column to 0.02%. γ RINGS before settling:
   0.0183 (t=4.7) → 0.1014 (t=9.3) → 0.0856 (t=14) → 0.1147 (t=18.7) → **0.0930 (t>50)**.

**Consequences — this reframes every parity comparison so far:**
- `docs/benchmarks.rst:156` presents a fixed-step smoke-probe reading as GX's "terminal
  diagnostic". Anyone comparing a converged solver against it sees a spurious ~8.6% γ gap.
  **This is a documentation defect to fix** (PR #45's matrix already uses the converged pair,
  which is a point in that PR's favour).
- The honest converged comparison at cyclone ky=0.3 is **GX 0.093049 vs GKX certified
  0.088930 = −4.4%** — a respectable cross-code agreement, not the 12.6% the tracked
  number implied.
- `examples/linear/axisymmetric/cyclone.toml` has `t_max=10`, i.e. γ·t_max ≈ **0.9** at the
  converged γ — an order of magnitude below the calibrated γ·t_max ≳ 7. The shipped example
  fits a ringing transient. Retire the "time fit reads 30% low" framing (it compared a
  short-horizon fit against a reduced-resolution dense eigenvalue — two confounds).
- Other converged cases vs PR #45's references: cyclone_miller Δγ +0.02%/−0.01%,
  kbm_miller Δγ +0.09%/+0.07% (KBM slightly larger, as expected — it is the EM branch the
  upstream bpar/g0 fixes touch). w7x_itg still running on office.

**Normalization independently confirmed as (d) "a premise was wrong"**, with two corrections
to the PR #45 agent's citations and one structural proof a ratio alone cannot give:
GX `src/parameters.cu:**1062-1065**` (not 1336-1338) has `vt=sqrt(temp/mass)`; and
GX `device_funcs.cu:3035` uses the **probabilists' Hermite ladder** `sqrtf(m+1), sqrtf(m)`
with no 1/√2 — valid only for weight `exp(-v∥²/2v_t²)`, i.e. `v_t²=T/m`; a GS2-family code
would carry `sqrt((m+1)/2)`. GKX `cache_arrays.py:60-61` + `streaming.py:474` use the
identical ladder. Empirically confirmed from the new run's output: `gbdrift(θ=0)=0.35999972
= 1/Rmaj`, not 2/Rmaj. Formulas (G=GKX/GX, S=stella): `ky_S=√2·ky_G`, `γ_S=γ_G/√2`,
`ω_S=ω_G/√2`, `t_S=t_G/√2`; **GKX↔GX is the identity**; tprim/fprim/β are convention-free.
The erroneous claim in `plan/notes/stella_study.md` has been corrected in place.

**Reproducibility gotcha for anyone running GX**: GX shells out to `python` (not `python3`
or `sys.executable`) for Miller/VMEC geometry, and the VMEC path needs `booz_xform`; both
fail with a misleading `Cannot open file *.eik.out`. Fix on office was a wrapper script at
`~/bin/python` pointing into the venv — a bare symlink does NOT work (breaks venv detection).
Worth reporting upstream.

Outputs archived on office under `~/gx_rebaseline_20260818/` (cyclone_salpha,
cyclone_salpha_t10 which regenerates the tracked pair in ~55 s, cyclone_miller, kbm_miller,
w7x_itg, plus `extract.py`), each keeping GX's `_correct.out.nc` alongside for three-way diffs.
Full report: `plan/notes/gx_rebaseline.md`.

**New work items this creates:**
- **2.1c** DONE → **PR [#55](https://github.com/uwplasma/GKX/pull/55)** `docs/gx-probe-convergence`
  (217 gate tests green). Both `docs/benchmarks.rst:156` and
  `benchmarks/capability_matrix.toml:9` now label the t=10 pair a smoke reading and give the
  converged pair alongside. Those were the only two places quoting 0.101814.
- **2.1d** raise `examples/linear/axisymmetric/cyclone.toml`'s `t_max` so γ·t_max ≳ 7
  (t_max ≈ 80–150), or make the auto-stop/warning machinery flag it. Coordinate with 0.3.
- **2.4-r0** an HSX GX deck must be authored — none exists in either repo, so the tracked
  HSX parity row has no regenerable GX side.

### 2026-08-18 agent/gx-rebaseline — 2.1b COMPLETE: one durable pattern
All five runnable cases finished (w7x_itg and kbm_miller landed after the previous entry):

| case | ky | Δγ | Δω |
|---|---|---|---|
| cyclone_salpha (t=150, converged) | 0.3 | +0.02% | −0.003% |
| cyclone_salpha (t=10 probe) | 0.3 | +0.03% | −0.006% |
| cyclone_miller | 0.3 / 0.4 | +0.02% / −0.01% | −0.002% / −0.003% |
| kbm_miller | 0.2 / 0.3 | −0.13% / +0.07% | −0.19% / −0.04% |
| w7x_itg | 1.6 / 1.0 | −0.003% / −0.014% | +0.0007% / +0.004% |
| w7x_itg | 0.3 | **−29.7%** | +1.5% |

**THE PATTERN — wherever the tracked table asserts convergence, the rebuilt GX reproduces
it to 1e-4 or better.** Both large discrepancies trace to reference values that were never
converged eigenvalues, NOT to the 56-commit code drift:
- Cyclone's 8.6% γ gap = the t=10 mid-transient probe (above).
- W7-X ky=0.3's 30% γ gap = a marginal low-ky mode that never settles at t_max=200 in
  either revision. PR #45's matrix already marks W7-X ky≤0.5 `converged=False`
  (ky=0.3 carries `gamma_half_time_shift = −0.63`, i.e. the value moved 63% between
  half-windows). Every W7-X mode the table marks `converged=True` (ky ≥ 0.6) reproduces
  to ~1e-4.

**So the actionable asymmetry is documentation, not physics**: PR #45 labels its
unconverged rows honestly; `docs/benchmarks.rst` presents the Cyclone t=10 probe as "its
terminal diagnostic" with no equivalent caveat. Fixing that (item 2.1c) is the whole of
the remediation. This also RETIRES the "GX upstream drift may invalidate the parity tables"
worry that motivated 2.1b — it does not.

(KBM correction: the final `.out.nc` values are −0.13%/+0.07%, superseding the plateau
readings +0.09%/+0.07% quoted in the previous entry.)

### 2026-08-18 agent/office-gkx — GKX RUNS ON THE A4000s; PR #52 BLOCKED ON GPU
**Environment recipe (office only had Python 3.10, where PyPI caps jax at 0.6.2 — below our
floor).** Solved with `uv` fetching a standalone CPython 3.12 into the user's home, no
system packages touched:
```
python3 -m venv ~/.venvs/bootstrap && ~/.venvs/bootstrap/bin/pip install -U pip uv
export PATH=$HOME/.venvs/bootstrap/bin:$PATH
uv python install 3.12 && uv venv --python 3.12 ~/.venvs/gkx-gpu
VIRTUAL_ENV=$HOME/.venvs/gkx-gpu uv pip install "jax[cuda12]>=0.10.1" pytest
cd ~/gkx-wt/main && VIRTUAL_ENV=$HOME/.venvs/gkx-gpu uv pip install -e .
```
→ jax/jaxlib 0.11.1, `[CudaDevice(0), CudaDevice(1)]`. cuda12 wheels work despite system
nvcc 11.5. **Gotcha: uv's editable install uses a metapath finder that BEATS `PYTHONPATH`**,
so branch switching needs a venv per worktree, not a path override. Worktrees at
`~/gkx-wt/{main,krylov}`; the pre-existing DIRTY `~/GKX` (275 changed paths on the feat
branch) was left untouched.

**CPU↔GPU parity gate: PASS.** Demo γ=0.089982/ω=0.289838 exact to all printed digits;
ky scan ≤2.8e-5; nonlinear short 2–4e-6 relative on Wg/Wphi/heat_flux (textbook f32).
**Reference correction**: the plan's `Wg=0.000406441` is a `7cf5e6d1` feature-branch value;
office GPU (0.000406695), office CPU (0.000406694) AND laptop CPU on main (0.000406694)
all agree — **re-baseline the Wg reference to 0.00040669**. Wphi and heat_flux reproduce on
main to 7e-7, so only Wg moved.

**BLOCKER — PR #52 hard-fails on GPU.** The certified adaptive path (which #52 makes the
default) dies after **19 minutes**: `residual=0.00337443 tolerance=0.000119209`. Root cause
is **TF32**: the A4000 is Ampere, so default f32 matmuls carry a ~10-bit mantissa, inflating
the residual ~28× above the `1000*eps(complex64)` floor. With
`JAX_DEFAULT_MATMUL_PRECISION=highest` the same run gives γ=0.088932/ω=0.280220 (matching
laptop CPU to 3.2e-6) in **28.7 s — 40× faster**, because the failing run burns every restart.
The gate is right; the arithmetic feeding it is not. → **#44 (tf32 audit) becomes a hard
dependency of #52**, and #52 needs its Krylov/residual contractions pinned in #44's idiom.
Fix agent dispatched; will push to the #52 branch and comment there.

**Bonus finding that strengthens #52's case**: main's *uncertified* krylov returns
γ=**−0.126120** on GPU and **−0.115960** on CPU for a mode whose true value is **+0.08893** —
a stable, WRONG-SIGN answer that differs between backends, unchanged by `highest` precision.
So #52 is fixing a genuine wrong answer, not adding hygiene.

**Timings (100-step nonlinear short case):**
| host | warm ms/step | cold compile | wall |
|---|---|---|---|
| GPU (1× A4000) | **20.3** | ~22.9 s | 27.6–33.8 s |
| office CPU (36 cores) | 403.1 | ~17.1 s | 59.7 s |
| laptop CPU | 341.8 | ~14.3 s | 52.6 s |

**~17–20× warm-step speedup**, linear scan 4.1×. But compile is 23 s of a 25 s integrator
wall, so GPU only wins end-to-end past ~70 steps → **JAX's persistent compilation cache is
the single biggest UX win available for the one-command goal** (promote it out of 4.3 into
Phase 1). Only ONE GPU is used today; nothing shards across both (Phase 4.2). The plan's
"65 s laptop CPU" figure is superseded by 52.6 s.
Full report: `plan/notes/office_gkx_setup.md`.

### 2026-08-18 RJ — 2.1c shipped as PR #55; #45 regeneration-command defect logged
Docs/capability-matrix now distinguish the GX smoke probe from the converged pair
(PR #55, 217 gate tests green; grep confirms those were the only two sites quoting 0.101814).

Separately, the PR #45 agent found a **regeneration-command defect cluster**: the command
documented in `docs/benchmarks.rst` and `benchmarks/results/manifest.toml` is a bare
`python tools/comparison/build_gx_parity_matrix.py`, missing the `PYTHONPATH=src` and
`GX_PARITY_REF_DIR` that `tools/benchmark_refresh_manifest.toml` correctly declares. Worse,
running all six cases in one process (as that command does) would contaminate
`gkx_peak_host_rss_mb`, because `_peak_rss_mb()` (`build_gx_parity_matrix.py:81-84`) reads
`ru_maxrss`, a per-process high-water mark that never resets. The tracked RSS values are
non-monotonic across cases (1340, 1409, 1335, 1303, 1954, 1548 MB), which proves the shipped
matrix was actually built case-by-case via `--cases`/`--merge` — i.e. the documented
regeneration command has never been the one used. Fix belongs with #45.

### 2026-08-18 agent/tf32-fix — #52 GPU BLOCKER CLEARED (commit 0be83ae1, pushed)
The GPU failure needed **one line**: `src/gkx/solvers/linear/krylov_propagator.py:85-90`,
the candidate lift in `dominant_eigenpairs_propagator_cached`, now carries
`precision=jax.lax.Precision.HIGHEST` in #44's exact idiom.

Method worth reusing: the agent WALKED THE JAXPR of the whole certified path (descending
into sub-jaxprs, since `fori_loop`/`scan` park their equations there) and found it contains
**exactly one** matrix-shaped `dot_general` — that lift, which builds the very vectors
`certify()` and solvax's residual are measured against. Everything else is vector-shaped and
needs no pin (Arnoldi Gram-Schmidt vdots, the vmapped Rayleigh quotient and residual, the
overlap einsum, the `candidate_count==1` Ritz lift, solvax's residual). Notably
`_apply_operator` lowers to **zero** `dot_general`, so the RHS precision policy needed no
decision at all.

**Results**: GPU with `JAX_DEFAULT_MATMUL_PRECISION` UNSET now certifies —
γ=0.08893223851919174, ω=0.2802199721336365 in **29.1 s**, against the previous
RuntimeError at 1148 s. That equals the `=highest` result to every printed digit, which is
the strongest possible evidence the lift was the only TF32-sensitive site. CPU unchanged
(γ=0.088930/ω=0.280209) and the scan CSV is **bit-identical pinned vs unpinned** —
`Precision.HIGHEST` is a no-op where there is no TF32 path, so the pin costs nothing on CPU.
GPU↔CPU agreement 3.2e-6 (γ) / 7.4e-7 (ω). Tests 159 passed / 4 failed, all four
pre-existing f32 failures confirmed by stashing.
New test `test_certified_candidate_lift_requests_exact_dot_precision`
(`tests/unit/solvers/test_linear_krylov_core.py:724`) — in a file #44 does not touch, so
zero conflict; verified non-vacuous.

**MERGE ORDER CHANGE**: #52 and #44 pin the same line with byte-identical arguments (only
the comment differs), so they do not disagree. Recommend **#52 BEFORE #44** — #52 is the
release blocker and should not sit behind an audit PR; #44 then rebases with its krylov hunk
collapsing to a comment-only change. #44 is still needed for its second unrelated pin
(`geometry/sensitivity.py` normal equations), its repo-wide shape-classified ratchet
(strictly stronger than the path-specific test added here, which becomes droppable once #44
lands), and its CI/manifest wiring.
Nit filed on #44: its prose calls the lift "a matvec at the default `candidates=1`", but the
jaxpr keeps the size-1 free axis, so it lowers matrix-shaped `(1,k)·(k,n)` even there — the
pin is right, the sentence understates it.

### 2026-08-18 RJ — CI caught two real defects in the new PRs
Opening the PRs against real CI surfaced two things local testing could not:

1. **Line budget** (`repo-hygiene`): `tools/package_architecture_manifest.toml` enforces a
   no-regression baseline on `installable_source_python_lines`, and every PR adding source
   must bump it **with a written reason** in the house style (a `# old -> new: why these
   lines exist` comment). Bumped on all five: #50 89815→89894, #52 →89913, #54 →90216,
   #51 →90055, #53 →90926. This is the mechanism that makes the Phase 5 slimming goal real
   — treat it as a design constraint, not a chore, and note that #53's +1111 is largely a
   MOVE (the movie tool shrinks by the same renderers).
2. **`jax>=0.10.1` publishes no Python 3.10 wheels**, so the `python-floor` job — which
   installs at the declared `requires-python` floor precisely to catch this class of lie —
   could not install the package. Resolved by moving the floor to **3.11** in
   `pyproject.toml` (dropping the 3.10 classifier) and in `.github/workflows/ci.yml`.
   This is the same defect class PR #41 fixed for the previous floor: a declared floor the
   dependency set cannot satisfy. **Consequence for the plan: GKX's minimum Python is now
   3.11**, forced by the differentiable-eigensolver jax requirement.
   (Also fixed a mypy narrowing introduced by the linear_scan plot branch: reusing
   `gamma`/`omega` across two branches of `plot_saved_output` made mypy adopt the
   non-optional type bound first. Renamed the scan locals.)

Standing instruction for future contributors: **open the PR early and read CI**, because
the line budget, the mypy gate, the python-floor job and the artifact/manifest checkers all
enforce contracts that a green local `pytest` will not.

### 2026-08-18 agent/fitrobust — 0.3 DONE (PR #56) + THE HEADLINE VALIDATION NUMBER
**GKX's certified eigensolver reproduces converged GX to +0.043% in γ and +0.008% in ω**
on the Cyclone s-α lane (GKX 0.093089/0.282015, residual 6.1e-5, converged; GX 0.093049/
0.281991 from re-running GX's own deck to t=150). That is the strongest cross-code linear
number GKX has, and it is ~150× tighter than the 6.8% "Cyclone ITG" mismatch the README
quotes — because the README figure compares against the t=10 transient probe (finding #1).
**The linear physics is validated; the old mismatch was a measurement artifact.**

Fit bias finding CLOSED (see finding #2 above) — horizon, plus a `z_index` mode-extraction
effect, neither of them the fit machinery.

Delivered in #56: stationary-window selection (longest late interval where instantaneous γ
is stationary, ≥2 growth times, warned fallback), AR(1)-corrected γ/ω stderrs + R² + a
half-horizon settled probe, all surfaced in the summary JSON and printed line
(`gamma=0.061743+/-0.000575`), and an under-resolved warning that names the horizon needed
("only 0.62 e-foldings … extend to t_max >~ 113"). Threshold is γ·t_max ≳ 7 per the stella
agent's calibration.
Bug fixed inside the selector: tolerance was `3σ + 0.1%·μ` but γ(t) is rolling-mean smoothed
first, so σ≈3e-6 made the band ~1e-4 and refused windows stationary to 0.5%; a 2% relative
floor is what makes the method work.
Honest gap: on fallback the loglinear number is still RETURNED (loudly warned), not withheld.

**NEW WORK ITEM 0.8**: fix the two decks — `examples/linear/axisymmetric/cyclone.toml` and
`cyclone_coulomb_collisions.toml` — to use `mode_method="project"` and a horizon reaching
γ·t_max ≳ 7 (t_max ≈ 80). Both currently ship settings that produce a wrong number by default.

### 2026-08-18 RJ — A SECOND, STRICTER CI CONTRACT: per-module complexity budgets
`repo-hygiene` enforces **two independent budgets**, and the second one is not obvious:
1. the repo-wide `installable_source_python_lines` no-regression baseline (bump with a
   written reason), and
2. a **per-module cap of 1000 lines**, plus explicit per-module baselines for files already
   over it, with "reviewed exceptions" required otherwise.

Three of the new PRs tripped (2): #53 `artifacts/plotting.py` 1510>1000, #54
`diagnostics/transport_windows.py` 1120>1000, #56 `artifacts/io.py` regressed 1088→1093.

**Design consequence — this is a good constraint, not an obstacle.** The right response to
(2) is almost never a bigger budget; it is a new module. Dispatched accordingly: #53's
figure functions move to `src/gkx/artifacts/transport_figures.py`, #54's stop machinery to
`src/gkx/diagnostics/saturation.py`, leaving the host modules at their pre-PR size. Only
#56 (5 lines of summary fields on a file with an explicit baseline) gets a documented bump.
**Guidance for contributors: append-only editing of a large module will pass local pytest and
fail CI — plan new surface as a new module from the start.** Note this interacts with the
earlier instruction to keep #53 append-only for conflict avoidance; that was right for
merge order but wrong for the budget, and the module split resolves both.

**PR status at this point:** #50 python-floor fix pushed (79caa79d, CI re-running);
#51 and #52 and #55 green; #53/#54/#56 module splits in progress.

### 2026-08-18 agent/pr45 — PARITY MATRIX REPRODUCES BIT-FOR-BIT ACROSS HARDWARE
The GKX side of PR #45's six-case matrix reproduces the tracked values to **1.9e-9 relative
across all 11 ky** — measured on macOS arm64 CPU against a matrix generated on an RTX A4000
GPU. **The parity protocol is deterministic and hardware-independent**, which is a stronger
property than the PR claims and is exactly what a regenerable reference needs.
Headline claim reproduced: peak ky=0.30 gives +0.0165% γ / +0.0095% ω (claimed +0.02%/+0.01%);
the small residual against the tracked +0.0196%/+0.0077% is on the REFERENCE side — this run
compared against GX's shipped `itg_salpha_adiabatic_electrons_correct.out.nc`, the PR against
its own GX re-run. Half-time convergence probe (backing the "10 of 11 settled" claim) still
integrating at time of writing; verdict pending on that alone.

Combined with the office re-baseline, the GX validation picture is now: the protocol is
deterministic across hardware (this entry), the references reproduce under a rebuilt GX
(2.1b), and where both sides are converged GKX's certified eigensolver agrees to **0.043%**
(PR #56 entry). The remaining GX-side gaps are documentation (#55, merged-pending), the
regeneration command defect, and the missing HSX deck (2.4-r0).

### 2026-08-18 agent/ci-budgets — module splits done; A THIRD CI GATE FOUND
#53: figure functions → new `src/gkx/artifacts/transport_figures.py`; `plotting.py` is now
**byte-identical to pre-PR** (816 lines, verified against fb559974). #54: stop machinery →
new `src/gkx/diagnostics/saturation.py`; `transport_windows.py` back to its exact 911 lines.
#56: `artifacts/io.py` per-module baseline 1088→1093 with a house-style reason (no source
change). All three: checker exit 0, tests green (38 / 33 / 145), ruff clean, pushed.

**THIRD GATE (document this for contributors):**
`tools/release/check_validation_coverage_manifest.py` (run by the docs job) **fails closed on
any `src/gkx` module with no coverage owner**. It had ALREADY been failing on #53 before the
split — `gkx.artifacts.snapshots`, added earlier on that branch, had no owner, so that PR's
docs job was red for a reason unrelated to the budget. Fixed by giving `gkx.artifacts.plotting`
ownership of both `snapshots` and `transport_figures`, and `gkx.diagnostics.transport_windows`
ownership of `saturation`; `docs/api.rst` `automodule` entries added for all three, since the
moved functions were previously documented through their old modules and would have silently
dropped off the API page.

**So the full contributor checklist for adding a new source module is: (1) repo-wide line
budget with a written reason, (2) per-module 1000-line cap, (3) a coverage owner in the
validation-coverage manifest, (4) a `docs/api.rst` automodule entry.** None of these is
caught by a local `pytest` run. Add this to `docs/testing.rst` in the Phase 5 CI pass.

### 2026-08-18 RJ — floor move followed through; ALL SEVEN WORK PRs GREEN
The 3.11 floor had a tail: `tomli` carries the marker `python_version < '3.11'`, so after the
bump it could never install, yet the floor job's own check imported it (and asserted 3.10).
Removed the now-unsatisfiable dependency and pointed the check at stdlib `tomllib`, which the
`gkx.utils.tomlcompat` shim already uses on every supported interpreter. A dead dependency
that CI was silently relying on — worth remembering when the floor moves again.

**State at handoff: #50–#56 all passing CI** (#50 last shard pending at time of writing).
Task #16/#17 of the session board are complete; the plan board above is authoritative.

### 2026-08-18 agent/pr45 — VERDICT: MERGE AS-IS. All five PR assessments now complete.
Reproduction landed (not stalled — the half-time probe took 1446 s under contention):
GKX side matches the tracked matrix to **1.9e-9 relative across all 11 ky** on macOS arm64
CPU against a matrix built on an A4000, and every half-time settled flag matches. Peak
ky=0.30: +0.0165% γ / +0.0095% ω (claimed +0.02%/+0.01%); settled 10 of 11, identical flags.

**One honest correction the PR should absorb**: the secondary figures "0.21%" and "0.6%" are
against the PR's own GX re-run; against GX's SHIPPED `_correct.out.nc` they are 0.25% and
0.7%. The excursions are on the REFERENCE side (GX uses adaptive RK4 and its late-window mean
over ~30 samples is noisy at marginal ky), not GKX's. The headline is robust; soften or
attribute those two.

**Fit protocol, fully specified** (candidate default for 0.3): signal = complex φ at outboard
midplane `z=Nz//2+1`, one ky at a time out of a batched multi-ky run; FIXED window
`[0.7·t_end, t_end]` with `auto_window=False`; two independent OLS fits (`log|φ|`→γ,
unwrapped `arg φ`→−ω), no weighting; stride capping samples near 2000 (563 points in-window);
Nl=16/Nm=48, Nz=96, imported GX geometry; `solver="time"`, imex2, fixed dt=0.002, float64;
settled gate = same fit at half integration time, ≤5% γ movement.
Two caveats if promoted: the estimator is ASYMMETRIC (GKX OLS over last 30% vs GX's mean of
its own instantaneous diagnostic over last 50%) — harmless when converged, and the source of
the marginal-ky reference noise above; and `min_points=80`/`require_positive=True` are passed
but **silently ignored** on this code path (`build_gx_parity_matrix.py:167-168`) — delete or wire up.

**New defects logged** (all minor, none blocking #45):
- `docs/normalization.rst` writes `ω/(v_th/R_0)` but NEVER defines whether v_th is sqrt(T/m)
  or sqrt(2T/m) — the exact ambiguity that caused this whole false alarm. **One sentence
  pinning it (and noting GX shares it, stella does not) is the highest-value doc change in
  the review.** → new item **2.7**.
- `docs/verification_matrix.rst:288-297` mixes units in one column (`+0.02%` beside
  `-5.6e-05`) — a 100× reading hazard.
- `contract="cyclone"` is DEAD SCAFFOLDING: every contract in
  `src/gkx/diagnostics/normalization.py:41-74` is identity and `diagnostic_norm="none"` is a
  passthrough. Candidate for the Phase 5 deletion list.
- **HSX parity row is not third-party reproducible**: no upstream deck, no shipped reference,
  and the flux tube needed "a locally patched copy of the GX geometry module" that exists
  nowhere in the repo. Tracking the six parity decks (esp. HSX + that patch) would close most
  of the regenerability gap cheaply → folds into item **2.4-r0**.
Office-GPU re-measure list recorded: the parity matrix is float64 so tf32 should not move it,
but the two-device bracket claims (route overhead, scaling, and especially the **bitwise 0.0
identity**) MUST be re-measured after #44/#52 pins; per-case risk mapped to each upstream GX
fix (g0/FLR → high-ky W7-X+HSX; bpar CFL → the two EM cases; pyvmec sign → W7-X/HSX geometry).

### 2026-08-18 agent/fbeta — 3.2b DONE (PR #57): the bridge had a MISLABELLED DRIFT
Not merely a missing pressure term. **The vmex-state bridge was returning the grad-B drift
under both names**: its `cvdrift` matched the runtime path's `gbdrift` to 6.7e-4. Because the
two coincide identically in vacuum, every existing test (all vacuum) was blind to it — so
every boundary-coefficient gradient ever taken at finite beta carried the wrong drift.

Parity (path B vs path A, normalized max abs): cvdrift 2.593e-1 → **2.995e-3** (s=0.25),
2.088e-1 → 3.252e-3 (s=0.64), 2.519e-1 → 7.796e-4 (+current) — 86×/64×/323×. Every other
array bit-identical. Residual 3e-3 is the bridge's own pre-existing metric floor (bgrad and
gds2 already disagree up to 4.7e-2 with no pressure involved), NOT missing physics.
Vacuum EXACTLY preserved (gbdrift−cvdrift ≡ 0.0). Gradient FD-consistent with clean
second-order convergence (2.6e-1 → 6.8e-3 → **7.0e-5** as h goes 1e-4 → 1e-6) on an
observable that is identically zero in vacuum, so it isolates the new terms.

**Change 4 (Hegna–Nakajima) deliberately NOT implemented — measured, not assumed.** Zeroing
`beta_b` inside path A moves drifts by 4.5e-4 to 8.1e-4, ~4× BELOW the 3e-3 parity floor, so
adding it would change adjoint gradients by an amount the gate cannot resolve. Number and the
cheap route to adding it later (Vprime cancels analytically in betamns_b; gmnc_b is already in
the booz_xform_jax output) recorded in `vmec_boozer_drifts.py`. Revisit when the metric floor drops.
Structure: `vmec_boozer_core.py` was at **997/1000** — three lines from the cap — so the drift
assembly moved to new `src/gkx/geometry/vmec_boozer_drifts.py` (core → 940), with coverage
owner + `docs/api.rst` entry. Line budget 89815 → 89964; `test_python_files` 96 → 97 (the
checker caught that second count too — a fourth CI contract to know about).
Coordinator note: the shipped test asserted exactness at rtol 1e-12/exact-equality, which
passes under CI's `JAX_ENABLE_X64=true` but FAILS a bare local `pytest` at float32 eps (1e-7).
Made the tolerances track `np.finfo(dtype).eps` so the suite is green both ways — the inverse
of the "green locally, red in CI" trap, and just as confusing.

### 2026-08-18 agent/stella-r2 — RUNG 2 PASSES + a currently-passing gate may be an artifact
**Rosenbluth–Hinton residual, the convention-independent check: stella 0.1050 ± 0.0010 vs
GKX 0.1021 ± 0.0010 = −2.8%** (analytic RH asymptote 0.1192; both codes on the same side,
−11.9% and −14.3%). Sharpest row, identical physical window and highest GKX resolution:
residuals differ by **2.5%**, GAM frequencies by **0.08%**.
GAM rates (which DO convert): ω ratio measured **1.4116–1.4131** vs √2=1.41421, ≤0.19% — an
independent confirmation of the rung-1 conversion. Two fit-free confirmations from the
geometry exports: stella's drift coefficients are **exactly 2×** GKX's and `kperp2` is
**exactly 2×** at matched physical kx.
Time axis subtlety worth recording: `t_stella = √2·t_gkx` (stella's a/vth unit is SMALLER);
getting this inverted inflates the apparent frequency error from 0.2% to 1.1%.
Three method systematics were measured and REMOVED rather than absorbed into tolerance:
stella's plateau decays secularly (reading it "at late time" is a 33% error on a converged
run), GKX can never reach the plateau (Hermite recurrence at t_rec = 2√Nm/k_par = 38 a/cs at
Nm=24), and the fit start matters (dropping the first GAM period costs −9 to −16%).

**DEFECT 1 — `Nm=24` is not converged, and a tracked gate may pass only because of it.**
The residual falls monotonically with Hermite resolution (0.11137 / 0.10838 / 0.10586 for
Nm=24/48/96 at fixed window), extrapolating in 1/√Nm to **0.10046** (0.15% fit). So the
tracked resolution reads **~11% high** — larger than the tracked gate's own tolerance
(0.015 on 0.19 = 7.9%). **`docs/_static/miller_zonal_response_pilot.json` (Merlo Case III,
residual 0.19245 vs published 0.19, currently one of the PASSING gate-index rows) sits at
exactly Nm=24.** If the same ~11% bias applies there, the converged value would be ≈0.173 and
the row would FAIL its own gate. → **NEW ITEM 2.8 (high priority): re-run the Merlo Case III
zonal artifact at Nm=48 and 96, extrapolate, and re-judge the gate.** `Nl` (≤0.3%) and `Nz`
(0.6%) are already converged, so this is a Hermite-only effect.

**DEFECT 2 — the kx trap is real but locally inert.** `benchmarks/runtime_miller_zonal_response.toml`
has `kx=0.05` with `Lx=125.6`, i.e. √2 too large physically. Measured effect here: residual
−0.4%, GAM ω −0.19%, GAM γ +4.4% — practically inert because both wavenumbers sit at kρ≈0.04–0.05
in the long-wavelength limit, but it MISLABELS the result and will bite in rung 3 where kxρ
reaches 0.30.

**Rung 3 (W7-X) prerequisites, now precise:** stella's `wout_w7x_standardConfig.nc` ships
(nfp=5, aspect 10.22, iota(s≈0.49)=0.8994 → q≈1.112) but has no VMEC regression case, so the
input is hand-built and `torflux` must be set to 0.49 (default 0.635). **`nfield_periods` is
its own trap**: default −1 resets to nfp (all five periods, NOT a flux tube), and stella then
multiplies `gradpar`/`b_dot_grad_z` by `nfp/nfield_periods` — so **stella's `gradpar` carries
the field-line-length convention and is not a pure geometric quantity**, while GKX's
`nperiod`/`ntheta` convention differs. The `input.geometry` hook contract: 3 list-directed
header skips (works only because stella's header has a blank third line) then fixed-width
`(13e12.4)`; `cvdrift0` is forced equal to `gbdrift0`; `gds23/24` have no GKX counterpart; and
**drift columns must be ×2 going in** (measured, not assumed). Note GKX's tracked W7-X toml is
NOT the target case — it points at a QI placeholder absent from `examples/vmec/` at
torflux=0.64. Recommended first step: smoke-test the geometry bridge on THIS CBC case, where
the answer (0.1050) is now known, before adding stellarator ambiguity.
Full report: `plan/notes/stella_vs_gkx_rung2.md`.

### 2026-08-18 agent/cliplots — 1.3b + compile cache DONE (PR #58, stacked on #53+#50)
The one-command goal is now closed end to end: every run writes its figures beside its
outputs, `gkx --plot` reads GX files too, and repeat runs stop recompiling.

**Compile cache — the measurement is the finding.** JAX persists only compilations longer
than 1 s by default, which is nearly useless here: a 16×16×32 nonlinear step compiles
**~320 kernels**, the fused `jit(scan)` is just 2.1 s of 13.5 s, and everything else is
~20 ms apiece — all refused by the default threshold. Keeping the default recovered under a
third of the compile. Storing everything costs 1.6 MB and gives:
compile 13.5 s → **0.9 s**; solver phase 23.3/24.6/23.4 s → **6.2/7.0/8.9 s**; small case
28.5 → 7.3 s. This matters most on GPU, where compile was 22.9 s of a 25 s wall — the reason
a GPU only won past ~70 steps. Keys are XLA's own, namespaced by `jax.__version__` so a
toolchain upgrade starts fresh. `jax_persistent_cache_max_size` is NOT settable via
`jax.config.update` on jax 0.9.2 (AttributeError) — eviction is `rm -rf`, documented.

**Pre-existing bug fixed**: the GKX NetCDF loader read `Diagnostics/t`, which NEITHER code
writes, so `gkx --plot case.out.nc` died on `KeyError: 't'` before drawing anything. NetCDF
plotting was simply broken.
**GX reading**: separate reader with all-optional variables (a GX LINEAR run carries none of
the `Wg_st`/`HeatFlux_st`/`ParticleFlux_st` that GKX's loader demands unconditionally);
files told apart by self-description (`code_info.value="gkx"` vs GX's `Title`/build
provenance), not filename.
**Containment done right**: matplotlib imported INSIDE the try (a broken backend fails at
import), plus an outer guard — a plotting failure costs a figure, never a completed run.
Escapes `--no-plots` / `[output] plots = false`.
Honest scope note: nothing on this base writes an averaging window yet, so window shading is
inert until #54 lands — deliberately not faked.
89 tests green here (371 across runtime+plotting, 180 in tools); both manifest checkers exit 0.
Line budget 90936 → 91752.

**Merge order addendum**: #58 is stacked on #53 and #50 — merge those first, then #58 shows
only its own commit. #54 (saturation windows) after #58 makes the window shading live.
