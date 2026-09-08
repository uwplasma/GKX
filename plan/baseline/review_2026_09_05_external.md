# External review of the research roadmap (PR #198), 2026-09-05

Reviewed: `plan/research-publication-20260904` at `8b944c19`, against `main`
`a99dac89` (2.0.0 plus #193/#195). This review re-derives the plan's load-bearing
claims from primary sources and from the repository, and separates what was
verified from what was only read.

## Verdict

The plan's diagnosis is largely correct and unusually candid: it names the
damping regression, the unreproducible records, the collision-table errors and
the unsettled velocity convergence, and it refuses several tempting claims. Its
execution is wrong in six places that matter, two of them technical directions
that would carry a defect forward:

1. It backs a **fixed-rate** end damping (#202) as the R0 exit. GX's published
   contract is the per-step fraction `A = 0.1/Δt`, and its convergence study is
   in the product `AΔt`. The fixed rate is the mechanism of #192, not its cure.
2. It reports a **velocity-convergence failure** (Nl 24→32 moves γ by −24.7%)
   and plans around it as physics. Against the literature it is anomalous, and
   every run behind it is time-integrated at small `dt` with end damping active,
   which is exactly the #192 population. It is a defect hypothesis until re-run.
3. It records that GX's linked-boundary kernels **cap at 65,535 indices** and
   that even the baseline reference exceeds the cap, but does not make reference
   validity an R2 gate.
4. The README on the branch still says **"5, through gyrokinetic Coulomb"**
   while the plan's own R1 records 22.9–95.8% errors in the shipped finite-k
   Coulomb tables and labels them research-only.
5. It **omits landed evidence**: the examples-in-CI coverage, the first-run wheel
   gate on the release path, the linkcheck gate, the mirror rebuild on a solved
   equilibrium and the README parity gate are not in "current truth", and R5
   re-plans work #185 already did.
6. It is a **logbook wearing a plan's heading**: R0 is 236 lines with 51
   hash/PID/run tokens, R1 is 241 lines. Neither can be read as an order of work.

None of this changes the destination. All of it changes the next ten PRs.

## What was verified, and how

| Claim in the plan | Source checked | Result |
|---|---|---|
| GX end-damping contract | Mandell et al. 2024 (arXiv 2209.06731) §4.2 eq. 4.30 and Appendix C, read from the PDF | `d(ẑ)=A[1−2ẑ²/(1+ẑ⁴)]`, `z_width=L_z/8`, **`A=0.1/Δt`**; Appendix C scans `AΔt` and finds fluxes converged for `AΔt ≳ 0.05`. The product is the knob. |
| GX precision | same, §7 | "exclusive use of single-precision arithmetic". |
| GX resolutions | same, Appendix G and Tables 1–2 | Linear CBC: 3×2π, Nz=24, Nm=48, Nl=16, ν=1e-2, **no hypercollisions**. Kinetic-electron linear: Nm=128. Nonlinear CBC: converged at **(Nl,Nm)≥(4,6) with hypercollisions**, ≥(6,12) without; Nx=192, Ny=64, Nz=24, D=0.05, n=4, f_hyp=1, p=Nm/2. W7-X linear: 6×2π, Nz=256, Nm=16, Nl=8 — identical to GKX's `w7x_itg` fixture. |
| GX multi-GPU design | same, §7.1 | Species×Hermite only, halo m±2, field all-reduce; ≥75% strong-scaling efficiency to 4 GPUs. |
| GX provenance | `git rev-parse HEAD` and `git ls-remote origin HEAD` on office | both `3865a53778862e1686f414bf6f416339e24887c9`; two untracked Makefiles preserved. |
| Audit line counts | `find`/`wc` on `main` | src 184 files / 88,910 lines; tests 76 / 86,377 — exact. |
| Gyromoment convergence | Hoffmann, Frei & Ricci 2023 (arXiv 2308.01016), read from the PDF | Linear CBC γ converges for **(P,J) ≳ (16,8)**, optimum near P≈2J; nonlinear flux needs ~16 gyromoments at η_v=1e-3; Dimits threshold recovered for **(P,J) ≳ (12,6)**, κ_T≈4; GENE needs ≥100 velocity points. |
| Kinetic-electron convergence | Frei et al. 2023 (arXiv 2210.05799), abstract | Trapped-particle and drift-driven modes need more gyromoments than pressure-driven ones. Consistent with GX needing Nm=128 for TEM. |
| Differentiable GK landscape | arXiv 2605.03086, 2604.06085, 2511.21891, 2512.12160, 2603.11231 | **iGENE** (TensorFlow, Phys. Plasmas Aug 2026), **gyaradax** (JAX, GKW-derived, Apr 2026), **GANDALF** (JAX kinetic RMHD, Nov 2025), **JAX-in-Cell** (Dec 2025), and a March 2026 review of differentiable plasma physics that does not mention GKX. |
| Nonlinear stellarator optimization | Kim et al. 2024 (arXiv 2310.18842), read from the PDF; targeted searches | SPSA, two evaluations per gradient, ~70 iterations in three stages, 2–4× flux reduction, **~50% variation across field-line label α**, ι near 5/4 flagged by the authors, fluid (4,2) still gives ~2×. No published adjoint/AD-based nonlinear stellarator optimization was found; the linear adjoint is Acton et al. 2024 (arXiv 2403.12621). |
| Cited 2025–2026 papers exist | arXiv 2606.25399, 2505.00838, 2508.06116 | Wang & Zaki (June 2026), Thakur & Nadarajah (May 2025; Lorenz-63 and Kuramoto–Sivashinsky only), Merlo multiscale W7-X (Aug 2025) — all real. |
| Verification standards | search | No community MMS standard for flux-tube GK; GENE-X applies MMS to its grid implementation; MMS is stated unsuitable for long-time turbulence. The plan's R1 "independent forcing" is the right instrument. |
| Plan structure | per-section `wc` and token count | R0 236 lines / 51 tokens; R1 241 / 12; R2 74; R3 86; R4 29; R5 61; R6–7 83; R8 56; R9 47. |
| Strict docs build on the branch | `sphinx -W -b html docs` | passes. |

Not verified: full text of Frei 2021 and Mandell 2018 (abstracts only); the code
in #202; any physics rerun. Those limits bound the findings below.

## Findings, ordered by consequence

### F1 (P0) — The damping direction contradicts the parity target

R0's "Established" table reads "PR202 uses a fixed rate across audited routes",
and item 1 says of the nonlinear decks "their production RHS already uses a
rate. Preserve A for these decks." That rate is the regression: 79064c4d removed
`damp_amp/dt`, and #192 is its consequence. GX defines the layer with
`A = 0.1/Δt` and converges it in `AΔt`. A code that fixes a rate while its
comparator fixes a per-step fraction has no dt-independent parity: agreement at
one `dt` is disagreement at another by construction.

#197 restores the per-step contract and, verified this session on a free GPU,
reproduces the recorded `cyclone_salpha_itg` artifact **bit-identically** at all
11 ky. It is the R0 exit. #202 is a legitimate research question — a rate does
converge as `dt→0` where a per-step fraction does not — but it is an R1
migration behind an explicit flag, with rescaled decks and regenerated
references, not the release contract.

### F2 (P0) — The velocity-convergence failure is a defect hypothesis, not a result

R0 records Nl 16→24: −7.0%, Nl 24→32: −24.7%, Nm 128→160: +5.6% at Nl=32, and
concludes "velocity convergence still fails". Hoffmann et al. converge the same
case at (P,J) ≳ (16,8); GX converges the nonlinear flux at (4,6). A −25% swing
between Nl=24 and 32 at Nm=48–160 is not a property of the Hermite–Laguerre
basis on the Cyclone case in any published study.

Every run in that ladder is time-integrated (imex2, dt=0.002 or 0.001) with
end damping active — the population #192 breaks, with severity ∝ 1/dt. Until
the ladder is re-run on #197, it says nothing about convergence, and no
reference may be migrated from it. The plan should record this as the first
R0 action, not as a finding.

### F3 (P0) — GX reference validity is a gate, not a footnote

R0 documents that `GradParallelLinked` caps `dG_all.z` at 65,535 and
`dampEnds_linked` lacks the grid-stride loop, so at Nz=96/Nl=32/Nm=96 only
65,535 of 294,912 indices are updated, and even the baseline
Nz=96/Nl=16/Nm=48 (73,728) exceeds the cap. It then reports GX and GKX
disagreeing by ~4% at ky=0.55 and treats the discrepancy as open.

A reference produced by a kernel that skips three quarters of its indices is
not a reference. R2 needs an explicit gate: no parity row is promoted against a
GX output unless the cap is shown not to bind or a repaired build is used and
labeled as such. The repaired build already exists in the campaign scratch
root; it needs to become the declared reference, and the defect needs to be
reported upstream.

### F4 (P0) — The public collision claim contradicts the plan's own evidence

README (branch) row: "Collision models | 5, through gyrokinetic Coulomb". R1
(branch): shipped 8-moment test blocks err by 22.9%/84.1% at B1/B4, 18-moment by
13.0%/64.2%, field-phi2 blocks by 95.79%/86.40%; "both tables remain
research-only". The audit lists correcting this as P1. It was not done on the
branch. This is the failure mode that produced #173 and #178, one PR after
those were closed.

### F5 (P1) — Landed evidence is missing from "current truth"

Since the audit's SHA the following merged and are absent from the plan's
tables: #185 (16 of 36 examples execute in CI, two bugs found by running them),
#186 (nightly linkcheck; three unregistered DOIs corrected), #188 (the release
workflow runs the documented first run from the wheel before publishing — it
passed on the 2.0.0 wheel), #191 (mirror case rebuilt on a solved equilibrium,
Q_QL 0.998→0.217), #193 (README parity table recomputed from its scans, all
seven reproduce), #195 (release artifacts fail CI on staleness). R5's gallery
section re-plans the example coverage #185 delivered. The PR ledger lists them;
the plan body does not.

### F6 (P1) — The plan is a logbook

Run numbers, PIDs, SHA256 digests, scratch paths and "do not restart" notes
belong in `plan/log.md`. A reader cannot extract the order of work from R0 or
the collision contract from R1 without reading both end to end. Each section
should fit on a screen and point to the log for evidence.

### F7 (P1) — Novelty needs the 2026 landscape

The plan already rejects "first differentiable GK code". It should name what
exists: iGENE, gyaradax, GANDALF, JAX-in-Cell, and the March 2026 review in
which GKX does not appear. The defensible niche, stated as a hypothesis to be
earned in R7: a Hermite–Laguerre code with physical Coulomb collisions and a
VMEX/ESSOS geometry chain that performs **adjoint-based nonlinear stellarator
transport optimization with held-out validation**, against a field whose only
published nonlinear optimization (Kim 2024) is gradient-free SPSA with a
documented ~50% α-sensitivity. iGENE's own finding — useful directions only for
short windows, 15–50% of finite-difference magnitude — sets the bar for what
"useful gradient" must mean.

### F8 (P1) — `hsx_itg` is unreproducible

No HSX reference, GX deck or wout exists on the office box or upstream. The
README's HSX parity row (0.577%/0.273%) rests on a tracked CSV nothing can
regenerate. This is #178's category. R2 needs the same decision: regenerate
with provenance, or delete the row and the claim.

### F9 (P2) — R2 statistics: sound anchors, one addition

Oberparleiter, Vaezi–Holland, Flegal–Jones and Vats–Flegal–Jones are the right
anchors and the plan's caution about optional stopping is correct. Add: the
drive scan Vaezi–Holland require near the critical gradient must be predeclared,
and the multivariate ESS should be the stated estimator for the three coupled
diagnostics rather than an aspiration.

### F10 (P2) — R3 should state the comparator's decomposition

GX parallelizes species×Hermite only, with m±2 halos and a field all-reduce, and
reports ≥75% efficiency to 4 GPUs. GKX's measured whole-state two-GPU slowdown
against a historical 2.14× species/Hermite result is the same lesson; the plan
should say so and drop whole-state as a first candidate.

## Revisions made in this PR

1. R0: #197 is the release contract; #202 becomes an R1 migration behind a
   flag. GX eq. 4.30 and Appendix C cited.
2. R0 gains a first action: re-run the Hermite/Laguerre ladder on #197 before
   any convergence statement; no reference migrates from the old runs.
3. R2 gains a GX-reference-validity gate (65,535 cap) and an `hsx_itg`
   regenerate-or-delete decision.
4. README collision row corrected on this branch.
5. "Current truth" gains the six landed items.
6. R0 and R1 logbook prose moved to `plan/log.md` under dated headings, with
   pointers left in place.
7. R6–R7 novelty paragraph rewritten with the 2026 landscape.

## Not done here

No physics was rerun. #202's code was not reviewed. R6–R8 are left substantially
as written; they are reasonable and no contrary evidence was found. Full texts
of Frei 2021 and Mandell 2018 were not read.
