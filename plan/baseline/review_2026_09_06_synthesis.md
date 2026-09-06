# Synthesis review of the three GKX plans, 2026-09-06

Inputs: [#198](https://github.com/uwplasma/GKX/pull/198) at 71c5cc48,
[#203](https://github.com/uwplasma/GKX/pull/203) at 1643f0e7,
[#204](https://github.com/uwplasma/GKX/pull/204) at 29e559e0, main a99dac89,
the code and tooling they describe, the five research reports in
[plan/research](../research/), and new measurements listed below. The output is
[plan.md](../../plan.md) on this branch. An independent collaborator will write
the authoritative plan from these inputs; this document records the reasoning
so that the disagreements can be adjudicated on evidence.

## 1. What the three plans agree on

- The 2.0.0 end-damping regression (#192) must be fixed before any
  time-integrated claim is renewed.
- The QA 12.26% reduction is not statistically resolved (4/48 traces fail
  drift) and must not be promoted.
- The finite-k Coulomb tables are research-only; the public capability claim
  was wrong and is corrected.
- Checkpointed finite-window reverse AD is the one nonlinear derivative;
  shadowing and long-horizon adjoints are parked.
- Species×Hermite is the distributed route; whole-state sharding is rejected.
- The README must be shorter and user-facing; the plan must not be a logbook.
- No release until the open PRs have dispositions.

## 2. Where they disagree, and the resolution taken here

| Question | #198 | #203 | #204 | This synthesis | Basis |
|---|---|---|---|---|---|
| Damping contract | fixed rate (#202) as R0 exit | per-step (#197) is the exit; rate is R1 research | "compare conventions; do not choose by chronology"; concedes published GX uses per-step | Merge #197; rate behind a flag | GX eq. 4.30 and App. C; `damp_ends_amp/dt` in both linked and NTFT kernels of the local source; #197 bit-identical to the recorded artifact |
| Velocity-convergence anomaly | a finding | a defect hypothesis caused by damping | "damping does not prove every failure" | a defect until shown otherwise, with eight isolating tests | Hoffmann (16,8), GX (4,6)/(16,48); no published HL study shows −25% at Nl 24→32; the deck is collisionless and the hypercollision exponent is capped at 20 |
| Scope gate | ten lanes, all open | three tiers | full three-field kinetic-electron EM (EM0–EM5) required before release | EM required; EM0–EM3 must pass with published references; EM4 linear required, nonlinear reported without parity; EM5 linear/QL required, nonlinear AD parked | no published local nonlinear EM stellarator benchmark to match; iGENE needed 192 GPUs for kinetic-electron EM AD |
| Reference source | self-run GX first | GX with cap gate | matched external codes, self-built stella/GS2 | ranked sources; self-run GX last | GX ships six goldens; the open dataset gives 100,705 nonlinear references; the cap is confined to the linked kernel |
| Plan form | 1001 lines, logbook | 784 lines | 696 lines plus 490-line technical page | 513 lines with phases, steps, costs, and a ledger | user's request for context and steps; #204's structure retained where it is good |
| Process change | none | none | "one correctness change and one measurement per PR" | ledger, tiers, statistics module, ranked references, merge cadence | CGYRO/stella/GX practice; #193 already does this for one table |
| Novelty | adjoint nonlinear stellarator optimization as hypothesis | same, named | "do not claim first differentiable GK" | five unclaimed contributions with evidence bars and GPU-days; two chosen for paper 1 | landscape report |

## 3. What #204 got right and this plan keeps

- The API defects its audit found: `prepare` drops deck Nl/Nm; `warmup` is a
  no-op; the demo warns on CFL; a precision-specific test fails. These are P0.
- Splitting #202 into justified repairs and a policy experiment; preserving the
  two unpushed local commits.
- The EM0–EM5 decomposition, the three-mode configuration table (ES, reduced
  EM, full EM), the cross-channel covariance rule and the field residual
  definition. Steps are referenced, not duplicated.
- The warm-start experiment design and the rule that rejected trials must not
  mutate accepted state.
- The README's structure at 239 lines and its qualification of every example.
- Recording that stella's shipped nonlinear smoke deck exits zero with NaNs, and
  that gyaradax has 89 skips for missing reference data. Both are reasons to
  prefer published numbers and shipped goldens over self-built comparators.

## 4. What #204 leaves out, and this plan adds

- **The open GX dataset.** Not mentioned in any plan. Verified on disk and from
  Appendix B: 100,705 tubes, two nonlinear GX runs each, (4,8) at 64×64×96
  periodic, ν=0.01, hypercollisions, t=800, with a 100-tube resolution check at
  R²=0.993 and a periodic-vs-twist check at R²=0.97. The seven stored geometry
  functions are the eik functions GKX reads; the missing ones are derivable;
  the stored flux-surface average is reproduced to 1e-6 from the tensor. This
  gives nonlinear cross-code validation at scale with zero GX runs, a
  calibration set for the quasilinear model, and optimization targets with
  public equilibria (24,044 QUASR tubes).
- **GX's shipped goldens.** Six linear cases with decks and reference outputs
  and GX's own tolerance are already in the local checkout; the plan gates on
  them rather than on new GX runs.
- **Measured cost.** 1.14 s per step on an M3 Max CPU at the dataset
  resolution (30 steps 51.6 s, 330 steps 392.2 s, grid verified from the output
  file). No earlier plan had a compute budget.
- **A statistics protocol** with sources for each default, and the finding
  that the GK literature's error bars are mostly undefined or are fluctuation
  amplitudes, so the protocol is itself a contribution.
- **The landscape**: iGENE, gyaradax, GANDALF, JAX-in-Cell, yancc, DESC bounce
  averaging, the 2026 review; Kim 2024's numbers; Jorge 2024's f_Q trajectory;
  what is unclaimed.
- **Software norms**: CITATION.cff, Zenodo DOI, CONTRIBUTING.md, and the README
  elements peers have.

## 5. New measurements made for this synthesis

| Measurement | Result | How |
|---|---|---|
| GX absorber in source | `damp_ends_amp/dt` in `grad_parallel_linked.cu:391` and `grad_parallel_NTFT.cu:328` | grep of local bc2fe552 |
| GX launch clamp | `min(MAX_BLOCK_DIM_YZ, nb3)` only at `grad_parallel_linked.cu:173`; `MAX_BLOCK_DIM_YZ 65535`; periodic launches uncapped | grep |
| GX bgrad | `Geometry::calculate_bgrad` uses `GradParallel1D::dz1D` (FFT derivative) | source read |
| Dataset contents | 35 HDF5 items; 100,705 tubes; classes 51,075/12,795/12,791/8,235/15,809; fixed gradients a/L_T=3, a/L_n=0.9; Q median 8.28, max 876, no NaN | h5py |
| Dataset z grid | 96 uniform points, dz=0.7854, z ∈ [−37.70, 36.91], last point excluded; tube functions not periodic at the seam (B 1.4987 → 1.4880) | h5py |
| FSA convention | stored ⟨∇x⟩ = Σ(|∇x|/B)/Σ(1/B) over the 96 points to 1e-6 on four tubes | numpy |
| Appendix B | GX commit b88d763; nx=ny=64, nz=96, nhermite=8, nlaguerre=4, x0=y0=10, periodic, rk3 at 0.9 CFL, t_max=800, ν=0.01, T_i/T_e=1; 100-tube ×2/×10 resolution check R²=0.993/0.995, stability accuracy 0.994; periodic vs twist R²=0.97; ~8 A100-min per run; <28,000 GPU-h total | arXiv HTML, appendix extracted from raw HTML |
| GKX cost at dataset resolution | 1.14 s/step CPU (M3 Max), 90 ns per element-step; output dims x=64, y=64, theta=96, m=8, l=4 | runtime CLI, two run lengths |
| GKX eik reader | required profiles: bmag, gds2, gds21, gds22, cvdrift, gbdrift, cvdrift0, gbdrift0, jacobian, grho; scalars rmaj, aminor, q, shat, alpha; grouped GX output also accepted | `flux_tube.py` |
| Office box | unreachable on 2026-09-05 and 2026-09-06 | ssh, two attempts |
| PR state | #197 CONFLICTING on one manifest file, CI green; #196/#199/#200/#201/#202 clean; main unchanged since 2026-09-03 | gh |

## 6. Risks this plan carries

- The dataset comparison assumes the cvdrift0 = gbdrift0 approximation is
  harmless for the finite-pressure classes; the pilot tests it by restricting to
  vacuum QUASR tubes first.
- The GPU cost factor (10–15× over the M3 Max) is assumed; the pilot measures it.
- EM4 nonlinear has no matched reference; the plan reports it without a parity
  claim, which reviewers may still question.
- The office box has been unreachable for two days; Phases 2–5 cannot start
  without it or an alternative allocation.
- The ledger is a new mechanism; if it is not adopted, the plan degrades to
  #204's.

## 7. Not done here

No physics rerun beyond the timing runs; #202's code not reviewed; no new GPU
job launched; no README claim changed except the additions listed in the PR;
the docs pages from #204 are kept as they are.
