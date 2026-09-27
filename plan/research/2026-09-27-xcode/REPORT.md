# Cross-code benchmark record, 2026-09-27 (BUG-XCODE lane: VAL-REF, VAL-XCODE, VAL-KE, EM-B-PAR)

This record covers runs on the office host: 2x RTX A4000 and 36 cores, shared with other users. The load average was 40–100 while the runs were going, so wall times are not benchmark timings and are not cited as such.

Software:
- GKX at `main` `cf1d40828` (2.4.0), JAX 0.10.2, SOLVAX 0.26.0.
- GS2 8.2.1 and stella v1.0 (`058d98db`), the builds from the 2026-09-14 Q20 campaign (`plan/research/scripts/2026-09-14-cross-code-cyclone/`). That campaign's case generator `cases.py` and fitter `fit.py` are reused unchanged.

Units: every number is in GX/GKX units (Lref = a, vt = sqrt(T/m)). GS2 and stella values are converted with gamma_gx = sqrt(2) gamma_code and ky_gs2 = sqrt(2) ky_gx (Q20 manifest `[normalization]`).

Files:
- `scripts/`: case generators and run queues. They run from a bench directory given by `$BENCH`.
- `results/`: raw fits and extracts. `supervisors.txt` holds the start, end, rc and wall time of every grid-code run.

## 1. VAL-REF: GX goldens regenerated with the repaired end-damping build

Plan §3.7 voids four GX goldens because `dampEnds_linked` does not loop past the launch cap: `Nz*Nl*Nm = 73,728` exceeds 65,535.

The four decks were rerun with the labelled repaired build (`gx` sha256 `96a53403…`). It is GX `3865a537` with two changes:
- a grid-stride loop in `dampEnds_linked`;
- a rearrangement of the kz-hypercollision coefficient that avoids float32 overflow but is algebraically identical.

The inputs are the upstream decks, unchanged except for geometry. The Miller cases read the geometry file that the Miller module had already written for the originals (`geo_option = "eik"`), because the run harness has no Python for the geometry module. File hashes are in `results/gx_runs.txt`; upstream golden hashes are in `results/gx_golden_sha256.txt`.

Final-sample gamma of the repaired build against the upstream golden (`results/gx_repaired_96a53403.txt`, `results/gx_upstream_goldens.txt`):

| Golden | ky points | max \|Δγ\|/γ | at ky | excluding ky ≤ 0.1 | max \|Δω\|/ω |
|---|---:|---:|---:|---:|---:|
| Cyclone s-α adiabatic | 11 | 0.46% | 0.05 | 0.06% | 1.05% (ky .05) |
| Cyclone Miller adiabatic | 15 | 0.27% | 0.10 | 0.04% | 1.04% (ky .05) |
| Miller kinetic electrons (t = 40) | 7 | 43.5% | 0.10 | 4.7% (ky .7); 1.0% at ky .3; 0.00% at ky .5 | 0.69% |
| KBM Miller, A∥ only (t = 40) | 5 | 5.0% | 0.10 | 0.28% | 0.35% |

**Finding: the clamp does not explain the published disagreements.**
- For both adiabatic Cyclone goldens the clamp is immaterial: ≤ 0.06% above ky = 0.1, inside the 0.3% build floor recorded in §3.7's reproduction.
- The large low-ky differences in the two t = 40 decks come from the final sample of a mode that has not yet separated from the transient by t = 40. The two builds differ in float32 summation order, and this shows up only at the lowest ky, where the growth rate is smallest. It is not an end-damping effect.
- So the disagreements published against these goldens (6.83% s-α, 5.51% Miller, 20.0% KBM) cannot be attributed to the clamp.

**GKX against the repaired goldens.**
- The certified adaptive eigenpairs (Q20, Nl16/Nm48, residual ≤ 1e-13) agree with the repaired build to 0.05% at s-α ky .30 (0.093091 vs 0.093049) and at Miller ky .55 (0.125975 vs 0.125906).
- The shipped parity table `docs/_static/cyclone_mismatch_table.csv` is a time-trace fit. Against the same golden it is 8.3% high at ky .15 and 8.5% low at ky .20.
- The headline disagreement is therefore a property of that fitted table, not of the GKX eigenvalue. Section 5 confirms this at six more points: the certified eigenpairs are within 0.3% of the repaired build everywhere except Miller ky .15 (+1.45%).

## 2. VAL-XCODE: stella's 1.4x on Cyclone, term by term (circular Miller, ky = 0.30)

Every code runs the same physics (Q20 case M): stella on the r1 grid (converged to 0.1% against r3 in Q20), GS2 on r3, and GKX as a certified eigenpair.

| Variant | GS2 γ | stella γ | GKX γ (Nl8/Nm24) | GKX γ (Nl16/Nm48) |
|---|---:|---:|---:|---:|
| base | 0.12546 | 0.17481 | 0.12865 | 0.12585 |
| mirror off | — (no knob) | 0.17536 (+0.3%) | 0.11115 (−13.6%) | 0.10977 |
| both drifts off | 0.0124 at t = 1500 (growing, unsettled) | decays to t = 1500 | 0.02937 | 0.02791 |
| drifts and mirror off | — | φ vanishes | 0.01392 | — |
| stella mirror explicit | — | 0.17481 (identical) | — | — |
| stella mirror implicit, semi-Lagrange off | — | 0.17483 (+0.01%) | — | — |
| stella `vpa_max = vperp_max = 4` | — | 0.17661 (+1.0%) | — | — |

**Finding 1: stella's growth rate does not respond to its mirror term.**
- Removing the mirror term moves stella's γ by +0.3%. Switching the mirror to the explicit scheme reproduces the implicit result to every printed digit, and turning off the semi-Lagrange mirror moves it by 0.01%. Widening the velocity box to 4 moves it by 1.0%, so the velocity grid is not the cause.
- In GKX the mirror force is worth 14% of γ at this ky.
- A term the stella build does not respond to is the first concrete lead on the 1.40–1.47x excess.
- The excess is not a normalization. stella's ω is only 1.17x high, and a pure time or ky rescaling would move γ and ω together.

**Finding 2: the no-drift limit differs between codes.**
- With both drifts off, GKX keeps a weakly unstable slab branch: γ = 0.028, converged to 5% between Nl8/Nm24 and Nl16/Nm48.
- Run to t = 1500 code units, GS2 shows a growing branch too: γ ≈ 0.012, still drifting 9% over the last window, ω 0.053 against GKX's 0.130. stella's field decays through t = 1500.
- So two of the three codes have the slab branch. The GS2–GKX rate difference in this weakly driven limit is open, and stella's absence of the branch matches its insensitivity to the mirror term.

**Next step for the stella excess.** The mirror scheme and the velocity box are ruled out (table above). What remains is to compare stella's mirror coefficient array (`mirror` in `gk_mirror.f90`, built from `dbdzed` and `gradpar`) against GS2's and GKX's `bgrad` on this surface, and to run a stella case whose mirror is known to matter (a trapped-particle mode) to check whether the term acts at all in this build.

## 3. VAL-KE: Cyclone Miller with kinetic electrons (m_e/m_i = 2.7e-4, β = 1e-5 with A∥)

GS2 rungs, each run to 100 code-time units:
- e2: ntheta 32, nperiod 2, negrid 12, ngauss 6, Δt .02.
- e3: ntheta 48, nperiod 3, negrid 16, ngauss 8, Δt .01.

| ky | GS2 e2 | GS2 e3 | e2→e3 | GX upstream (t = 40) | GX repaired (t = 40) | GS2 e3 vs GX repaired |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.07741 | 0.08396 (unsettled) | +8.5% | 0.07091 | 0.10176 | — (neither settled) |
| 0.30 | 0.23482 | 0.23484 | 0.01% | 0.23443 | 0.23209 | +1.2% |
| 0.50 | 0.25533 | 0.25354 | −0.7% | 0.25409 | 0.25410 | −0.2% |

ω at ky .30 / .50: GS2 e3 gives 0.2210 / 0.4571; GX repaired gives 0.2365 / 0.4677.

**Finding: first independent kinetic-electron reference.**
- GS2 and GX agree on the kinetic-electron growth rate to 1.2% at ky .30 and 0.2% at ky .50. The GS2 value is converged to 0.7% over two rungs.
- This is the first independent reference for the Miller kinetic-electron cell: rank 5, two external codes.
- The GKX side is the remaining step (VAL-KE): a certified two-species eigenpair at Nl16/Nm48, about 1 h per ky on office CPU.

## 4. EM: KBM at β = 1.5%, A∥ only and with B∥ (EM-B-PAR)

| ky | GS2 A∥ e2 | GS2 A∥ e3 | GS2 A∥+B∥ e2 | GX A∥ upstream (t = 40) | GX A∥ repaired |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.07807 | 0.13463 | 0.19364 (e3 0.25216) | 0.20305 | 0.21323 |
| 0.30 | 0.26185 | 0.29322 | 0.35765 (e3 0.38548) | 0.31411 | 0.31392 |
| 0.50 | 0.14093 | 0.14398 | 0.18867 (e3 0.19562) | 0.17432 | 0.17418 |

**Finding: no GS2 KBM number is cited yet.**
- GS2's two-field KBM is not converged over e2 → e3: +72% at ky .10, +12% at ky .30, +2% at ky .50.
- At ky .50 GS2 e3 is 17% below GX. At ky .30 it is 7% below GX and rising toward it.
- The GX KBM golden can be called independently confirmed only with an e4 rung (ntheta 64, negrid 24).
- At e3, adding B∥ raises GS2's γ by 36% at ky .30 and 87% at ky .10. That is the size of effect EM-B-PAR has to resolve, and the GX golden (fbpar = 0) cannot speak to it.

## 5. GKX certified eigenpairs against the repaired goldens

`results/gkx_cert.txt` has one line per (geometry, ky) from `run_runtime_linear(solver="krylov")` (the certified adaptive route) at Nl16/Nm48, on the shipped GX-matched decks.

| Geometry | ky | GKX γ | GX repaired γ | Δγ | GKX ω | GX repaired ω | Δω |
|---|---:|---:|---:|---:|---:|---:|---:|
| s-α | 0.15 | 0.054824 | 0.054946 | −0.22% | 0.127005 | 0.126858 | +0.12% |
| s-α | 0.20 | 0.075159 | 0.075035 | +0.17% | 0.177909 | 0.177842 | +0.04% |
| s-α | 0.30 (Q20) | 0.093091 | 0.093049 | +0.05% | 0.28203 | 0.281983 | +0.02% |
| s-α | 0.40 | 0.080646 | 0.080885 | −0.30% | 0.375066 | 0.374934 | +0.04% |
| Miller | 0.15 | 0.059215 | 0.058369 | +1.45% | 0.091971 | 0.091850 | +0.13% |
| Miller | 0.30 | 0.125854 | 0.125874 | −0.02% | 0.215476 | 0.215461 | +0.01% |
| Miller | 0.40 | 0.143122 | 0.143138 | −0.01% | 0.306690 | 0.306705 | −0.00% |
| Miller | 0.55 (Q20) | 0.125975 | 0.125906 | +0.05% | 0.43362 | 0.433638 | −0.00% |

At the same points, GS2 (converged ladder, Q20) gives Miller 0.05783 / 0.12546 / 0.14297 at ky .15 / .30 / .40 and s-α 0.09219 at ky .30. That puts GKX within 0.3% of GS2 at Miller ky .30 and .40, and within 1.0% at s-α ky .30. At Miller ky .15 GKX is 2.4% above GS2 and 1.45% above GX, the one point where the three codes spread by more than 1%.

**Ready to promote, but not promoted here.** For the adiabatic Cyclone cells, the certified eigenpair against the repaired build is ≤ 0.3% at seven of eight points. The ledger rows `L-lin-cyclone-salpha` and `L-lin-cyclone-miller` cite the time-fitted tables, not these eigenpairs. Moving them to the eigenpairs and the repaired reference is a ledger change for the owner of `tools/evidence_ledger.toml`.

## 6. GS2 at s-α ky = 0.55 (energy-grid limit)

The Q20 ladder r1–r4 did not converge: r3 → r4 moved γ by +20.9%. Two further rungs run at ntheta 64 / nperiod 4:
- r5 (negrid 32, ngauss 12): γ = 0.02313, ω = 0.49995, settled.
- r6 (negrid 48, ngauss 16): γ = 0.01841, ω = 0.49061, settled.

GS2 is still not converged in the energy grid. Across negrid 16 / 24 / 32 / 48 its γ reads 0.0220 / 0.0278 / 0.0231 / 0.0184 (r5 → r6 is −20%). GKX's certified eigenpairs follow a similar path in Laguerre resolution: Nl 24 / 32 / 48 give 0.03301 / 0.02485 / 0.01835 (Q16). At the finest rung of each, the two codes agree to 0.3% (0.01841 vs 0.01835), but neither ladder has converged.

So there is still no converged cross-code reference at s-α ky = 0.55:
- The GX golden (0.035189, Nl 16) is a truncation value.
- The ledger's `truncation-limited` label for this ky stands.
- gyaradax's apparently converged 0.02485 (Q20) is not confirmed by either ladder.

## What is promoted, and what is not

Nothing in this record is promoted into ledger rows or fixtures:
- The repaired goldens reproduce the upstream ones to 0.06% for the two adiabatic cases, so they change no fixture value.
- The GS2–GX kinetic-electron agreement is two external codes against each other. It becomes a GKX ledger row only with the GKX eigenpair next to it.
- Wall times were taken on a host at load 40–100, so they are not time-to-solution evidence under §5.1.
