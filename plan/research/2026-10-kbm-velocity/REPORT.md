# KBM velocity-space record, 2026-09-28 (KBM-VEL lane: EM2 item 2, EM-B-PAR, velocity basis)

Question from users: GX (and so GKX) may give wrong KBM growth rates, because of its collision
operators or because a Hermite-Laguerre basis spans all of velocity space; they propose a
truncated or rescaled Hermite-Laguerre basis that concentrates resolution on a few thermal speeds.

Short answer:
- The basis is not what makes GX/GKX KBM numbers differ. On the GX KBM case, the GKX Hermite
  ladder is converged to 0.7% by Nm = 32 (with the GX hypercollisions), GX and GKX agree to 1.0%
  (A∥) and 0.5% (A∥ + B∥), and hypercollisions speed convergence without biasing the limit
  beyond what the ladders resolve (section 3).
- What moves KBM γ by 10–60% is the parallel domain and its end damping. The shipped GX deck uses
  three 2π segments (nperiod 2); nperiod 3 raises γ by 7.5% at ky 0.3 and by 21% at ky 0.1. On
  that deck GKX's timestep-free routes (eigen/Krylov) use an end-damping rate 1000x weaker than
  the time route and return γ = 0.208 against 0.346 at the same resolution (section 2).
- A velocity-scaled Hermite basis gives no gain at equal cost (section 5). No code was added.

All runs are on office CPU (2x A4000 busy with other users; GX got GPU 1 for two runs). Load was
50–80 on 36 cores, so wall times are not timings. Units are GX/GKX (Lref = a, vt = sqrt(T/m));
GS2/stella are converted as in the 2026-09-27 xcode record.

## 1. Literature (primary sources)

- **GX, Mandell et al., JPP 90, 905900402 (2024), arXiv:2209.06731.** The KBM benchmark is the
  ITG→KBM β scan at kyρ = 0.3 with A∥ only (δB∥ = 0), against GS2. Appendix G.2: GX used three 2π
  segments, Nz = 24 per segment, **Nm = 128**, Nℓ = 16; GS2 used 32 energies and 33 pitch angles.
  The shipped `kbm_miller.in` uses Nm = 48. For kinetic electrons the paper says agreement
  required 128 Hermite modes. Its hypercollision operator is ν m^p with p = Nm/2 and ν ∝ |k∥| v_ts,
  acting on m > 2 (eqs. 4.27–4.28), and it notes that plain truncation without a sink is a poor
  closure at low collisionality. No GX KBM defect report was found; the known defect is the
  `dampEnds_linked` launch cap (plan §3.7), which the repaired build fixes.
- **Mandell, Dorland & Landreman, JPP 84 (2018).** The Laguerre-Hermite pseudo-spectral
  formulation GX and GKX both use; the basis is weighted by the local Maxwellian at each species'
  own thermal speed.
- **Velocity-scaled Hermite in Vlasov:** Schumer & Holloway, JCP 144 (1998) (scaled Hermite
  functions, scale chosen from the solution); Camporeale et al. (Vlasov–Maxwell linear stability
  with shifted/scaled bases); Delzanno, JCP 301 (2015) and Pagliantini et al., arXiv:2208.14373
  (AW-Hermite with an adapted scale α and shift u); Parker & Dellar, JPP 81 (2015)
  (Fourier–Hermite, hypercollisions, recurrence). The gain in this literature is for
  distributions that drift or heat away from the reference Maxwellian (beams, bump-on-tail,
  strong heating), where α and u track the bulk.
- **Gyrokinetic moment codes:** Frei et al., JPP (2020, 2022, 2023) and Hoffmann et al., JPP
  (2023) (Hermite-Laguerre with full collision operators, TEM and microtearing convergence) use
  the unscaled local-Maxwellian basis. No finite-support ("truncated") Hermite-Laguerre basis is
  established for gyrokinetics that I could find: δf gyrokinetics linearizes about the local
  Maxwellian, and the fields and drives are its low moments, which a finite-support basis loses.
  Finite-support v-grids are what GS2 and stella already use.

## 2. The parallel domain and end damping drive the KBM number

The deck is `scripts/kbm_miller.toml`: `examples/06_electromagnetic/case_full.toml` with only
`[geometry]` replaced by the circular Miller surface of GX `benchmarks/linear/KBM/kbm_miller.in`
(ntheta 32, nperiod 2, Nz 96). The shipped GKX KBM deck is s-α, while its ledger reference
(`L-lin-kbm`) is this Miller deck.

`LinearParams.end_damping_strength`: an explicit `damp_ends_rate` wins; otherwise the rate is
`damp_ends_amp / dt`, or `damp_ends_amp` when there is no dt. So the time route runs the
GX-legacy deck (amp 0.1) at rate 100 for dt 1e-3, and the eigen/Krylov routes run it at 0.1.

ky 0.3, Nl 4, Nm 8, certified sparse-direct eigenpairs (residual ≤ 1.4e-13):

| end-damping rate | nperiod 2 | nperiod 3 | nperiod 4 |
|---|---:|---:|---:|
| 0.1 (timestep-free default) | 0.2078 | 0.3239 | 0.3535 |
| 100 (time route at dt 1e-3) | 0.3462 | 0.3659 | 0.3648 |
| 250 (dt 4e-4) | 0.3437 | — | — |

- The time route at dt 1e-3 and 4e-4 (imex2 and rk4) reproduces the rate-100 and rate-250
  eigenvalues to 2e-5. The time route looks dt-independent only because the damped band is
  saturated.
- With strong damping the domain converges by nperiod 3 (0.3%). With weak damping it has not
  converged at nperiod 4.
- At resolution (rate 100), nperiod 2 → 3 moves γ from 0.3234 to 0.3462 at Nm 16 and from 0.3157
  to 0.3395 at Nm 32/48 (+7.5%), and ky 0.1 from 0.2101 to 0.2536 (+21%, Nm 32).

So the GX KBM golden (nperiod 2) is a domain-truncated number. Any KBM number from GKX's eigen
routes on a legacy deck (no explicit `damp_ends_rate`) answers a different, weakly damped
problem. The migration helper `migrate_end_damping_reference` exists; the KBM decks do not use it.

## 3. Velocity convergence in GKX (nperiod 2, rate 100, ky 0.3)

Hermite ladder, Nl 4. Where both routes ran, the time route and sparse-direct agree to 1e-5.

| Nm | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| with GX hypercollisions | 0.3462 | 0.3255 | 0.3234 | 0.3219 | 0.3157 | 0.3160 | 0.3160 | 0.3140 |
| no hypercollisions | 0.4122 | 0.3967 | 0.3869 | — | 0.3647 | 0.3421 | 0.3370 | 0.3297 |

- With hypercollisions the ladder is flat to 0.7% from Nm 32 to 128.
- Without them it converges algebraically from above and still moves 2% per doubling at Nm 128.
  At Nl 2, Nm 96 it gives 0.3256, against 0.3076 with hypercollisions.
- The two ladders approach each other: the gap shrinks from 0.066 to 0.016. So, within what Nm 128
  resolves, hypercollisions accelerate convergence rather than shift the answer.
- Dense spectra at Nm 8 show the KBM is the most unstable eigenvalue with and without
  hypercollisions, so there is no spurious truncation mode.
- Laguerre, Nm 32: Nl 2 / 4 / 8 = 0.3121 / 0.3157 / 0.3162 (converged at Nl 4 to 0.2%).
- Collisions (ν_i 0.01, ν_e 0.6, default operator, plus hypercollisions), Nm 32: 0.3770 (+19%).
  This is a physics sensitivity, not a basis defect.
- Eigenfunctions: the electron Hermite spectrum is 98.8% m = 1 (current), and the ion spectrum
  decays over m. Laguerre spectra are dominated by l ≤ 1 (`results/projection.txt`).

## 4. Cross-code (circular Miller, β 0.015, kinetic electrons)

| ky 0.3 | γ | ω | notes |
|---|---:|---:|---|
| GKX A∥, nperiod 2, Nl 4 Nm 128 | 0.3140 | 1.0792 | converged ladder |
| GX repaired A∥, Nm 48 (golden deck) | 0.3139 | 1.0758 | 2026-09-27 record |
| GX repaired A∥, Nm 128 | 0.3108 | 1.0832 | GKX +1.0% |
| GKX A∥, nperiod 3, Nm 48 | 0.3395 | 1.0152 | domain-converged (Nm 8 test) |
| GS2 A∥ (nperiod 3) e3 / e4v / e4 | 0.2932 / 0.2939 / 0.2980 | 0.995 / 0.993 / 1.000 | bakdif 0.05 |
| GS2 A∥ bakdif 0: e4v / e4 | 0.3159 / 0.3146 | 1.016 / 1.017 | centred |
| GKX A∥+B∥, nperiod 2, Nm 32 | 0.4084 | 1.0038 | |
| GX repaired A∥+B∥, Nm 48 | 0.4065 | 1.0093 | GKX +0.5% |
| GKX A∥+B∥, nperiod 3, Nm 32 | 0.4313 | 0.9579 | |
| GS2 A∥+B∥ e4v / e4 | 0.3864 / 0.3898 | 0.941 / 0.947 | bakdif 0.05 |

- GS2 is converged in velocity (e3 → e4v: 0.2%) but not in its parallel discretization. Changing
  the upwinding parameter `bakdif` from 0.05 to 0 raises γ 7.5%. With bakdif 0, GS2 (0.315) is 7%
  below GKX at the same nperiod 3 (0.3395). That gap is open. Candidates are GS2's θ grid (the
  ntheta 96 run was queued and did not finish) and the ballooning-end treatment.
- The B∥ effect agrees in size: +27% in GKX (nperiod 3) and +31% in GS2 (e4v).
- ky 0.1:
  - GKX nperiod 2, Nm 64 gives 0.2077, against GX Nm 128 0.2177 (t = 40; this is a low-ky
    transient per the 2026-09-27 record).
  - nperiod 3 gives 0.2536.
  - GS2 e4 gives 0.148, but is not converged in θ (e3 → e4 +10%).
- ky 0.5: GKX Nl 4 Nm 32 eigenpair 0.1668 / 1.592, GX Nm 128 0.1806 / 1.605, GS2 e4 0.146 / 1.509.
- **stella v1.0 is not usable here.** Every run returns γ ≈ 1.43–1.49 and ω ≈ 0.31–0.33:
  v-grid 24/12 and 48/24, dt 5e-3 and 1e-3, A∥ and A∥+B∥. The result does not respond to β, B∥
  or the grid. The first attempt also exposed a trap: stella ignores `&electromagnetic` unless
  `include_electromagnetic = .true.` is set in `&gyrokinetic_terms` (`scripts/stella_kbm.py`).

## 5. Velocity-scaled Hermite basis: negative result

Two tests, both on GKX's exact operator.

**Representation** (`scripts/scaled_hermite_projection.py`). Project the converged eigenvector
onto the first N functions ψ_k(v/α)F(v/α)/α in the free-energy norm ∫f²/F.
- That norm is finite only for α < √2, so wider bases are not admissible.
- At Nm 32, α = 0.8 represents the ion profile better for N ≥ 12 (error 3.2e-2 against 8.0e-2 at
  N 12).
- The electron profile is best at α = 1 for N ≤ 8.

**Dynamics at equal cost** (`scripts/galerkin_scaled.py`).
- The N-function scaled basis evolves under Q^H A Q. A is the exact sparse GKX operator at Nm 64
  without hypercollisions. Q expands the scaled functions in the first 64 Hermite functions
  (truncation loss ≤ 3e-3).
- α = 1 reproduces plain truncation at Nm = N exactly. Checked: N 8 and 16 give 0.41219 and
  0.38694, equal to the Nm 8 and Nm 16 eigenpairs.
- `--hyper` adds GKX's own hypercollision operator at truncation N to the scaled coefficients.

| γ at N (ky 0.3, Nl 4) | 4 | 6 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|
| α 1.0, no hyper | 0.468 | 0.413 | 0.412 | 0.397 | 0.387 | — |
| α 0.9, no hyper | no KBM | 0.449 | 0.386 | 0.380 | 0.365 | — |
| α 0.8, no hyper | no KBM | 0.258 | 0.268 | 0.357 | 0.359 | — |
| α 0.7, no hyper | no KBM | no KBM | no KBM | 0.255 | 0.343 | — |
| α 1.0 + hyper (production) | — | — | 0.346 | 0.325 | 0.323 | 0.322 |
| α 0.9 + hyper | — | — | 0.221 | — | — | — |
| α 0.8 + hyper | — | — | no KBM | 0.265 | 0.314 | — |

The references are the no-hyper Nm 128 value, 0.330, and the converged production value, 0.314.
"No KBM" means no unstable eigenvalue near the KBM.

- Without hypercollisions, α 0.8–0.9 moves the N-term answer toward the Nm 128 value faster than
  α = 1. It does so non-monotonically, and α ≤ 0.8 loses the mode at N ≤ 8.
- No scaled basis beats what GKX actually runs (α = 1 with hypercollisions) at equal N. The
  production route is within 3% of converged at N 12. The best scaled row reaches that only at
  N 16, after missing the mode at N 8.
- Why: in the Maxwellian-weighted basis at the species' own thermal speed, density, current and
  temperature are single coefficients (m = 0, 1, 2), and the drive and field solves act on them.
  A scaled basis spreads the Maxwellian and those moments over every even coefficient.
- The KBM's resolution need is the ion Hermite tail (phase mixing), and a width change does not
  remove it. The Vlasov-literature gains come from tracking a drifting or heating bulk, which δf
  gyrokinetics does not have.

Decision (per the lane brief): no code. A scale-factor parameter would touch streaming, mirror,
drifts, drive, field moments, collisions and hypercollisions for no measured gain.

## 6. Other findings for owners

- **Krylov fails on EM KBM.** `solver = "krylov"` (used by the `[scan]` of `case_full.toml`)
  fails its outer gate on the Miller KBM deck at Nl 8 / Nm 16.
  - The inner FGMRES does not converge: relative residual 0.97–0.99 after 768–4800 iterations.
  - Tried with the KBM preset, with an explicit shift at the GX eigenvalue, and with its conjugate.
  - The research sparse-direct route (`scripts/sparse_eig.py`) certifies at residual ≤ 4e-13 in
    4–40 min up to n = 98k and matches the time route to 1e-5.
- **Time-route fit accepted a non-growth window.** At ky 0.5 (dt 2.1e-4, above that ky's CFL) the
  runtime returned γ = 0.850, ω = 1.879 with `fit_settled = true`. It also warned that the window
  "spans only 0.01 growth times". The operator has no eigenvalue there: a shift-invert at
  0.85 − 1.88i finds 0.1668 as the largest. A settled flag should not coexist with that warning.
- **The KBM parity table's reference column does not match the golden it cites.**
  `docs/_static/kbm_mismatch_table.csv` has gamma_ref 0.145 / 0.263 / 0.219 / 0.145 at ky 0.1–0.4.
  `src/gkx/data/kbm_reference.csv`, the golden it cites, has 0.206 / 0.338 / 0.314 / 0.243.
- Time runs without hypercollisions overflow (non-finite) at 0.7x the CFL estimate; 0.3x is stable.

## 7. What is not done

- GS2 ntheta 96 (θ convergence), and bakdif 0 at ky 0.1/0.5 and with B∥: queued, stopped at the pause.
- GKX nperiod 4 at Nm 32, and B∥ at nperiod 3, Nm 48 (sparse): queued, stopped.
- GKX Nl 16 Nm 32 (time run): stopped.
- GX at nperiod 3 needs a new `eik` geometry file, which the harness cannot generate; not run.
- Nothing is promoted into ledger rows.

## Files

- `scripts/kbm_eig.py` is the runner. Variants: base, nohyper, coll, bpar, rk4, damp<r>, np<k>.
  Routes: time, adaptive time, or sparse-direct.
- Runner support files: `sparse_eig.py`, `run1.sh`, `lane.sh`, `kbm_miller.toml`.
- `scripts/scaled_hermite_projection.py` and `scripts/galerkin_scaled.py` (section 5).
- `scripts/gs2_e4.py` and `scripts/stella_kbm.py` run from the 2026-09-27 xcode bench and reuse its
  `cases.py`, `em.py`, `fit.py` and `run_grid_queue.sh`.
- `results/gkx_runs.txt`: one line per GKX run (tag, geometry, ky, Nl, Nm, variant, γ, ω, route,
  residual, settled, wall).
- Other results: `results/galerkin.txt`, `results/gs2_stella_fits.csv`, `results/gx_runs.txt`,
  `results/projection.txt`, `results/spectra_and_failures.txt`.
