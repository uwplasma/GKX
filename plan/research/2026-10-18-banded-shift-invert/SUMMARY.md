# Banded shift-invert: measurements (2026-10-18)

Code: branch `perf/banded-shift-invert` (GKX), uwplasma/SOLVAX#149. Harness: `scripts/eig.py` (one `run_runtime_linear` call, RESULT json with SHA, host, wall, peak RSS, load average). Raw logs: office `~/gkx-evidence/banded/out/`; the RESULT lines are in `results/`.
Host: office, 2x RTX A4000 (16 GB), 36 threads, shared (load average 5-21 at launch of each run, see `uptime` in each RESULT). GPU runs use one card; CPU runs `JAX_PLATFORMS=cpu`. float64 unless `x64: "0"`. Wall = the whole call (geometry, cache, solve).

## Routes compared (Cyclone adiabatic = `examples/01_linear_tokamak/case_full.toml`; Miller decks from the KE/EM study, `plan/research/2026-10-15-ke-em-solvers`)

| case | adaptive | time route | banded (`auto`) |
|---|---|---|---|
| Cyclone ky 0.3 (8,24), GPU | 0.0986916-0.276870j, 38.7 s | 0.098754-0.276896j, 22 s | 0.0986916-0.276870j, res 1.2e-13, 174 solves, 30 s |
| Cyclone ky 0.3 (8,24), CPU | 217 s | | 146 s |
| Cyclone ky 0.3 (16,48), GPU | 0.0931303-0.281967j, 90.6 s | 0.093018-0.282025j, 40-47 s | out of memory (4 x 1.8 GB complex64 arrays on a shared card) |
| Cyclone ky 0.3 (16,48), CPU | 0.0931303156-0.2819665405j, res 1e-14, 1109 s | | same pair to 1e-10, res 1.5e-9, 692 s (`banded` with one correction) |
| Miller KE ky 0.3 (8,24) | 0.2311519502-0.2511135689j, res 5e-13, 430 s (CPU, 2026-10-15: 6919 s) | non-finite at the deck dt | 0.2311519502-0.2511135689j, res 2.7e-10, 371 solves, 54 s |
| Miller KE (8,24), float32 | refuses (research, (4,8): res 2.3e-4 > 1.2e-4) | | res 1.17e-4 < 1.19e-4 gate, 55 s |
| Miller KE (12,32), CPU | not run (hours) | | 0.232175-0.242723j, res 8.1e-7 (one correction, 8 Ritz values), 718 solves, 3171 s, 19.5 GB peak |
| KBM ky 0.2 (8,24) | 0.3427091263-0.8313555546j, res 1.8e-12, 428 s | non-finite at the deck dt | 0.3427091263-0.8313555547j, res 1.1e-9, 428 solves, 58 s |
| ETG ky 15 (24,8) | not finished in 40 min (2026-10-15) | 5.72668+13.69882j, 12 s | 5.73056+13.70390j, res 1.1e-13, 275 solves, 45 s |

Cyclone ky 0.5 (16,48), CPU, banded: 0.0539170-0.455642j, res 2.4e-10, 574 s.

The 2026-10-15 prototype (`bandeig.py`, sigma = 1, 12 Ritz values, complex128 factor) certified KBM 0.0331533-0.195312j: a real eigenpair of the operator, but not the fastest (adaptive and the route above agree on 0.3427). The shift matters.

## GS2 8.2.1 (Cyclone base case, `gs2/cyc_ky03.in`)

s-alpha, ntheta 32, nperiod 2, ngauss 8, negrid 16, no collisions, ky_GS2 = 0.42426 = sqrt(2) x 0.3. Converged (t = 80): omega 0.559395, gamma 0.178022 (GS2 units, vth/R). Rescaled by (R/a)/sqrt(2) = 1.96419 (a/L_T = 2.49 <-> R/L_T = 6.9): gamma 0.09063, omega 0.28481, against GKX (16,48) 0.09313 and 0.28197 (2.7% and 1.0%). The unit mapping is an inference, not a parity run.

## Tuning (GPU, Miller KE and KBM (8,24), Cyclone (8,24); one change at a time)

| defect corrections | Ritz values | Cyclone solves, residual | KE solves, residual, s | KBM solves, residual, s |
|---|---|---|---|---|
| 1 | 8 | 151, 3.5e-9 | 436, 5.3e-7, 52 | 1066, 2.2e-7, 79 |
| 2 | 8 | 151, 1.2e-13 | 436, 2.7e-10, 58 | 1066, 9.8e-10, 93 |
| 1 | 4 | 166, 3.5e-9 | 385, 6.2e-7, 49 | 4919, 2.7e-7, 241 |
| 1 | 6 | 116, 3.5e-9 | 529, 4.7e-7, 59 | 1661, 2.8e-7, 107 |
| 2 | 12 | (final run: 174, 1.2e-13) | 371, 2.7e-10, 56 | 428, 1.1e-9, 59 |
| 2 | 16 | | 401, 2.7e-10, 61 | 348, 1.1e-9, 53 |

Chosen: 2 corrections, 12 Ritz values.

## Other measurements behind decisions

- GPU block solve of Cyclone (8,24) (12 blocks of 1536, complex64): 10 ms through LU triangular solves, 2.8 ms with stored inverses (`block_thomas_factor_ops(store="inverse")`). An XLA scatter of the result into the state cost 140 ms, ten times the solve; replaced by a gather.
- Dense truncations of the first Hermite pairs, top eigenvalue against the converged one: Cyclone (Nl<4, Nm<4) 0.0962-0.2685j, (Nl<4, Nm<8) 0.1014-0.2393j, (Nl<8, Nm<4) 0.1045-0.3049j for 0.0987-0.2769j; KE (Nl<4, Nm<4) 0.1178-0.2330j, (Nl<2, Nm<8) 0.1865-0.2665j for 0.2312-0.2511j; KBM 0.4478-0.8250j, 0.4189-0.8183j for 0.3427-0.8314j. The estimate is within 0.03-0.12 of the answer, which the 12 Ritz values and the re-centring pass cover.
- A coarse test grid (Nl 8, Nm 8, Nz 16, 1024 unknowns): the Laguerre<4 two-pair truncation sat 0.12 from the true mode (0.0540-0.1307j) and the 6 nearest Ritz values returned -0.0026; the whole operator is now solved densely up to 3072 unknowns.
