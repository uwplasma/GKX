# SOLVAX-DIRECT report (plan G.2; PERF-LIT items 5 and 6)

## Final results (2026-09-23; supersede the interim notes below)

Measured on the office host. CPU rows: single thread, x64. GPU rows: one idle RTX A4000, running cuDSS through nvmath-python 1.0 in complex128. Host load was 30-90 on 36 cores, so wall times are upper bounds. To print every table: `python summarize.py records`.

**Verdict.**
- Item 5 (sparse direct) wins at every size measured and is implemented:
  - uwplasma/SOLVAX#121 adds `sparse_solve`, `sparse_eigenvalue` and `csr_data_from_products`, plus multi-RHS MUMPS solves.
  - The GKX consumer is #295: `solver_growth_rate_from_geometry(eigensolver="sparse-direct")`.
- Item 6 (device `pr3-cm` apply) was not built. Even a free preconditioner apply leaves 15.8k-27.8k inner iterations at d96. That is at best parity with `adaptive` (33.9k), and the direct route is already about 6x faster.

**Growth-rate gradient, n = 3,072** (PERF-LIT geometry; nine profiles differentiated):

| route | value+grad s (warm) | vs dense |
|---|---|---|
| dense | 154, 177, 193 | reference |
| adaptive-propagator | 117, 183 | 1.9e-13 |
| sparse direct, jitted (`gradient.py`, two processes) | 4.1-5.5 | 2.9e-13 |
| GKX #295, eager | 19.2, 19.9 | 2.0e-13 |

Both acceptance conditions hold:
- The gradient matches dense to 1e-8; the measured difference is 3e-13.
- It is faster than `adaptive`: 25-40x jitted, 6-9x through GKX eagerly. The eager call re-probes the pattern on every call.

**Certified dominant eigenpair:**

| case | n | direct (assembly + factor + Arnoldi) | adaptive | pr3-cm route |
|---|---|---|---|---|
| d96 | 3,072 | about 5 s (2.8-3.7 + 0.5-0.7 + 1.4-1.5) | 31.6-36.3 s | 433 s |
| r96 | 18,432 | about 109 s (12.7 + 20.2 + 75.7) | 300 s | - |
| prod | 73,728 | about 1,250 s (119 + 117 + 1,011) | - | - |

- The direct and adaptive routes certify the same eigenvalues:
  - d96: 0.1012864978-0.2454963203j, agreeing to 7e-14.
  - r96: 0.0987757461-0.2769045684j.
- prod gives 0.0930911733-0.2820327315j, residual 1.7e-13.
- Left eigenvectors from the same factor have residuals of at most 5e-12. Use the B3/B4 rows for d96: B1/B2 predate the left-shift fix.
- At prod, the 16 Arnoldi candidates cost 717 solves at 1.4 s each; that is where to cut.

**One shifted solve:**

| case | MUMPS factor, solve (CPU) | cuDSS factor, solve (A4000) | SuperLU factor | pr3-cm solve to 1e-6 / 1e-10 |
|---|---|---|---|---|
| d96 | 0.5-0.7 s, 0.017 s | - | 4.3 s | 2.6 s / 4.5 s |
| r96 | 20.2 s, 0.37 s | 1.6-1.7 s, 0.0065 s | 381 s | 72 s / 121 s |
| prod | 117 s, 1.0-1.7 s (6.3 GB) | 15.5 s, 0.026 s | not run | 322 s / 543 s |

- Direct backward errors are at most 1e-15.
- At prod, MUMPS needs 7.3-8.3 GB.
- MUMPS orderings are within 1.5x of each other.

**Remaining:**
- A SOLVAX release with #121, then the GKX floor for #295.
- A GPU host factor behind the same primitive. cuDSS is 7-8x faster to factor and 50x faster to solve here, but no JAX binding has a transposed solve on CUDA 12.
- Caching the pattern in #295.
- Fewer Arnoldi candidates at prod.

## Interim notes (2026-09-22, paused)

Office host, CPU, single thread, x64, JAX 0.10.2, MUMPS 5.8.2 (conda-forge) and
PyMUMPS 0.4.0. The host load was 50-90 on 36 cores throughout, so wall times are
upper bounds. Ratios measured within one process are the reliable numbers.
Records are in `records/`; print the tables with `python summarize.py records`.

## Verdict so far
- **Item 5 (sparse direct) wins where memory allows.**
  - d96 (n=3,072):
    - The direct route certifies the same eigenvalue as `adaptive` to 7e-14 (0.1012865-0.2454963j, residual 2.7e-12).
    - It takes about 5 s end to end: assembly 3.5-4 s including compile, MUMPS factorization 0.6 s, Arnoldi 1.2 s.
    - `adaptive` takes 31.6-36.3 s (33,916 matvec-equivalents).
    - GKX's `pr3-cm` shift-invert route takes 433 s (94k matvec-equivalents).
  - One `pr3-cm` FGMRES solve to 1e-6 costs 155 iterations (2.6 s), against a 0.017 s MUMPS solve.
  - r96 (n=18,432):
    - MUMPS/METIS factors in 20 s (1.07 GB); SuperLU COLAMD takes 381 s.
    - A MUMPS solve takes 0.37 s; a `pr3-cm` solve takes 72 s at 1e-6 and 121 s at 1e-10.
  - prod (n=73,728):
    - MUMPS estimates 7.3-8.3 GB, so all three orderings were refused under the 5 GB budget (the host had about 5 GB free).
    - A `pr3-cm` solve there takes 322 s at 1e-6 and 543 s at 1e-10.
- **Item 6 (device pr3-cm apply): not implemented.**
  - Even a free preconditioner apply (c_P = 0) leaves the route's inner iterations: 15.8k-27.8k at d96 in Q28 and in this run.
  - At best that is parity with `adaptive` (33.9k), and the direct route is already about 6x faster.
- **SOLVAX uwplasma/SOLVAX#121 (draft)** adds `sparse_solve`, `sparse_eigenvalue` and `csr_data_from_products`, plus multi-RHS MUMPS solves.
  - The tests pass against real MUMPS.
  - On a GKX objective at n=144, `gradient.py` matches the dense growth-rate gradient to 3.2e-14.
- Collision operator: z-local, 3.3 nnz/row, RCM bandwidth 15. A batched dense per-z solve covers it, not sparse direct.
- cuDSS: not run. Both A4000s were in use all session.

## Caveat
The d96 rows B1/B2 predate the fix to the left-eigenvector shift, so their `left_residual` of 0.58 is invalid. Their right eigenpairs are valid. The r48 and r96 left residuals are 3.7e-12 and 1.1e-13. r48 used GKX's seed shift (-0.014-0.038j, in the marginal spectrum); there `pr3-cm` did not converge in 2,400 iterations.

## Not done
- The n=3,072 gradient rows (`run_gradient.sh`: dense, direct, adaptive, direct).
- Rerunning the d96 direct row alongside `adaptive` at r96 (`run_extra.sh`).
- c48 (killed at the pause).
- Prod on a host with at least 10 GB free.
