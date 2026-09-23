# SOLVAX-DIRECT report (plan G.2; PERF-LIT items 5 and 6): interim, paused

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
