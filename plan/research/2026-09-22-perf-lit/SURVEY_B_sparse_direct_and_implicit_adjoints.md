# Survey B: sparse direct and implicit-adjoint routes from JAX

Survey date 2026-09-22. Sources are the primary ones listed at the end: package source code, official docs, and PyPI JSON metadata for versions and dates. Anything I did not read from a primary source is marked **UNVERIFIED**.

## B.1 SOLVAX inventory (read-only)

**Checkout note.** the local SOLVAX checkout has `main` checked out at 0.20.0 (`5a49926`). That is **27 commits behind `origin/main`**, which is at 0.25.0 (`7b8ca55`, PR #120). The sparse-direct work landed after 0.20.0: #111 compression, #114/#116 equilibration and #118 MUMPS. I therefore read the tree with `git show origin/main:src/solvax/<file>.py`, which changes nothing in the repository. `research/sparse-direct-gkx` points at the same commit as `origin/main`. There are no `direct/`, `compression/` or `native/` *packages*. All three are single modules.

| Module (origin/main) | What exists | How it runs from JAX | AD / adjoint |
|---|---|---|---|
| `src/solvax/native.py` | `SpluFactorization(matrix, backend="superlu"\|"mumps", memory_limit_bytes, memory_safety_factor)`, `.solve(b, trans="N"\|"T"\|"H")`, `._solve_numpy`, `.close()`, `splu_solve`. The MUMPS path is `_MumpsFactorization`, built on the PyMUMPS `{S,D,C,Z}MumpsContext` classes on `MPI.COMM_SELF`. It runs `job=1` (analysis), then checks memory: `INFOG(16)` times a safety factor must fit the budget, with `ICNTL(23)` capping the workspace. Then `job=2` (factor) and `job=3` (solve). `trans="T"` sets `ICNTL(9)=0`. `trans="H"` conjugates the RHS and the solution around the T-solve. | Host CPU only. **Eager only**: `_check_not_traced` raises `RuntimeError` under jit, vmap or grad. There is no `pure_callback` and no FFI anywhere in `src/`. | None built in. The docstring tells users to wrap the call in `jax.pure_callback` themselves. The factor serves N, T and H solves. |
| `src/solvax/native_eigen.py` | `sparse_operator_matrix`, which reads a JAX matvec into CSR with one-hot column batches (n products). `sparse_eigenpairs(matrix, shift, factorization=..., adjoint=...)` calls SciPy ARPACK `eigs` with `OPinv` from a supplied factor. `adjoint=True` finds **left eigenvectors by reusing the same factor with `trans="H"`**, with no second factorization. It also returns residual certificates. | Eager, host. | Supplies the left and right vectors that `eigen.eigenpair_reverse` needs. |
| `src/solvax/compression.py` | Curtis–Powell–Reid seeding. `column_groups(pattern)` does a greedy distance-2 colouring. `matrix_from_products(apply, pattern, groups)` needs one matvec per colour group. `verify_products` checks the recovered matrix against random probes. This is the "compression route" DKX uses. It recovers sparse Jacobians, and has nothing to do with low-rank compression. | Eager. It calls `apply` once per group from a Python loop, not vmapped. | n/a |
| `src/solvax/equilibration.py` | `equilibrate(matrix)`: Ruiz row and column scaling, which keeps complex phase (#116). Returns `Equilibration` with `scale_rhs` and `unscale_solution`. The docstring reports a 66,004-unknown DKX operator that factored in 103 s instead of 201 s, and whose residual went from 9.4e-2 to acceptable. | Eager, SciPy. | n/a |
| `src/solvax/eigen.py` | `eigenpair_reverse(theta, build, v0, primal_solver, left_solver, tangent_preconditioner, transpose_tangent_solver, condition_limit)`: a `jax.custom_vjp` for a simple eigenpair solved outside JAX. The eigenvalue cotangent uses `lᴴ(dA)r`. The eigenvector cotangent solves Nelson's bordered operator `A − λI + r lᴴ` with GMRES through `implicit.linear_solve`. It rejects the input when `‖l‖‖r‖/|lᴴr|` exceeds the condition limit. | The forward pass calls Python `complex(...)` and `float(...)`, so it is eager. | Implicit. No Krylov iterations are recorded on the tape. |
| `src/solvax/implicit.py` | `linear_solve(matvec, b, solver, transpose_solver)` wraps `lax.custom_linear_solve`. Also `recycled_linear_solve` (a `custom_vjp` that carries a recycle space), `newton_krylov` and `root_solve` (`lax.custom_root`). | Traced. | Implicit function theorem (IFT) with one transposed solve per VJP. |
| `src/solvax/krylov.py` | `gmres` (FGMRES, CGS2, complex Givens rotations, pytrees) and `gcrot`. | jit, vmap and grad. | Wrapped through `implicit.linear_solve` rather than unrolled. |
| `src/solvax/direct.py` | Block Thomas. `block_thomas_factor`/`_solve(transpose=)` reuses the Schur LU with `trans=1` for Aᵀ. `block_thomas_factor_fn`, and `block_thomas_factor_ops`/`solve_ops` for operator couplings (#106–#108). `block_thomas_checkpointed_fn` uses `custom_linear_solve` with a transpose solve. `mixed_precision_block_thomas` and `block_thomas_truncated*`/`selected_tail_fn` have `custom_vjp`s with exact-window adjoints. `certified_adjoint_window`. | Pure JAX (`jax.scipy.linalg.lu_factor`, so LAPACK or cuSOLVER getrf). Runs on GPU. | Factor reuse for Aᵀ. |
| `src/solvax/tridiagonal.py` | `tridiagonal_solve(method="thomas"\|"lax"\|"auto")`, `tridiagonal_factor`/`_solve_factored` (#117), `cyclic_tridiagonal_solve` and checked variants. The `"lax"` path `_lax_solve` wraps `jax.lax.linalg.tridiagonal_solve` in `custom_linear_solve` with transposed bands. | Pure JAX, GPU through cuSPARSE. | Transpose solve reuses the same bands. |
| `src/solvax/banded.py` | `lu_factor_banded` (no pivoting, equilibration plus a static-pivot floor with a clamp counter), `lu_solve_banded`, and a periodic Woodbury variant. | Pure JAX. | Differentiated by tracing through the solve. |

**SOLVAX gaps for GKX**
1. **No traced sparse-direct primitive.** SuperLU and MUMPS cannot sit inside `jit` or inside `custom_linear_solve`. `custom_linear_solve` traces both `solve` and `transpose_solve` into jaxprs (`jax/_src/lax/control_flow/solves.py`, `pe.trace_to_jaxpr`). So even `eigenpair_reverse` running eagerly cannot call a host factor inside its pullback unless that call goes through `pure_callback` or FFI.
2. **The adjoint convention matters.** `custom_linear_solve` builds `vecmat` as the *linear transpose* of `matvec`, which is **Aᵀ**, not Aᴴ. A host bridge must hand it `trans="T"` solves. Left eigenvectors need `"H"`. SOLVAX supports both.
3. **MUMPS solves one RHS per `job=3` call.** It loops over columns and does not use MUMPS's native dense multi-RHS path. It also exposes no ordering control (`ICNTL(7)`) and no block-low-rank control. The MUMPS user-guide specifics here are **UNVERIFIED** (not read this session).
4. **No re-factorization API.** There is no path that reuses the symbolic analysis when σ or geometry changes (`job=2` only).
5. **`pymumps` ships only as an sdist** (PyPI 0.4.0, 2026-02-18). Every install needs system MUMPS and MPI plus mpi4py.
6. **No GPU sparse direct solver.** There is no cuDSS path.
7. **`matrix_from_products` is not vmapped** over its colour groups.

## B.2 Routes from JAX to sparse and structured direct solvers

| Route | Backend | GPU | complex128 | Factor reuse across RHS | Transpose / adjoint with the same factor | jit / vmap | AD | Latest release | License |
|---|---|---|---|---|---|---|---|---|---|
| SOLVAX `SpluFactorization` (superlu) | SciPy SuperLU | no | yes | yes | yes (N/T/H) | eager only | manual | SOLVAX 0.25.0 | MIT (SOLVAX), BSD (SciPy) |
| SOLVAX `SpluFactorization` (mumps) | MUMPS 5.9.1 through PyMUMPS 0.4.0 | no | yes (Z context) | yes | yes (T via `ICNTL(9)`, H via conjugation) | eager only | manual | PyMUMPS 2026-02-18; MUMPS 5.9.1, July 2026 | BSD-3 (PyMUMPS), CeCILL-C (MUMPS, **UNVERIFIED**, from memory; the docs page I read did not state it) |
| `python-mumps` (Kwant) | MUMPS, sequential API | no | yes (**UNVERIFIED**) | yes (analyze, factor, solve) | **UNVERIFIED** | host only | none | 0.0.6, 2026-01-11 | BSD-2 |
| `jax.experimental.sparse.linalg.spsolve` | GPU: cuSOLVER `csrlsvqr` (sparse QR) through the FFI target `cusolver_csrlsvqr_ffi`. CPU: SciPy `spsolve` through `emit_python_callback`. | yes | dtype-generic dispatch; complex path **UNVERIFIED** | **no**: refactors on every call, single-vector RHS | builds an explicit CSR transpose and runs a new solve | jit yes, **no vmap** | JVP in data and b; transpose rule only in b | jax 0.11.2, 2026-09-17 | Apache-2.0 |
| `jax.lax.linalg.tridiagonal_solve` | CPU: LAPACK `gtsv`. GPU: cuSPARSE `gtsv2StridedBatch` when batch > 1 and there is one RHS column, otherwise a loop of `gtsv2` calls. For m ≤ 2 it falls back to a JAX scan. | yes | yes (Z kernels) | n/a (no factor object) | transpose rule with swapped bands (Aᵀ) | jit, vmap and grad (JVP, transpose and batching rules) | yes | jax 0.11.2 | Apache-2.0 |
| lineax | dense LU/QR/SVD/Cholesky, Tridiagonal, CG, BiCGStab, GMRES, LSMR | dense on GPU | yes | yes (state) | `solver.transpose(state)` reuses the LU with `trans=1` | yes | IFT primitive with JVP and transpose rules | 0.1.1, 2026-05-01 | Apache-2.0 |
| klujax | SuiteSparse KLU through `jax.ffi.ffi_call` | **no** | yes | yes (`analyze`/`factor`/`refactor`/`solve_with_numeric`) | `tsolve_with_*` (Aᵀ, not conjugate) | jit plus custom batching rules | primitive JVPs | 0.5.2, 2026-09-14 | LGPL-2.1 |
| sparsax | CHOLMOD, KLU and UMFPACK through XLA FFI | no | not documented (README grep found nothing) | yes (opaque tokens, caching) | transpose solve used for the VJP | jit; vmap loops in C++ | VJP in Ax and b | 0.10.0, 2026-09-17 (macOS arm64 wheels) | BSD-3 wrapper; binaries GPL through UMFPACK |
| pardiso-mkl-jax | oneMKL PARDISO through FFI (`_pardiso_ffi.cc`) | no | **no**: complex types raise "not yet supported" | yes | `transpose=True` via `iparm[11]`, reusing the factor | jit and vmap (tests) | not found in `primitive.py`/`solver.py` (**UNVERIFIED**) | 0.3.0, 2026-09-18; wheels only for x86-64 Linux and Windows | MIT (MKL proprietary) |
| PyPardiso | MKL PARDISO | no | **no** (real only) | — | — | host only | none | 0.4.7, 2025-11-02; no Apple silicon | BSD-3 |
| spineax | cuDSS 0.8 through FFI | **yes**: CUDA 13, Linux x86-64, Python ≥ 3.12, compute capability ≥ 7.5 | yes (F32/F64/C64/C128) | yes (`FactorToken`, `refactorize`, uniform and heterogeneous batches) | **general matrices: no**. `_transpose_solve_general` transposes the CSR on device and runs a **new analyze + factorize** on every backward pass, because cuDSS's `CUDSS_CONFIG_SOLVE_MODE` is "not supported right now". Symmetric and Hermitian matrices reuse the factor. | jit and vmap | `custom_linear_solve` plus a lineax solver API | 0.0.5, 2026-07-23 | MIT (cuDSS proprietary; runtime from `nvidia-cudss-cu13` 0.8.0.10) |
| xolky | cuDSS / CHOLMOD Cholesky (SPD only) | yes | no (f64 only) | yes | — | jit | **none** | — | Apache-2.0 |
| nvmath-python `nvmath.sparse.advanced.DirectSolver` | cuDSS (plan, factorize, solve; hybrid memory mode) | yes | yes (per cuDSS) | yes | as cuDSS (no transpose mode) | host Python API; JAX interop not documented (**UNVERIFIED**) | none | 1.0.0, 2026-07-09 | Apache-2.0 |
| petsc4py / slepc4py | PETSc KSP/PC (MUMPS, SuperLU_DIST, …); SLEPc EPS with `STSINVERT` | PETSc CUDA builds | needs a complex-scalar PETSc build | yes | `KSPSolveTranspose` (**UNVERIFIED** in this session) | no JAX binding. JetSCI bridges with DLPack, CuPy and ctypes; jax-firedrake uses tangent and adjoint equations. | via adjoint solve | 3.25.5, 2026-08-30 / 3.25.2, 2026-09-18 | BSD-2 |
| SuperLU_DIST 9.x | distributed LU; GPU offload of Schur updates, 3-D communication-avoiding algorithm | yes | yes | yes | **UNVERIFIED** | reachable only via PETSc | — | GitHub releases | BSD |
| JAX-AMG (`jaxamg`) | NVIDIA AmgX (AMG plus Krylov) | yes | not stated | setup caching | adjoint via `custom_vjp` | jit, batched, MPI | IFT VJP | 0.1.4, 2026-07-24 | Apache-2.0 |
| jax-fem | PETSc KSP, SciPy `spsolve`, AmgX via `pure_callback`, `bicgstab` | partial | — | no | `implicit_vjp` builds `A.transpose(A_T)` explicitly and solves again (no reuse) | host | `custom_vjp` (adjoint) | 0.0.12, 2026-06-13 | GPL-3 |
| JAXbind | binds any Python callable as a primitive with user JVP/VJP rules | **UNVERIFIED** | yes | caller's job | caller's job | jit and vmap | user-supplied | 1.3.1, 2026-02-12 | BSD-2 |
| Kokkos / Trilinos bridge | none found (**UNVERIFIED**, one search only) | | | | | | | | |

**Platform facts that constrain the choice**
- **cuSolverSp is deprecated.** NVIDIA's cuSOLVER docs say cuSolverSp "is deprecated and will be removed in a future major release", with cuDSS as the named replacement. This deprecation covers `spsolve`'s GPU path.
- **cuDSS 0.8 capabilities.** It handles general, symmetric and Hermitian real or complex matrices, has an analysis/factor/refactor/solve pipeline, multi-RHS, uniform and non-uniform batching, hybrid host/device memory (`CUDSS_CONFIG_HYBRID_MEMORY_MODE`) and multi-GPU multi-node execution. `CUDSS_CONFIG_SOLVE_MODE` (transpose / conjugate-transpose) exists in the API but accepts only 0.
- **The FFI docs now steer differentiation and batching toward HiJAX primitives.** Differentiation goes through `HiPrim` with `vjp_fwd`/`vjp_bwd_retval`. The `vmap_method` argument of `ffi_call` still works but is no longer preferred. JAX core maintainers pointed an earlier proposal to put a differentiable sparse solver in JAX core toward lineax instead (discussion #18452, 2023). The cuDSS thread (#33205, Nov 2025) has no maintainer response.
- **cuSPARSE tridiagonal kernels.** JAX never calls `gtsvInterleavedBatch`. `gtsv2StridedBatch` uses no pivoting (cyclic reduction / parallel cyclic reduction), while `gtsv2` pivots. Those algorithm details are **UNVERIFIED**, because the doc page I fetched did not state them. For a batched implicit streaming step on GPU, the RHS should be laid out as many single-column systems so that the strided-batch kernel is picked.

## B.3 Differentiation rules

1. **`lax.custom_linear_solve(matvec, b, solve, transpose_solve, symmetric, has_aux)`.**
   - The JVP reuses `solve` on `ḃ − Ȧx`. Reverse mode transposes that linear map through `transpose_solve(vecmat, ·)`, where `vecmat` is the linear transpose of `matvec` (for complex matrices this is Aᵀ, not Aᴴ).
   - With `symmetric=True`, `solve` is reused for the transpose. Without a `transpose_solve`, reverse mode is unavailable.
   - Parameter cotangents come from `jax.vjp` of the matvec closure at `x`, contracted with `λ = A⁻ᵀx̄`. No explicit Ā is formed. This is `solvax.implicit.linear_solve`. If both callbacks close over one host factor handle, the factor is reused for the tangent and adjoint solves.
2. **Implicit-function-theorem libraries.**
   - jaxopt's `custom_root`/`implicit_diff` (Blondel et al.): the README says jaxopt "is no longer maintained nor developed". Do not depend on it.
   - optimistix `ImplicitAdjoint(linear_solver=lx.AutoLinearSolver(well_posed=None))` solves one linear system in the backward pass.
   - lineax `linear_solve` is itself an equinox primitive with JVP and transpose rules. `solver.transpose(state)` and `solver.conj(state)` reuse the factors: LU uses `lu_solve(trans=1)`, Tridiagonal uses swapped bands.
   - This `AbstractLinearSolver` interface is where spineax plugs in cuDSS. A MUMPS solver could plug in the same way.
3. **Eigenpair derivatives (non-Hermitian, simple λ).**
   - The eigenvalue derivative is `λ̇ = lᴴ Ȧ r / lᴴ r` (Nelson 1976; Andrew, Chu & Lancaster 1993). It needs only the **left eigenvector**. With a shift-invert factor of A − σI, the left eigenvector comes from Arnoldi on (A − σI)⁻ᴴ using the **same factor** (`sparse_eigenpairs(adjoint=True, factorization=F)` already does this).
   - The gradient of a growth-rate or frequency objective then costs **zero extra linear solves**. It is one VJP of the operator application `θ ↦ A(θ) r` with cotangent `l`.
   - Eigenvector cotangents need one solve with the bordered or Nelson operator. SOLVAX uses `A − λI + r lᴴ` (nonsingular for simple λ) with GMRES.
   - The cheapest exact-enough route is to precondition that GMRES with the existing `(A − σI)⁻ᵀ` factor. Because σ ≈ λ, the preconditioned operator is identity plus a small perturbation plus a rank-one term, so convergence should take only a few iterations. That estimate is my own analysis, not measured.
   - The alternative is to factor the (n+1) bordered matrix `[[A−λI, r],[lᴴ, 0]]`, whose dense border fill stays in the last row and column. That means a second sparse factorization.
   - The forward pass must reject near-defective pairs (the SOLVAX `condition_limit` gate).
4. **Adjoints of matrix-free Krylov solves.** Do not unroll GMRES. Wrap it in `custom_linear_solve` with a GMRES solve on Aᵀ and the linear transpose of the preconditioner (SOLVAX `linear_solve`, `recycled_linear_solve` and `eigenpair_reverse` already do this). Skene & Burns (arXiv 2506.14792) give the kinetic-equation version. Giles (2008) collects the matrix adjoint identities, e.g. `Ā = −λxᵀ`.

## B.4 What goes in SOLVAX versus GKX, and the risks

| Route | Generic work in SOLVAX | Physics-specific work in GKX | Main risks |
|---|---|---|---|
| **R1. Host factor as a traced primitive** (recommended first) | ~200 lines. A `HostSparseFactor` handle kept in a Python closure. Solve via `jax.pure_callback` for N/T/H, with `vmap_method` for multi-RHS. Wrap it in `custom_linear_solve(transpose_solve=trans "T")`. Make MUMPS take multi-RHS in one `job=3` call. Expose `ICNTL(7)` ordering, BLR and `job=2` refactorization. Add an `eigenvalue_gradient(l, r, build, θ)` helper and a factor-preconditioned `transpose_tangent_solver` for `eigenpair_reverse`. | Operator pattern and colour groups for `matrix_from_products` (Hermite l±1 streaming, dense spectral-z blocks, chain linking, field-solve border). Choose σ. | CPU only. On GPU every solve is a device→host→device round trip. Callbacks block XLA fusion and do not export. MUMPS contexts are not thread-safe. Wheels: pymumps is sdist-only, so use conda-forge `mumps` or python-mumps. Still limited by the n^1.7 factor time and 10–12× fill. |
| **R2. CPU FFI backend** (klujax / sparsax pattern) | Depend optionally on an existing package rather than building C++ in SOLVAX. | same as R1 | KLU has no supernodes and is expected to be slow on this fill-heavy operator (**UNVERIFIED** expectation). klujax runs `jax.config.update("jax_platform_name", "cpu")` and forces x64 **at import**, a global side effect that would pin GKX to CPU. sparsax complex support is not documented. pardiso-mkl-jax is real-only and x86-only. |
| **R3. GPU cuDSS via spineax** | Optional backend behind the same factor interface. For the general adjoint, factor Aᵀ once and cache it next to A (2× memory) rather than refactoring on every backward pass as spineax does. | same pattern; complex128 | CUDA 13 on Linux only; spineax is 0.0.x. There is no transpose solve, so every adjoint needs a second factor. FP64 throughput on consumer GPUs is low (e.g. the office A4000s, **UNVERIFIED** figure). Hybrid memory mode may be needed at n ~ 7e4 with 10–12× fill. |
| **R4. Structured direct solve in pure JAX** (block Thomas, tridiagonal) | Already present: `block_thomas_factor_ops`/`solve_ops(transpose=)`, `tridiagonal_factor`, `lax` path with `gtsv2StridedBatch`. Maybe add a Woodbury border for low-rank field and collision couplings. | Order unknowns so that the chain operator is block-tridiagonal (e.g. in Hermite l, or in z for a banded z discretization). Treat the field solve and conserving collision terms as a low-rank border. Physics-specific; my analysis, not from a source. | Block size m makes the cost O(N m³). Spectral-in-z blocks are dense. Unpivoted `gtsv2StridedBatch` can be unstable if the operator is not diagonally dominant. |
| **R5. PETSc/SLEPc** (shift-invert EPS with MUMPS or SuperLU_DIST) | Out of scope for SOLVAX core; at most an example bridge (DLPack plus host callback). | assembling the Mat | Heavy build (complex PETSc), no JAX AD; the adjoint must be hand-wired as in R1. |
| **R6. Matrix-free** (GMRES / shift-invert Arnoldi with a structured preconditioner) | Present (`gmres`, `gcrot`, `linear_solve`, `eigenpair_reverse`). | The preconditioner: R4 used as the preconditioner for the full operator. | Iteration counts near σ ≈ λ. |

**Recommendation.** Take R1 with MUMPS on CPU now: it is generic, small, and DKX has already measured MUMPS 20× faster than SuperLU. Put the left-eigenvector and eigenvalue-gradient path first, since it needs no extra solve. In parallel, benchmark R4 (structured block elimination), which is the only route that keeps GKX resident on the GPU. Keep R3 (cuDSS) as an optional GPU backend once there is a transpose-solve plan: either cache a second factor, or wait for cuDSS `SOLVE_MODE`.

## References
1. SOLVAX source, `origin/main` @ `7b8ca55` (0.25.0): `src/solvax/{native,native_eigen,compression,equilibration,eigen,implicit,krylov,direct,tridiagonal,banded}.py` — https://github.com/uwplasma/SOLVAX
2. JAX FFI guide — https://docs.jax.dev/en/latest/ffi.html
3. `jax.lax.custom_linear_solve` — https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html ; source `jax/_src/lax/control_flow/solves.py` — https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/solves.py
4. `jax.experimental.sparse.linalg` (spsolve) — https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py
5. `tridiagonal_solve` lowering — https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py ; kernels — https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/sparse_kernels.cc ; csrlsvqr — https://github.com/jax-ml/jax/blob/main/jaxlib/gpu/solver_kernels_ffi.cc
6. cuSOLVER docs (cuSolverSp deprecation) — https://docs.nvidia.com/cuda/cusolver/index.html
7. cuSPARSE docs — https://docs.nvidia.com/cuda/cusparse/index.html
8. cuDSS docs — https://docs.nvidia.com/cuda/cudss/ ; types and config — https://docs.nvidia.com/cuda/cudss/types.html
9. nvmath-python DirectSolver — https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html
10. lineax solvers — https://docs.kidger.site/lineax/api/solvers/ ; source — https://github.com/patrick-kidger/lineax
11. optimistix adjoints — https://docs.kidger.site/optimistix/api/adjoints/
12. jaxopt (status) — https://github.com/google/jaxopt
13. spineax — https://github.com/johnviljoen/spineax (`spineax/cudss/solver.py`); Viljoen et al., arXiv:2606.26341 (2026)
14. xolky — https://github.com/pedrozudo/xolky ; JAX discussion #33205 — https://github.com/jax-ml/jax/discussions/33205
15. klujax — https://github.com/flaport/klujax (`klujax.py`)
16. sparsax — https://github.com/knaaptime/sparsax
17. pardiso-mkl-jax — https://github.com/nardi/pardiso-mkl-jax ; PyPardiso — https://github.com/haasad/PyPardiso
18. PyMUMPS — https://github.com/PyMUMPS/pymumps ; python-mumps — https://gitlab.kwant-project.org/kwant/python-mumps ; MUMPS — https://mumps-solver.org/index.php?page=doc
19. PETSc MATSOLVERMUMPS — https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/ ; SLEPc EPS manual — https://slepc.upv.es/release/documentation/manual/eps.html
20. JetSCI (JAX–PETSc), arXiv:2604.22087 — https://arxiv.org/abs/2604.22087
21. I. Yashchuk, *Bringing PDEs to JAX with forward and reverse modes AD*, arXiv:2309.07137 — https://arxiv.org/abs/2309.07137
22. SuperLU_DIST — https://github.com/xiaoyeli/superlu_dist ; Li et al., ACM TOMS (2023), DOI 10.1145/3577197
23. JAX-AMG, arXiv:2606.09001 (SoftwareX 35, 102966, 2026) — https://arxiv.org/abs/2606.09001
24. jax-fem — https://github.com/deepmodeling/jax-fem (`jax_fem/solver.py`)
25. JAXbind, Roth, Reinecke & Edenhofer, JOSS 9(98), 6532 (2024) — https://arxiv.org/abs/2403.08847
26. JAX discussion #18452 (sparse solver in core) — https://github.com/jax-ml/jax/discussions/18452
27. R. B. Nelson, *Simplified calculation of eigenvector derivatives*, AIAA J. 14(9), 1201 (1976), DOI 10.2514/3.7211
28. A. L. Andrew, K.-W. E. Chu & P. Lancaster, *Derivatives of eigenvalues and eigenvectors of matrix functions*, SIAM J. Matrix Anal. Appl. 14(4), 903 (1993)
29. M. B. Giles, *Collected matrix derivative results for forward and reverse mode AD*, LNCSE 64 (2008), DOI 10.1007/978-3-540-68942-3_4
30. M. Blondel et al., *Efficient and modular implicit differentiation*, NeurIPS 2022, arXiv:2105.15183
31. C. S. Skene & K. J. Burns, *Fast adjoints for kinetic solvers*, arXiv:2506.14792 (cited from the SOLVAX docstring; not re-read)
32. PyPI JSON metadata for all version and date entries — https://pypi.org/pypi/<name>/json (queried 2026-09-22)
