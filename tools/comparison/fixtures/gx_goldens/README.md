# GX shipped linear goldens: provenance and reference validity

GX ships reference outputs for its own linear benchmarks in its repository. They
are rank-3 references under [plan.md](../../../../plan.md) section 3.1: produced
by the reference code's authors, versioned in its tree, and usable without
running GX. This directory records their provenance, GX's own acceptance
tolerance, and a reference-validity analysis required by plan.md section 0.4.3.

The reference `.out.nc` files are large and are **not** vendored here. Point
`GX_PARITY_REF_DIR` at a GX checkout's `benchmarks/linear` directory. The
SHA-256 values below identify the exact files this analysis used.

## Provenance

| Field | Value |
|---|---|
| GX commit inspected | `bc2fe5523c23e3d0198181a3e3b7c8a482e25ba5` |
| GX upstream HEAD at the time | `3865a53778862e1686f414bf6f416339e24887c9` |
| Upstream | https://bitbucket.org/gyrokinetics/gx |
| GX's own acceptance gate | relative difference `gamma < 1e-3`, `omega < 5e-3` (`benchmarks/linear/ITG_cyclone/check.py`) |
| Precision | GX is compiled in single precision; GKX comparisons run float64 |

| Case | Deck | Reference output | SHA-256 |
|---|---|---|---|
| Cyclone ITG, s-alpha, adiabatic | `ITG_cyclone/itg_salpha_adiabatic_electrons.in` | `..._correct.out.nc` | `4887335d61b79495a39dd222d795c22c1a285aee6044eee8d0f5655da2da083e` |
| Cyclone ITG, Miller, adiabatic | `ITG_cyclone/itg_miller_adiabatic_electrons.in` | `..._correct.out.nc` | `96947cfcf118548ca4b4543d79f91cef4a26ab4ceac733b90f0719247b586fc5` |
| Cyclone ITG, Miller, kinetic electrons | `ITG_cyclone/itg_miller_kinetic_electrons.in` | `..._correct.out.nc` | `f9af8f2d4dfe453f0281aeeaa5a595ebcfa7f0fef809a452789faae4d3aa9fc0` |
| KBM, Miller | `KBM/kbm_miller.in` | `kbm_miller_correct.out.nc` | `3d3096f51da012a062364816cfd4919037fc183e98c41034f29027d4beed2ba8` |
| KAW, slab | `KAW/kaw_betahat10.0_kp0.01.in` | `..._correct.out.nc` | `4c98bdc4ef63b2a388f800c473f19568caf9f29f3bf68e144d8b2a7726be8238` |

W7-X (`ITG_w7x`) ships a deck, a `wout` and an `eik`, but **no** `*_correct.out.nc`;
any W7-X reference is therefore self-run (rank 5), not shipped.

## Reference validity: the linked-boundary end-damping clamp

GX launches three kernels with the same clamped grid
(`src/grad_parallel_linked.cu:169-173`):

```c
nn3 = grids_->Nz*grids_->Nm*grids_->Nl;   nt3 = min(nn3, 1);   nb3 = (nn3-1)/nt3 + 1;
dG_all = dim3(nb1, nb2, min(MAX_BLOCK_DIM_YZ, nb3));   // MAX_BLOCK_DIM_YZ = 65535
```

The comment above that line states: *"kernels have grid-stride loop in z to
handle case when z dim = MAX_BLOCK_DIM_YZ."* That is true of two of the three:

- `linkedCopyBackAll` and `linkedAccumulateBackAll` (`src/device_funcs.cu`) both
  open with
  `for (int idzlm = __umul24(blockIdx.z, blockDim.z) + threadIdx.z; idzlm < nz*nMoms; idzlm += __umul24(blockDim.z, gridDim.z))`.
- `dampEnds_linked` (`src/device_funcs.cu:3149`) instead takes a single index,
  `unsigned int idzlm = get_id3();` where
  `get_id3() = __umul24(blockIdx.z, blockDim.z) + threadIdx.z`, with **no loop
  and no use of `gridDim.z`**.

With `blockDim.z = 1`, `dampEnds_linked` therefore covers only
`idzlm` in `[0, 65535)`. When `Nz*Nl*Nm > 65535` the remaining indices are never
visited and those moments receive **no parallel end damping**. Because
`idm = (idzlm / Nz) / Nl`, the uncovered indices are the highest Hermite moments.

`Nz` is the *extended* parallel grid: `parameters.cu:755-757` rescales the input
`ntheta` as `ntgrid = ntheta/2 + (nperiod-1)*ntheta; Nz = 2*ntgrid`. For
`ntheta = 32, nperiod = 2` this gives `Nz = 96`, not 32.

### Per-case verdict

| Case | ntheta | nperiod | Nz | Nl | Nm | Nz·Nl·Nm | covered | undamped | share | undamped moments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cyclone s-alpha, adiabatic | 32 | 2 | 96 | 16 | 48 | 73,728 | 65,535 | 8,193 | 11.11% | m = 43…47 fully, m = 42 partly |
| Cyclone Miller, adiabatic | 32 | 2 | 96 | 16 | 48 | 73,728 | 65,535 | 8,193 | 11.11% | same |
| Cyclone Miller, kinetic e- | 32 | 2 | 96 | 16 | 48 | 73,728 | 65,535 | 8,193 | 11.11% | same |
| KBM Miller | 32 | 2 | 96 | 16 | 48 | 73,728 | 65,535 | 8,193 | 11.11% | same |
| W7-X ITG (self-run) | 256 | 1 | 256 | 8 | 16 | 32,768 | 32,768 | 0 | 0% | none |

Only non-zonal modes are damped in the first place (`idy > 0` in the kernel), so
this affects the driven modes the parity scans measure.

### What follows for GKX

1. The four shipped tokamak goldens are **provisional** references in
   `tools/evidence_ledger.toml`: they are legitimate GX output, but the physics
   they encode at high Hermite index is not the physics the deck requests.
2. GKX's published Cyclone (5.51% / 6.83% in gamma) and KBM (20.0% in gamma)
   disagreements are therefore **not yet attributable to GKX**. Closing those
   rows requires regenerating the reference from a repaired GX build and
   labeling it, per plan.md sections 0.4.3 and 1.1.
3. W7-X is unaffected and stays a valid comparison at its own rank.
4. The defect is reported upstream; see `upstream_report.md` in this directory.

Nothing here is a claim that GKX is correct. A reference-code defect makes the
comparison uninformative, not the comparison's other side right.
