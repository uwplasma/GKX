# Upstream defect report: `dampEnds_linked` misses the grid-stride loop

Prepared for https://bitbucket.org/gyrokinetics/gx (plan.md Phase 0.4.4). Filed
by the GKX project as a reference-validity finding; it is not a GKX bug report.

## Summary

`dampEnds_linked` is launched with a z grid clamped to `MAX_BLOCK_DIM_YZ`
(65535) but does not implement the grid-stride loop that the neighbouring
kernels use and that the launch site's own comment says is present. When
`Nz*Nl*Nm > 65535`, the parallel end-damping layer is silently skipped for the
highest Hermite moments.

## Where

`src/grad_parallel_linked.cu:169-173` sets the shared launch grid:

```c
// z block dim = min(Nz*Nl*Nm, MAX_BLOCK_DIM_YZ). kernels have grid-stride loop
// in z to handle case when z dim = MAX_BLOCK_DIM_YZ.
nn3 = grids_->Nz*grids_->Nm*grids_->Nl;   nt3 = min(nn3, 1);   nb3 = (nn3-1)/nt3 + 1;
dG_all = dim3(nb1, nb2, min(MAX_BLOCK_DIM_YZ, nb3));
```

`src/grad_parallel_linked.cu:391` launches the damping kernel on that grid:

```c
dampEnds_linked <<<dG_all, dB_all>>> (...);
```

`src/device_funcs.cu` then differs between kernels:

- `linkedCopyBackAll` and `linkedAccumulateBackAll` open with
  `for (int idzlm = __umul24(blockIdx.z, blockDim.z) + threadIdx.z; idzlm < nz*nMoms; idzlm += __umul24(blockDim.z, gridDim.z))`.
- `dampEnds_linked` (line 3149) uses `unsigned int idzlm = get_id3();` with no
  loop, where `get_id3() = __umul24(blockIdx.z, blockDim.z) + threadIdx.z`.

With `blockDim.z = 1` the damping kernel covers `idzlm` in `[0, 65535)` only.

## Reproduction

Any linked-boundary run with `Nz*Nl*Nm > 65535`. GX's own shipped benchmarks
qualify, because `parameters.cu:755-757` rescales `ntheta`:
`ntgrid = ntheta/2 + (nperiod-1)*ntheta; Nz = 2*ntgrid`.

`benchmarks/linear/ITG_cyclone/itg_salpha_adiabatic_electrons.in` has
`ntheta = 32, nperiod = 2, nlaguerre = 16, nhermite = 48`, giving `Nz = 96` and
`Nz*Nl*Nm = 73,728`. Indices `65535…73,727` are never visited, so Hermite
moments `m = 43…47` (and part of `m = 42`) receive no end damping. The same
holds for `itg_miller_adiabatic_electrons.in`,
`itg_miller_kinetic_electrons.in` and `KBM/kbm_miller.in`.

`benchmarks/linear/ITG_w7x` (`Nz = 256`, `Nl = 8`, `Nm = 16`, product 32,768) is
below the clamp and is unaffected.

A direct check: instrument the kernel to record `max(idzlm)` actually visited,
or compare `G` before and after `applyBCs` at `m = 47` versus `m = 0`; only the
low-`m` slice changes.

## Suggested fix

Give `dampEnds_linked` the same loop its siblings use:

```c
for (int idzlm = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;
     idzlm < nz*nMoms;
     idzlm += __umul24(blockDim.z, gridDim.z)) {
  ...existing body...
}
```

`dampEnds_linkedNTFT` (`src/device_funcs.cu:3259`) should be checked the same
way; it is launched from `src/grad_parallel_NTFT.cu:328`.

## Impact

Runs below the clamp are unaffected. Above it, the absorbing layer is absent
exactly where Hermite recurrence is strongest, so results can depend on
`Nl`/`Nm` in a way that looks like a velocity-space convergence failure. Shipped
reference outputs produced above the clamp encode this behaviour.
