# Draft upstream issue for openxla/xla (NOT filed; for the maintainer to file)

**Title:** XLA:CPU: FFT fails `RET_CHECK IsMonotonicWithDim0Major` when its operand is an all-gather along a non-leading dimension

**Summary**

On the CPU backend, an SPMD-partitioned program in which an FFT consumes an all-gather along a
non-leading dimension fails at execution time with

```
INTERNAL: RET_CHECK failure (xla/backends/cpu/runtime/fft_thunk.cc:168)
LayoutUtil::IsMonotonicWithDim0Major(input_shape_.layout())
```

`CpuLayoutAssignment::AddBackendConstraints` (`xla/service/cpu/cpu_layout_assignment.cc`) pins
every all-gather's result layout so that the gather dimension is the most major. That is correct,
because XLA:CPU only implements that case. FFT, however, only gets the generic row-major operand
constraint, and here that constraint does not win. After layout assignment the FFT consumes the
all-gather's non-default layout directly, and `FftThunk` rejects it at run time. Dot, Convolution
and TopK avoid this through `OperandsAndResultMustHaveRowMajorLayout`; FFT is not on that list.

The failure only appears when the result is read, because dispatch is asynchronous.

**Reproducer** (4 fake CPU devices; `xla_cpu_fft_layout_repro.py` next to this file)

```python
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("d",))
sharding = NamedSharding(mesh, P(None, "d", None))  # split a non-leading, non-FFT axis
x = np.random.default_rng(0).standard_normal((4, 8, 16)).astype(np.complex64)

@jax.jit
def f(c):
    return jnp.fft.fft(2.0 * jnp.fft.fft(c, axis=-1), axis=-1)

print(np.asarray(f(jax.device_put(x, sharding))))
```
`XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python repro.py`

Optimized HLO (JAX 0.10.2):
```
%all-gather = c64[4,8,16]{2,0,1} all-gather(%copy), ..., dimensions={1}
%fft.6      = c64[4,8,16]{2,0,1} fft(%all-gather), fft_type=FFT, fft_length={16}   <- non-default layout
%all-gather.1 = c64[4,8,16]{2,0,1} all-gather(%broadcast_multiply_fusion), ...
%fft.7      = c64[4,8,16]{2,1,0} fft(%copy.1), ...                                  <- copy inserted here
```

**Observed on:** jax/jaxlib 0.10.2 and 0.11.1 (macOS arm64, CPU), with 2, 4 and 8 fake devices,
using the Shardy partitioner (the default). The failure needs:
- the split axis is not the leading axis (splitting axis 0 of the same array runs fine);
- two FFTs in sequence, so that the first FFT's result layout is free (a single FFT that is the
  entry ROOT gets a copy, and runs).

**Expected:** a copy to the default layout before the FFT, or an FFT thunk that accepts
permuted layouts.

**Suggested fix:** add `kFft` to `OperandsAndResultMustHaveRowMajorLayout` in
`cpu_layout_assignment.cc`, or make the operand constraint mandatory for FFT.

**Related observation (performance, not a crash):** both the Shardy and GSPMD partitioners
all-gather the FFT operand even when only batch (non-transformed) dimensions are split. For
example, `jax.lax.fft` on `c64[8,16]` split on axis 0, with `in_shardings` and `out_shardings` both
split on axis 0, still emits `all-gather(dimensions={0})` before the FFT. A batch-sharded FFT
could run locally.

**Downstream workaround used in GKX:** gather explicitly inside `shard_map`, along a leading axis
(`moveaxis(axis -> 0)`, `all_gather(axis=0, tiled=True)`, then `moveaxis` back). The gathered array
then has the default layout, and XLA inserts a real transpose before the FFT.
