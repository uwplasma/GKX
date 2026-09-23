"""XLA:CPU FFT thunk RET_CHECK on a non-default layout inherited from an all-gather.

Run with:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python xla_cpu_fft_layout_repro.py
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()), ("d",))
sharding = NamedSharding(mesh, P(None, "d", None))  # split a NON-leading, non-FFT axis

x = np.random.default_rng(0).standard_normal((4, 8, 16)).astype(np.complex64)


@jax.jit
def f(c):
    return jnp.fft.fft(2.0 * jnp.fft.fft(c, axis=-1), axis=-1)


xs = jax.device_put(x, sharding)
compiled = f.lower(xs).compile()
for line in compiled.as_text().splitlines():
    if " fft(" in line or "all-gather(" in line:
        print(line.strip()[:120])

out = f(xs)  # dispatch is async: the error surfaces when the result is read
print(
    np.max(np.abs(np.asarray(out) - np.fft.fft(2.0 * np.fft.fft(x, axis=-1), axis=-1)))
)
