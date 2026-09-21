"""Q30: why an eager ``lax.scan`` recompiles, in four lines and no GKX.

``jax._src.core.ClosedJaxpr`` inherits ``object.__eq__`` and ``object.__hash__``,
so a jaxpr compares by identity. An eager ``lax.scan`` traces its body into a
fresh jaxpr on every call and is then dispatched on it, so every lowering cache
below misses --- whatever the body closes over, and whether or not anything
about the problem changed. This script prints the identity of those dunders and
counts the XLA compilations of three identical calls.

Usage: python jaxpr_identity.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax._src.compiler as compiler
import jax._src.core as core

_COMPILES = {"n": 0}
_original = compiler.backend_compile_and_load


def _counting(*args, **kwargs):
    _COMPILES["n"] += 1
    return _original(*args, **kwargs)


compiler.backend_compile_and_load = _counting

print("ClosedJaxpr.__eq__  ", core.ClosedJaxpr.__eq__)
print("ClosedJaxpr.__hash__", core.ClosedJaxpr.__hash__)


def eager_scan(weight):
    """One eager scan whose body closes over a device array."""

    def body(carry, x):
        return carry * weight + x, None

    xs = jnp.arange(4, dtype=jnp.float32)
    return jax.lax.scan(body, jnp.float32(1.0), xs)[0]


def eager_scan_constant_body():
    """The same scan with a Python constant, to rule the closure out."""

    def body(carry, x):
        return carry * 0.5 + x, None

    xs = jnp.arange(4, dtype=jnp.float32)
    return jax.lax.scan(body, jnp.float32(1.0), xs)[0]


weight = jnp.float32(0.5)
for label, call in (
    ("closed-over array", lambda: eager_scan(weight)),
    ("python constant", eager_scan_constant_body),
):
    for call_index in range(3):
        _COMPILES["n"] = 0
        jax.block_until_ready(call())
        print(f"{label}: call {call_index} compiled {_COMPILES['n']} module(s)")
