"""Velocity-parallel linear RHS helpers."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.solvers.linear.parallel_common import *  # noqa: F403
from spectraxgk.solvers.linear.parallel_common import (
    __all__ as _common_all,
    _is_electrostatic_slice_terms,
    _is_streaming_only_terms,
)
from spectraxgk.solvers.linear.parallel_electrostatic import *  # noqa: F403
from spectraxgk.solvers.linear.parallel_electrostatic import (
    __all__ as _electrostatic_all,
    linear_rhs_electrostatic_slices_velocity_sharded,
)
from spectraxgk.solvers.linear.parallel_streaming import *  # noqa: F403
from spectraxgk.solvers.linear.parallel_streaming import (
    __all__ as _streaming_all,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_streaming_velocity_sharded,
)


def linear_rhs_parallel_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    parallel: Any | None = None,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute linear RHS with an explicit, disabled-by-default parallel route.

    ``parallel=None`` and ``parallel.strategy="serial"`` are exact aliases for
    :func:`linear_rhs_cached`. The non-serial velocity routes are opt-in,
    Hermite-axis-only identity gates. ``backend="auto"`` selects the most
    complete currently gated electrostatic route when the term set is eligible;
    otherwise callers must request a narrower explicit backend.
    """

    from spectraxgk.operators.linear.rhs import linear_rhs_cached

    if (
        parallel is None
        or str(getattr(parallel, "strategy", "serial")).lower() == "serial"
    ):
        return linear_rhs_cached(
            G,
            cache,
            params,
            terms=terms,
            use_jit=use_jit,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
        )

    strategy = str(getattr(parallel, "strategy", "serial")).lower().replace("-", "_")
    backend = str(getattr(parallel, "backend", "auto")).lower().replace("-", "_")
    axis = str(getattr(parallel, "axis", "hermite")).lower().replace("-", "_")
    if strategy == "velocity" and backend == "auto":
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "velocity sharding currently supports only the Hermite axis"
            )
        if _is_electrostatic_slice_terms(terms):
            backend = "electrostatic_linear_slices"
        else:
            raise NotImplementedError(
                "backend='auto' can only select gated electrostatic velocity routes; "
                "disable collision/EM/end-damping terms or request an explicit backend"
            )
    if strategy == "velocity" and backend in {
        "streaming_only",
        "linear_streaming_only",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "streaming-only velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
        )
    if strategy == "velocity" and backend in {
        "streaming_electrostatic",
        "linear_streaming_electrostatic",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic streaming velocity sharding currently supports only the Hermite axis"
            )
        if not _is_streaming_only_terms(terms):
            raise NotImplementedError(
                "electrostatic velocity streaming route requires streaming-only LinearTerms"
            )
        return linear_rhs_streaming_electrostatic_velocity_sharded(
            G,
            cache,
            params,
            num_devices=getattr(parallel, "num_devices", None),
            use_custom_vjp=use_custom_vjp,
        )
    if strategy == "velocity" and backend in {
        "electrostatic_linear_slices",
        "linear_electrostatic_slices",
    }:
        if axis not in {"m", "hermite"}:
            raise NotImplementedError(
                "electrostatic slice velocity sharding currently supports only the Hermite axis"
            )
        if not _is_electrostatic_slice_terms(terms):
            raise NotImplementedError(
                "electrostatic slice route requires collision/EM terms to be disabled"
            )
        return linear_rhs_electrostatic_slices_velocity_sharded(
            G,
            cache,
            params,
            terms=terms,
            num_devices=getattr(parallel, "num_devices", None),
        )

    raise NotImplementedError(
        "parallel linear RHS currently supports only strategy='velocity' with gated electrostatic backends"
    )


__all__ = list(
    dict.fromkeys(
        [
            *_common_all,
            *_streaming_all,
            *_electrostatic_all,
            "linear_rhs_parallel_cached",
        ]
    )
)
