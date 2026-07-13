"""Velocity-parallel linear RHS helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
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
    linear_rhs_electrostatic_species_sharded,
)
from spectraxgk.solvers.linear.parallel_streaming import *  # noqa: F403
from spectraxgk.solvers.linear.parallel_streaming import (
    __all__ as _streaming_all,
    linear_rhs_electrostatic_species_hermite_sharded,
    linear_rhs_streaming_electrostatic_velocity_sharded,
    linear_rhs_streaming_velocity_sharded,
)


@dataclass(frozen=True)
class _ParallelLinearRoute:
    strategy: str
    backend: str
    axis: str
    num_devices: int | None


def _normalize_parallel_token(value: Any, default: str) -> str:
    return str(value if value is not None else default).lower().replace("-", "_")


def _parallel_linear_route(parallel: Any) -> _ParallelLinearRoute:
    return _ParallelLinearRoute(
        strategy=_normalize_parallel_token(
            getattr(parallel, "strategy", "serial"), "serial"
        ),
        backend=_normalize_parallel_token(getattr(parallel, "backend", "auto"), "auto"),
        axis=_normalize_parallel_token(getattr(parallel, "axis", "hermite"), "hermite"),
        num_devices=getattr(parallel, "num_devices", None),
    )


def _use_serial_linear_route(parallel: Any | None) -> bool:
    return parallel is None or _parallel_linear_route(parallel).strategy == "serial"


def _is_mixed_electrostatic_terms(terms: LinearTerms | None) -> bool:
    """Return whether terms need no conserving or electromagnetic collectives."""

    active = terms or LinearTerms()
    return not any(
        float(value) != 0.0 for value in (active.collisions, active.apar, active.bpar)
    )


def _serial_linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    *,
    use_jit: bool,
    use_custom_vjp: bool,
    dt: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from spectraxgk.operators.linear.rhs import linear_rhs_cached

    return linear_rhs_cached(
        G,
        cache,
        params,
        terms=terms,
        use_jit=use_jit,
        use_custom_vjp=use_custom_vjp,
        dt=dt,
    )


def _require_hermite_axis(route: _ParallelLinearRoute, message: str) -> None:
    if route.axis not in {"m", "hermite"}:
        raise NotImplementedError(message)


def _resolve_velocity_backend(
    route: _ParallelLinearRoute,
    terms: LinearTerms | None,
    *,
    state_ndim: int,
) -> _ParallelLinearRoute:
    if route.backend != "auto":
        return route
    if route.axis in {"species_hermite", "s_m", "mixed"} and state_ndim == 6:
        if _is_mixed_electrostatic_terms(terms):
            return _ParallelLinearRoute(
                strategy=route.strategy,
                backend="electrostatic_species_hermite",
                axis=route.axis,
                num_devices=route.num_devices,
            )
        raise NotImplementedError(
            "mixed species-Hermite routing currently supports collision-free "
            "electrostatic linear terms"
        )
    if route.axis in {"s", "species"} and state_ndim == 6:
        if _is_electrostatic_slice_terms(terms):
            return _ParallelLinearRoute(
                strategy=route.strategy,
                backend="electrostatic_species",
                axis=route.axis,
                num_devices=route.num_devices,
            )
        raise NotImplementedError(
            "species sharding currently supports electrostatic linear terms"
        )
    _require_hermite_axis(
        route, "velocity sharding supports only the Hermite axis or species axis"
    )
    if _is_electrostatic_slice_terms(terms):
        return _ParallelLinearRoute(
            strategy=route.strategy,
            backend="electrostatic_linear_slices",
            axis=route.axis,
            num_devices=route.num_devices,
        )
    raise NotImplementedError(
        "backend='auto' can only select gated electrostatic velocity routes; "
        "disable collision/EM/end-damping terms or request an explicit backend"
    )


def _streaming_velocity_rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    route: _ParallelLinearRoute,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _require_hermite_axis(
        route,
        "streaming-only velocity sharding currently supports only the Hermite axis",
    )
    if not _is_streaming_only_terms(terms):
        raise NotImplementedError(
            "velocity streaming route requires streaming-only LinearTerms"
        )
    return linear_rhs_streaming_velocity_sharded(
        G,
        cache,
        params,
        num_devices=route.num_devices,
    )


def _streaming_electrostatic_velocity_rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    route: _ParallelLinearRoute,
    *,
    use_custom_vjp: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _require_hermite_axis(
        route,
        "electrostatic streaming velocity sharding currently supports only the Hermite axis",
    )
    if not _is_streaming_only_terms(terms):
        raise NotImplementedError(
            "electrostatic velocity streaming route requires streaming-only LinearTerms"
        )
    return linear_rhs_streaming_electrostatic_velocity_sharded(
        G,
        cache,
        params,
        num_devices=route.num_devices,
        use_custom_vjp=use_custom_vjp,
    )


def _electrostatic_slice_velocity_rhs(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    route: _ParallelLinearRoute,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _require_hermite_axis(
        route,
        "electrostatic slice velocity sharding currently supports only the Hermite axis",
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
        num_devices=route.num_devices,
    )


def _velocity_parallel_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    route: _ParallelLinearRoute,
    *,
    use_custom_vjp: bool,
    dt: jnp.ndarray | float | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    route = _resolve_velocity_backend(route, terms, state_ndim=G.ndim)
    if route.backend == "electrostatic_species_hermite":
        if route.axis not in {"species_hermite", "s_m", "mixed"}:
            raise NotImplementedError(
                "mixed species-Hermite routing requires axis='species_hermite'"
            )
        if not _is_mixed_electrostatic_terms(terms):
            raise NotImplementedError(
                "mixed species-Hermite routing requires electrostatic terms "
                "without conserving collisions"
            )
        if G.ndim != 6:
            raise NotImplementedError(
                "mixed species-Hermite routing requires a multi-species 6D state"
            )
        if route.num_devices != 4:
            raise NotImplementedError(
                "the gated mixed species-Hermite mesh currently requires four devices"
            )
        return linear_rhs_electrostatic_species_hermite_sharded(
            G,
            cache,
            params,
            terms=terms,
            dt=dt,
            species_chunks=2,
            hermite_chunks=2,
        )
    if route.backend in {"electrostatic_species", "linear_electrostatic_species"}:
        if route.axis not in {"s", "species"}:
            raise NotImplementedError(
                "electrostatic species route requires axis='species'"
            )
        return linear_rhs_electrostatic_species_sharded(
            G, cache, params, terms=terms, num_devices=route.num_devices
        )
    if route.backend in {"streaming_only", "linear_streaming_only"}:
        return _streaming_velocity_rhs(G, cache, params, terms, route)
    if route.backend in {"streaming_electrostatic", "linear_streaming_electrostatic"}:
        return _streaming_electrostatic_velocity_rhs(
            G,
            cache,
            params,
            terms,
            route,
            use_custom_vjp=use_custom_vjp,
        )
    if route.backend in {"electrostatic_linear_slices", "linear_electrostatic_slices"}:
        return _electrostatic_slice_velocity_rhs(G, cache, params, terms, route)
    raise NotImplementedError(
        "parallel linear RHS currently supports only strategy='velocity' with gated electrostatic backends"
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
    if _use_serial_linear_route(parallel):
        return _serial_linear_rhs_cached(
            G,
            cache,
            params,
            terms=terms,
            use_jit=use_jit,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
        )

    route = _parallel_linear_route(parallel)
    if route.strategy == "velocity":
        return _velocity_parallel_rhs_cached(
            G,
            cache,
            params,
            terms,
            route,
            use_custom_vjp=use_custom_vjp,
            dt=dt,
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
