"""Shared construction fixtures for the linear-operator unit tests.

Every fixture here is a *factory* -- it returns a callable, because the tests in
this directory vary the thing they build (grid sizes, which term switches are
on, which stubbed RHS a diagnostics run sees) far more often than they repeat it
verbatim. A fixed-value fixture would only serve the handful of call sites that
happen to want its exact arguments; a factory serves all of them and keeps each
test's own numbers visible at its own call site.

No fixture here standardises a value a test was choosing for itself: every
dimension a test stated stays stated at its own call site, and the term
factories write out the zeros a literal used to spell, rather than inheriting
dataclass defaults. What they remove is the boilerplate that assembles those
choices into objects.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import SpectralGrid, build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.params import LinearTerms
from gkx.terms.config import TermConfig

#: Every switch :class:`LinearTerms` carries, in declaration order. Named here so
#: :func:`only_terms` fails loudly if the dataclass grows a field rather than
#: silently leaving the new one at its default.
_LINEAR_TERM_SWITCHES = (
    "streaming",
    "mirror",
    "curvature",
    "gradb",
    "diamagnetic",
    "collisions",
    "hypercollisions",
    "hyperdiffusion",
    "end_damping",
    "apar",
    "bpar",
)

#: The same, for :class:`TermConfig`, which adds the nonlinear switch.
_TERM_CONFIG_SWITCHES = _LINEAR_TERM_SWITCHES + ("nonlinear",)


@pytest.fixture
def cyclone_world():
    """Factory for the Cyclone flux-tube triple a linear test runs against.

    ``cyclone_world(Nx=8, Ny=6, Nz=8)`` returns ``(cfg, grid, geom)`` where
    ``cfg`` is ``CycloneBaseCase(grid=GridConfig(**kwargs))``, ``grid`` is
    ``build_spectral_grid(cfg.grid)`` and ``geom`` is
    ``SAlphaGeometry.from_config(cfg.geometry)``.

    Keywords go straight to :class:`~gkx.config.GridConfig` and nothing else is
    supplied, so the box size, boundary and everything past it are the caller's
    to state -- the only default in play is ``GridConfig``'s own. A fresh grid
    and geometry are built per call; nothing is shared between tests.
    """

    def _build(**grid_kwargs) -> tuple[CycloneBaseCase, SpectralGrid, SAlphaGeometry]:
        cfg = CycloneBaseCase(grid=GridConfig(**grid_kwargs))
        grid = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        return cfg, grid, geom

    return _build


@pytest.fixture
def spectral_grid():
    """Factory for ``build_spectral_grid(GridConfig(**kwargs))``.

    For the tests that want a bare spectral grid without a
    :class:`~gkx.config.CycloneBaseCase` around it. Keywords go straight to
    :class:`~gkx.config.GridConfig`; there are no defaults beyond that class's
    own.
    """

    def _build(**grid_kwargs) -> SpectralGrid:
        return build_spectral_grid(GridConfig(**grid_kwargs))

    return _build


@pytest.fixture
def only_terms():
    """Factory for a :class:`LinearTerms` with every switch off but the named ones.

    ``only_terms(streaming=1.0)`` is the eleven-line "all zeros except one"
    literal these tests keep writing out. Every switch not named is set to
    ``0.0`` *explicitly*, so the result never depends on the dataclass defaults
    -- which are not uniform (``bpar`` defaults to 1.0, ``hyperdiffusion`` to
    0.0). A call site that wants a default-on switch left on has to say so:
    ``only_terms(bpar=1.0)``.
    """

    def _build(**switches: float) -> LinearTerms:
        unknown = set(switches) - set(_LINEAR_TERM_SWITCHES)
        if unknown:
            raise TypeError(f"unknown LinearTerms switches: {sorted(unknown)}")
        values = dict.fromkeys(_LINEAR_TERM_SWITCHES, 0.0)
        values.update(switches)
        return LinearTerms(**values)

    return _build


@pytest.fixture
def only_term_config():
    """Factory for a :class:`TermConfig` with every switch off but the named ones.

    The :func:`only_terms` contract, for the modular term-assembly config: all
    twelve switches -- ``nonlinear`` included -- are written as ``0.0`` unless
    the caller names them.
    """

    def _build(**switches: float) -> TermConfig:
        unknown = set(switches) - set(_TERM_CONFIG_SWITCHES)
        if unknown:
            raise TypeError(f"unknown TermConfig switches: {sorted(unknown)}")
        values = dict.fromkeys(_TERM_CONFIG_SWITCHES, 0.0)
        values.update(switches)
        return TermConfig(**values)

    return _build


@pytest.fixture
def diagnostics_cache():
    """Factory for the stub cache ``integrate_linear_diagnostics`` reads.

    ``diagnostics_cache((1, 2, 2, 1, 1, 2))`` returns a namespace whose
    ``lb_lam`` is float32 zeros of that shape and whose ``Jl`` is float32 ones
    of the same shape with the Hermite axis dropped -- axis 2 for a
    species-resolved six-axis shape, axis 1 for the five-axis one. Those are the
    only two attributes the diagnostics path touches.
    """

    def _build(lb_lam_shape: tuple[int, ...]) -> SimpleNamespace:
        if len(lb_lam_shape) not in (5, 6):
            raise ValueError("lb_lam_shape must have five or six axes")
        hermite_axis = 2 if len(lb_lam_shape) == 6 else 1
        jl_shape = lb_lam_shape[:hermite_axis] + lb_lam_shape[hermite_axis + 1 :]
        return SimpleNamespace(
            lb_lam=jnp.zeros(lb_lam_shape, dtype=jnp.float32),
            Jl=jnp.ones(jl_shape, dtype=jnp.float32),
        )

    return _build


@pytest.fixture
def patch_diagnostics_kernels(monkeypatch):
    """Factory that stubs the two kernels a diagnostics step calls out to.

    Calling it patches ``hypercollision_damping`` to zeros shaped like
    ``cache.lb_lam``, and ``linear_rhs_cached`` to return ``(rhs(G), ones((1, 1,
    2), complex64))``. ``rhs`` defaults to :func:`jax.numpy.ones_like`; pass
    ``jnp.zeros_like`` for the no-drive variant. Both patches are undone by
    ``monkeypatch`` at teardown.
    """

    def _patch(rhs=jnp.ones_like) -> None:
        monkeypatch.setattr(
            "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
            lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
        )
        monkeypatch.setattr(
            "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
            lambda G, cache, params, **kwargs: (
                rhs(G),
                jnp.ones((1, 1, 2), dtype=jnp.complex64),
            ),
        )

    return _patch
