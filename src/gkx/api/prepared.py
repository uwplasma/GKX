"""The public prepared-simulation object.

``gkx.prepare`` used to return the solver's internal
``PreparedExplicitNonlinearDiagnostics`` and to refuse linear cases outright,
so the reusable-execution concept existed only for half the product and only
as a private type. :class:`PreparedSimulation` is the public object the API
contract describes: it accepts linear and nonlinear cases, carries the static
topology and the compilation and cache metadata that decide whether a second
call is cheap, and exposes typed ``solve``, ``scan``, ``value_and_grad``,
``warmup``, ``estimate_memory`` and ``summary``.

It owns no numerics. Every method delegates to the runtime path the case would
have taken anyway, so preparing a case and solving it cannot disagree with
solving it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np


def _case_kind(case: Any) -> str:
    """Return ``"nonlinear"`` or ``"linear"`` for a validated case."""

    return "nonlinear" if bool(case.physics.nonlinear) else "linear"


def _state_shape(case: Any, *, n_laguerre: int, n_hermite: int) -> tuple[int, ...]:
    """Return the distribution-array shape this case will allocate."""

    grid = case.grid
    n_species = sum(1 for s in case.species if getattr(s, "kinetic", False))
    n_ky = int(grid.Ny) // 2 + 1
    return (
        max(n_species, 1),
        int(n_laguerre),
        int(n_hermite),
        n_ky,
        int(grid.Nx),
        int(grid.Nz),
    )


@dataclass(frozen=True)
class PreparedSimulation:
    """A reusable, compiled simulation for one case.

    Construct with :func:`gkx.prepare`. The object is frozen: deriving a
    different topology means preparing a different case, which is what keeps
    the compiled callables and the reported metadata describing the same run.
    """

    case: Any
    kind: str
    state_shape: tuple[int, ...]
    n_laguerre: int
    n_hermite: int
    options: dict[str, Any] = field(default_factory=dict)
    _backend: Any = None

    # -- execution -------------------------------------------------------

    def solve(self, parameters: Any = None, initial_state: Any = None) -> Any:
        """Run this simulation and return its typed result.

        ``parameters`` is reserved for the differentiable path and must be
        ``None`` here; a prepared object must not silently solve a different
        physics problem than the one it reports.
        """

        if parameters is not None:
            raise NotImplementedError(
                "solve(parameters=...) requires the differentiable prepared path; "
                "use value_and_grad for parameter sensitivities"
            )
        if self.kind == "nonlinear" and self._backend is not None:
            if initial_state is None:
                return self._backend.run()
            return self._backend.run(initial_state)
        from gkx import runtime as _runtime

        if initial_state is not None:
            raise NotImplementedError(
                "linear prepared solves do not accept an initial state yet"
            )
        return _runtime.solve(self.case, **self.options)

    def scan(
        self,
        parameter: str,
        values: Sequence[float],
        *,
        parallel: str = "auto",
    ) -> Any:
        """Solve this case once per value of a named scalar parameter."""

        from gkx import runtime as _runtime

        if parameter != "ky":
            raise NotImplementedError(
                f"prepared scan supports parameter='ky'; got {parameter!r}"
            )
        workers = 1 if parallel in {"auto", "serial", "off"} else int(parallel)
        return _runtime.run_runtime_scan(self.case, list(values), workers=workers)

    def value_and_grad(self, objective: Callable[..., Any], parameters: Any) -> Any:
        """Return ``(value, gradient)`` of ``objective`` at ``parameters``."""

        import jax

        return jax.value_and_grad(objective)(parameters)

    # -- introspection ---------------------------------------------------

    def warmup(self) -> "PreparedSimulation":
        """Force compilation now so a later timed call measures execution.

        Returns ``self`` so a caller can chain. Preparing already builds the
        compiled callables for the nonlinear path, so this is a no-op there;
        it exists so timing code does not have to know which path it holds.
        """

        return self

    def estimate_memory(self) -> dict[str, Any]:
        """Estimate the resident bytes this simulation's arrays require.

        The estimate covers the distribution array and the working copies an
        explicit stage holds live at once. It is a floor, not a ceiling: it
        does not model the diagnostic buffers a long sampled run accumulates,
        and it says so rather than pretending to a precision it lacks.
        """

        elements = int(np.prod(self.state_shape))
        itemsize = 8 if self.precision == "float64" else 4
        # A complex spectral state, and the stages an explicit step holds live.
        state_bytes = elements * itemsize * 2
        stages = 4 if str(self.case.time.method).lower() in {"rk4", "rk45"} else 3
        return {
            "state_shape": self.state_shape,
            "elements": elements,
            "state_bytes": state_bytes,
            "working_set_bytes": state_bytes * stages,
            "precision": self.precision,
            "is_floor_not_ceiling": True,
        }

    @property
    def precision(self) -> str:
        """Return the JAX floating precision this simulation will run in."""

        import jax

        return "float64" if jax.config.read("jax_enable_x64") else "float32"

    @property
    def devices(self) -> tuple[str, ...]:
        """Return the visible JAX devices as their platform strings."""

        import jax

        return tuple(f"{d.platform}:{d.id}" for d in jax.devices())

    def compilation_metadata(self) -> dict[str, Any]:
        """Report whether a second run of this topology can reuse a cache."""

        import gkx.compilation_cache as compilation_cache

        enabled = compilation_cache.compilation_cache_enabled()
        return {
            "persistent_cache_enabled": enabled,
            "compiled_at_prepare": self.kind == "nonlinear",
            "devices": self.devices,
            "precision": self.precision,
        }

    def summary(self) -> dict[str, Any]:
        """Return the topology, precision, and cache facts that identify this."""

        memory = self.estimate_memory()
        return {
            "kind": self.kind,
            "state_shape": self.state_shape,
            "n_laguerre": self.n_laguerre,
            "n_hermite": self.n_hermite,
            "geometry_model": self.case.geometry.model,
            "precision": self.precision,
            "devices": self.devices,
            "working_set_bytes": memory["working_set_bytes"],
            **self.compilation_metadata(),
        }

    def print_summary(self, *, stream: Any = None) -> None:
        """Print the summary one field per line."""

        for key, value in self.summary().items():
            print(f"{key}: {value}", file=stream)


def prepare_simulation(case: Any, **options: Any) -> PreparedSimulation:
    """Build a :class:`PreparedSimulation` for ``case``.

    A nonlinear case compiles its scan closure here, which is what makes a
    later ``solve`` cheap. A linear case is validated and described but not
    compiled, because the linear runtime chooses its solver per call; the
    difference is reported by ``compiled_at_prepare`` rather than hidden.
    """

    case.validate()
    kind = _case_kind(case)
    # Nl/Nm are run-time selections, not case fields: the deck carries them in
    # its [run] table and the runtime resolves them per call. Mirror the
    # runtime's own defaults so the reported topology matches what solve builds.
    n_laguerre = int(options.get("Nl") or 4)
    n_hermite = int(options.get("Nm") or 8)
    backend = None
    if kind == "nonlinear":
        from gkx import runtime as _runtime

        backend = _runtime.prepare(case, **options)
    return PreparedSimulation(
        case=case,
        kind=kind,
        state_shape=_state_shape(case, n_laguerre=n_laguerre, n_hermite=n_hermite),
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        options=dict(options),
        _backend=backend,
    )
