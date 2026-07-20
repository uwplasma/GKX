"""Domain-decomposition contracts for nonlinear parallel diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import jax


@dataclass(frozen=True)
class NonlinearDomainDecompositionPlan:
    """Static decomposition plan for a local nonlinear state-domain local-stencil diagnostic."""

    state_shape: tuple[int, ...]
    axis: int
    chunk_sizes: tuple[int, ...]
    halo: int = 1

    @property
    def num_domains(self) -> int:
        """Return the number of state-domain chunks."""

        return len(self.chunk_sizes)

    @property
    def domain_size(self) -> int:
        """Return the global size of the decomposed axis."""

        return self.state_shape[self.axis]

    @property
    def offsets(self) -> tuple[int, ...]:
        """Return chunk start offsets along the decomposed axis."""

        offsets: list[int] = []
        start = 0
        for size in self.chunk_sizes:
            offsets.append(start)
            start += size
        return tuple(offsets)

    @property
    def chunk_bounds(self) -> tuple[tuple[int, int], ...]:
        """Return half-open ``(start, stop)`` bounds for owned chunk cells."""

        return tuple(
            (offset, offset + size)
            for offset, size in zip(self.offsets, self.chunk_sizes, strict=True)
        )

    @property
    def boundary_indices(self) -> tuple[int, ...]:
        """Return global cells that touch a decomposed halo interface."""

        if (
            not self.state_shape
            or not (0 <= int(self.axis) < len(self.state_shape))
            or len(self.chunk_sizes) <= 1
        ):
            return ()
        domain_size = int(self.state_shape[int(self.axis)])
        if domain_size <= 0:
            return ()

        indices: set[int] = set()
        for offset in self.offsets:
            indices.add((offset - 1) % domain_size)
            indices.add(offset % domain_size)
        return tuple(sorted(indices))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the decomposition plan."""

        return asdict(self)

    def decomposition_metadata(self) -> dict[str, Any]:
        """Return derived metadata for diagnostic decomposition artifacts."""

        return {
            **self.to_dict(),
            "num_domains": self.num_domains,
            "domain_size": self.domain_size,
            "offsets": self.offsets,
            "chunk_bounds": self.chunk_bounds,
            "boundary_indices": self.boundary_indices,
        }


@dataclass(frozen=True)
class NonlinearDomainIdentityReport:
    """Numerical identity report for a decomposed nonlinear local-stencil step."""

    gate_name: str
    plan: NonlinearDomainDecompositionPlan
    atol: float
    rtol: float
    max_abs_error: float
    max_rel_error: float
    plan_valid: bool
    blocked_reasons: tuple[str, ...]
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    boundary_max_abs_error: float = 0.0
    boundary_max_rel_error: float = 0.0
    boundary_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the identity report."""

        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


@dataclass(frozen=True)
class NonlinearDomainTransportWindowReport:
    """Transport-window identity report for the nonlinear domain local-stencil diagnostic."""

    gate_name: str
    plan: NonlinearDomainDecompositionPlan
    steps: int
    dt: float
    atol: float
    rtol: float
    max_abs_state_error: float
    max_rel_state_error: float
    max_abs_boundary_error: float
    max_rel_boundary_error: float
    mass_trace_max_abs_error: float
    mass_trace_max_rel_error: float
    free_energy_trace_max_abs_error: float
    free_energy_trace_max_rel_error: float
    flux_proxy_trace_max_abs_error: float
    flux_proxy_trace_max_rel_error: float
    serial_mass_drift: float
    decomposed_mass_drift: float
    serial_free_energy_drift: float
    decomposed_free_energy_drift: float
    plan_valid: bool
    blocked_reasons: tuple[str, ...]
    identity_passed: bool
    decomposed_path_enabled: bool
    claim_scope: str
    boundary_indices: tuple[int, ...] = ()
    serial_mass_trace: tuple[float, ...] = ()
    decomposed_mass_trace: tuple[float, ...] = ()
    serial_free_energy_trace: tuple[float, ...] = ()
    decomposed_free_energy_trace: tuple[float, ...] = ()
    serial_flux_proxy_trace: tuple[float, ...] = ()
    decomposed_flux_proxy_trace: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the transport-window report."""

        data = asdict(self)
        data["plan"] = self.plan.to_dict()
        return data


_NONLINEAR_DOMAIN_GATE_NAME = "nonlinear_domain_local_stencil_identity"
_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME = "nonlinear_domain_transport_window_identity"
_NONLINEAR_DOMAIN_CLAIM_SCOPE = (
    "diagnostic nonlinear state-domain identity gate only; "
    "bounded local-stencil diagnostic with no production routing or speedup claim"
)
_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE = (
    "diagnostic nonlinear state-domain transport-window identity gate only; "
    "serial-vs-halo-decomposed state, boundary, mass, free-energy, and flux-proxy "
    "traces with no production routing or speedup claim"
)


def _nonlinear_domain_plan_blockers(
    plan: NonlinearDomainDecompositionPlan,
) -> tuple[str, ...]:
    blockers: list[str] = []

    if not plan.state_shape:
        blockers.append("state_shape_empty")
        axis_is_valid = False
    else:
        axis_is_valid = 0 <= int(plan.axis) < len(plan.state_shape)

    if any(int(size) <= 0 for size in plan.state_shape):
        blockers.append("state_shape_non_positive")
    if not axis_is_valid:
        blockers.append("axis_not_canonical")
    if int(plan.halo) != 1:
        blockers.append("unsupported_halo")
    if not plan.chunk_sizes:
        blockers.append("chunk_sizes_empty")
    if any(int(size) <= 0 for size in plan.chunk_sizes):
        blockers.append("chunk_size_non_positive")
    if axis_is_valid and plan.chunk_sizes:
        domain_size = int(plan.state_shape[int(plan.axis)])
        if sum(int(size) for size in plan.chunk_sizes) != domain_size:
            blockers.append("chunk_sizes_do_not_cover_axis")

    return tuple(blockers)


def _nonlinear_domain_identity_blockers(
    serial_state: jax.Array,
    decomposed_state: jax.Array,
    plan: NonlinearDomainDecompositionPlan,
) -> tuple[str, ...]:
    blockers = list(_nonlinear_domain_plan_blockers(plan))
    serial_shape = tuple(int(size) for size in serial_state.shape)
    decomposed_shape = tuple(int(size) for size in decomposed_state.shape)

    if serial_shape != plan.state_shape:
        blockers.append("serial_shape_does_not_match_plan")
    if decomposed_shape != serial_shape:
        blockers.append("decomposed_shape_does_not_match_serial")

    return tuple(blockers)


__all__ = [
    "NonlinearDomainDecompositionPlan",
    "NonlinearDomainIdentityReport",
    "NonlinearDomainTransportWindowReport",
    "_NONLINEAR_DOMAIN_CLAIM_SCOPE",
    "_NONLINEAR_DOMAIN_GATE_NAME",
    "_NONLINEAR_DOMAIN_TRANSPORT_CLAIM_SCOPE",
    "_NONLINEAR_DOMAIN_TRANSPORT_GATE_NAME",
    "_nonlinear_domain_identity_blockers",
    "_nonlinear_domain_plan_blockers",
]
