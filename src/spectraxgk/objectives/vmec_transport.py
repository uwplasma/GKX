"""Public VMEC-JAX transport-objective facade.

Implementation lives in smaller domain modules:
``vmec_transport_config`` owns optional-backend path policy and configuration,
``vmec_transport_tables`` owns sample-table/reduction kernels, and
``vmec_transport_branch`` owns eigenbranch locality gates.  This facade keeps
existing imports and monkeypatch seams stable for examples and tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import os as os
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import (
    flux_tube_geometry_from_vmec_boozer_state,
    prewarm_vmec_boozer_equal_arc_cache,
)
from spectraxgk.objectives.core import (
    solver_growth_rate_from_geometry,
    solver_linear_operator_matrix_from_geometry,
)
from spectraxgk.objectives import vmec_transport_branch as _branch
from spectraxgk.objectives import vmec_transport_config as _config
from spectraxgk.objectives import vmec_transport_tables as _tables
from spectraxgk.objectives.vmec_transport_config import (
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
    _reference_wout_from_context,
)
from spectraxgk.objectives.vmec_transport_tables import (
    _apply_objective_transform as _apply_objective_transform,
    _geometry_transport_weights as _geometry_transport_weights,
    _solver_table_to_nonlinear_window_proxy as _solver_table_to_nonlinear_window_proxy,
    _static_grid_options_from_ky_values as _static_grid_options_from_ky_values,
    _surface_chunk_sample_sets as _surface_chunk_sample_sets,
)


_module_search_root = _config._module_search_root


def _pin_current_optional_backend_paths() -> None:
    """Pin optional backend paths through the public facade seam."""

    _config._module_search_root = _module_search_root
    return _config._pin_current_optional_backend_paths()


def _sync_table_dependencies() -> None:
    """Route implementation dependencies through facade-level monkeypatch seams."""

    _tables._pin_current_optional_backend_paths = _pin_current_optional_backend_paths
    _tables.flux_tube_geometry_from_vmec_boozer_state = (
        flux_tube_geometry_from_vmec_boozer_state
    )
    _tables.solver_growth_rate_from_geometry = solver_growth_rate_from_geometry


def _sync_branch_dependencies() -> None:
    """Route branch-locality dependencies through facade-level monkeypatch seams."""

    _branch._pin_current_optional_backend_paths = _pin_current_optional_backend_paths
    _branch.flux_tube_geometry_from_vmec_boozer_state = (
        flux_tube_geometry_from_vmec_boozer_state
    )
    _branch.solver_linear_operator_matrix_from_geometry = (
        solver_linear_operator_matrix_from_geometry
    )


def _transport_feature_table_from_state(*args: Any, **kwargs: Any) -> jnp.ndarray:
    """Facade wrapper for VMEC/Boozer solver-objective sample tables."""

    _sync_table_dependencies()
    return _tables._transport_feature_table_from_state(*args, **kwargs)


def _transport_objective_raw_value_from_state(*args: Any, **kwargs: Any) -> jnp.ndarray:
    """Facade wrapper for untransformed VMEC transport objective values."""

    _sync_table_dependencies()
    return _tables._transport_objective_raw_value_from_state(*args, **kwargs)


def _chunked_transport_objective_raw_value_from_state(
    *args: Any, **kwargs: Any
) -> jnp.ndarray:
    """Facade wrapper for chunked VMEC transport objective values."""

    _sync_table_dependencies()
    return _tables._chunked_transport_objective_raw_value_from_state(*args, **kwargs)


def vmec_jax_transport_objective_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig | None = None,
) -> jnp.ndarray:
    """Evaluate a scalar SPECTRAX-GK transport objective from a VMEC-JAX state."""

    _sync_table_dependencies()
    return _tables.vmec_jax_transport_objective_from_state(
        state,
        static,
        indata,
        wout_reference,
        config,
    )


def vmec_jax_transport_growth_branch_locality_report_from_states(
    *args: Any,
    **kwargs: Any,
) -> dict[str, object]:
    """Facade wrapper for VMEC/Boozer dominant-growth branch locality gates."""

    _sync_branch_dependencies()
    return _branch.vmec_jax_transport_growth_branch_locality_report_from_states(
        *args, **kwargs
    )


@dataclass(frozen=True)
class VMECJAXSpectraxTransportObjective:
    """Least-squares objective object for ``vmec_jax`` QA optimizers."""

    config: VMECJAXTransportObjectiveConfig = field(
        default_factory=VMECJAXTransportObjectiveConfig
    )
    wout_reference: Any | None = None
    name: str = "spectraxgk_transport"

    def J(self, ctx: Any, state: Any) -> jnp.ndarray:
        """Return the scalar transport objective for VMEC-JAX callbacks."""

        wout_ref = (
            self.wout_reference
            if self.wout_reference is not None
            else _reference_wout_from_context(ctx)
        )
        return vmec_jax_transport_objective_from_state(
            state,
            ctx.static,
            ctx.indata,
            wout_ref,
            self.config,
        )

    def to_objective_term(
        self, *, target: float | np.ndarray, residual_weight: float
    ) -> Any:
        """Return a VMEC-JAX ``ObjectiveTerm`` when VMEC-JAX is installed."""

        _pin_current_optional_backend_paths()
        objective_term = getattr(importlib.import_module("vmec_jax"), "ObjectiveTerm")

        def _prepare(ctx: Any) -> Any:
            _pin_current_optional_backend_paths()
            wout_ref = (
                self.wout_reference
                if self.wout_reference is not None
                else _reference_wout_from_context(ctx)
            )
            prewarm_vmec_boozer_equal_arc_cache(
                ctx.static,
                wout_ref,
                mboz=int(self.config.mboz),
                nboz=int(self.config.nboz),
            )
            return objective_term(
                self.name,
                self.J,
                target=target,
                weight=residual_weight,
                metadata={
                    "objective_family": "spectraxgk_transport",
                    "spectraxgk_transport_kind": self.config.kind,
                    "gradient_scope": self.config.gradient_scope,
                    "mboz": int(self.config.mboz),
                    "nboz": int(self.config.nboz),
                    "ntheta": int(self.config.ntheta),
                    "surface_chunk_size": int(self.config.surface_chunk_size),
                    "objective_transform": self.config.objective_transform,
                    "objective_scale": float(self.config.objective_scale),
                },
            )

        return objective_term(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            metadata={
                "objective_family": "spectraxgk_transport",
                "spectraxgk_transport_kind": self.config.kind,
                "gradient_scope": self.config.gradient_scope,
                "mboz": int(self.config.mboz),
                "nboz": int(self.config.nboz),
                "ntheta": int(self.config.ntheta),
                "surface_chunk_size": int(self.config.surface_chunk_size),
                "objective_transform": self.config.objective_transform,
                "objective_scale": float(self.config.objective_scale),
            },
            prepare=_prepare,
        )


def spectrax_transport_objective_tuple(
    *,
    weight: float,
    config: VMECJAXTransportObjectiveConfig | None = None,
    wout_reference: Any | None = None,
    target: float = 0.0,
) -> tuple[Any, float, float]:
    """Return a tuple that can be appended to ``LeastSquaresProblem.from_tuples``."""

    objective = VMECJAXSpectraxTransportObjective(
        config=config or VMECJAXTransportObjectiveConfig(),
        wout_reference=wout_reference,
    )
    return (objective.J, float(target), float(weight))


__all__ = [
    "VMECJAXSpectraxTransportObjective",
    "VMECJAXTransportObjectiveConfig",
    "VMECJAXTransportObjectiveKind",
    "VMECJAXTransportObjectiveTransform",
    "_apply_objective_transform",
    "_chunked_transport_objective_raw_value_from_state",
    "_geometry_transport_weights",
    "_module_search_root",
    "_pin_current_optional_backend_paths",
    "_reference_wout_from_context",
    "_solver_table_to_nonlinear_window_proxy",
    "_static_grid_options_from_ky_values",
    "_surface_chunk_sample_sets",
    "_transport_feature_table_from_state",
    "_transport_objective_raw_value_from_state",
    "os",
    "spectrax_transport_objective_tuple",
    "vmec_jax_transport_growth_branch_locality_report_from_states",
    "vmec_jax_transport_objective_from_state",
]
