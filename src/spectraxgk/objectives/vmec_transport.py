"""Public VMEC-JAX transport objectives and optimizer callback objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import prewarm_vmec_boozer_equal_arc_cache
from spectraxgk.objectives.vmec_transport_branch import (
    vmec_jax_transport_growth_branch_locality_report_from_states,
)
from spectraxgk.objectives.vmec_transport_config import (
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
    _pin_current_optional_backend_paths,
    _reference_wout_from_context,
)
from spectraxgk.objectives.vmec_transport_tables import (
    vmec_jax_transport_objective_from_state,
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
    "spectrax_transport_objective_tuple",
    "vmec_jax_transport_growth_branch_locality_report_from_states",
    "vmec_jax_transport_objective_from_state",
]
