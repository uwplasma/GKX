"""Evaluate SPECTRAX-GK transport objectives from VMEC-JAX states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from spectraxgk.objectives.vmec_transport_branch import (
    vmec_jax_transport_growth_branch_locality_report_from_states,
)
from spectraxgk.objectives.vmec_transport_config import (
    VMECJAXTransportObjectiveConfig,
    VMECJAXTransportObjectiveKind,
    VMECJAXTransportObjectiveTransform,
    _reference_wout_from_context,
)
from spectraxgk.objectives.vmec_transport_tables import (
    vmec_jax_transport_objective_from_state,
)


@dataclass(frozen=True)
class VMECJAXSpectraxTransportObjective:
    """Evaluate a configured transport metric from a solved VMEC-JAX state."""

    config: VMECJAXTransportObjectiveConfig = field(
        default_factory=VMECJAXTransportObjectiveConfig
    )
    wout_reference: Any | None = None

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

__all__ = [
    "VMECJAXSpectraxTransportObjective",
    "VMECJAXTransportObjectiveConfig",
    "VMECJAXTransportObjectiveKind",
    "VMECJAXTransportObjectiveTransform",
    "vmec_jax_transport_growth_branch_locality_report_from_states",
    "vmec_jax_transport_objective_from_state",
]
