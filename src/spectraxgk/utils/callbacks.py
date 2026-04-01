"""Host-side progress callbacks used by JAX-integrated loops."""

from __future__ import annotations

from typing import Any

import jax


def _emit_progress(
    idx: Any,
    steps: Any,
    gamma: Any,
    omega: Any,
    wphi: Any,
    wg: Any,
) -> None:
    idx_i = int(idx)
    steps_i = max(int(steps), 1)
    gamma_f = float(gamma)
    omega_f = float(omega)
    wphi_f = float(wphi)
    wg_f = float(wg)
    pct = 100.0 * float(idx_i + 1) / float(steps_i)
    print(
        f"[spectrax-gk] step={idx_i + 1}/{steps_i} "
        f"progress={pct:5.1f}% gamma={gamma_f:.6g} omega={omega_f:.6g} "
        f"Wphi={wphi_f:.6g} Wg={wg_f:.6g}"
    )


def print_callback(
    state: Any,
    idx: Any,
    steps: Any,
    gamma: Any,
    omega: Any,
    wphi: Any,
    wg: Any,
) -> Any:
    """Emit a host-side progress update and return ``state`` unchanged."""

    jax.debug.callback(_emit_progress, idx, steps, gamma, omega, wphi, wg)
    return state
