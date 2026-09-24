"""ARCH-A behaviour fingerprints for source-contraction PRs.

Run from a checkout root with ``JAX_ENABLE_X64=true GKX_X64=1`` and
``PYTHONPATH=src``; prints JSON. Compare base and head bitwise (``float.hex``
and SHA-256 prefixes of the raw array bytes). On XLA:CPU all three are
deterministic run to run; on CUDA the linear eigenvalue and the window gradient
are, the nonlinear trace is not (compare its final value there).

1. Linear Cyclone eigenvalue: ``examples/linear/axisymmetric/cyclone.toml``.
2. Nonlinear heat-flux trace: the shipped short Cyclone deck with a stable
   100-step window (dt 0.005, t_max 0.5). The deck's own dt 0.05 is 5.7x its
   CFL bound and overflows.
3. Window gradient: d<Q>/d(tprim) through ``nonlinear_heat_flux_window`` on the
   small deck of ``test_block_checkpointed_nonlinear_heat_flux_gradient_matches_finite_difference``.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.operators.linear.params import LinearParams
from gkx.solvers_nonlinear_state_integration import nonlinear_heat_flux_window
from gkx.terms.config import TermConfig


def digest(array: object) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array)).tobytes()
    ).hexdigest()[:16]


def linear(root: Path) -> dict[str, object]:
    result = gkx.solve(gkx.load(root / "examples/linear/axisymmetric/cyclone.toml"))
    gamma = float(np.asarray(result.gamma).ravel()[0])
    omega = float(np.asarray(result.omega).ravel()[0])
    return {
        "gamma": gamma.hex(),
        "omega": omega.hex(),
        "eigenfunction_sha": digest(result.eigenfunction),
    }


def nonlinear(root: Path) -> dict[str, object]:
    deck = (
        root / "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml"
    ).read_text()
    deck = re.sub(r"(?m)^t_max = .*$", "t_max = 0.5", deck)
    deck = re.sub(r"(?m)^dt = .*$", "dt = 0.005", deck)
    path = Path(tempfile.mkdtemp()) / "fingerprint_nonlinear.toml"
    path.write_text(deck)
    trace = np.asarray(gkx.solve(gkx.load(path)).diagnostics.heat_flux_t)
    return {
        "heat_flux_sha": digest(trace),
        "n": int(trace.size),
        "last": float(trace.ravel()[-1]).hex(),
    }


def window_gradient() -> dict[str, object]:
    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, ky_layout="full")
    )
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base = LinearParams()
    dtype = jnp.complex128 if bool(jax.config.jax_enable_x64) else jnp.complex64
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = jnp.zeros((2, 2, 4, 4, 8), dtype=dtype)
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 0, :].set(0.25j * profile)
    state = state.at[0, 0, 1, 1, :].set((0.2 - 0.1j) * profile)

    def mean_heat_flux(rlt: jax.Array) -> jax.Array:
        return nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            replace(base, tprim=rlt),
            dt=0.01,
            steps=11,
            method="rk2",
            tail_steps=7,
            terms=TermConfig(nonlinear=1.0),
            compressed_real_fft=False,
        )

    value, gradient = jax.value_and_grad(mean_heat_flux)(jnp.asarray(6.9))
    return {"value": float(value).hex(), "grad": float(gradient).hex()}


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out: dict[str, object] = {"jax": jax.__version__, "device": str(jax.devices()[0])}
    for name, run in (
        ("linear", lambda: linear(root)),
        ("nonlinear", lambda: nonlinear(root)),
        ("window_gradient", window_gradient),
    ):
        start = time.perf_counter()
        out[name] = {**run(), "seconds": round(time.perf_counter() - start, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
