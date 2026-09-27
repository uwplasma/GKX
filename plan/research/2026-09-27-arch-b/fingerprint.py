"""ARCH-B behaviour fingerprints for source-contraction PRs.

Run from a checkout root with ``JAX_ENABLE_X64=true`` and ``PYTHONPATH=src``;
prints JSON. Compare base and head bitwise (``float.hex`` and SHA-256 prefixes
of the raw array bytes). Every name is resolved through the public ``gkx``
facade, so the script runs unchanged on both sides of a module move.

1. Linear Cyclone eigenvalue: ``examples/01_linear_tokamak/case.toml``.
2. Nonlinear heat-flux trace: ``examples/03_nonlinear_tokamak/case.toml`` for
   100 fixed steps (dt 0.005, t_max 0.5).
3. Window gradient: d<Q>/d(tprim) through ``nonlinear_heat_flux_window`` on the
   small deck ARCH-A used (Nx = Ny = 4, Nz = 8, 11 rk2 steps, 7-step tail).
4. Quasilinear flux: ``examples/08_quasilinear/case.toml`` scanned at two k_y.
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
import gkx.geometry


def digest(array: object) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(array)).tobytes()
    ).hexdigest()[:16]


def linear(root: Path) -> dict[str, object]:
    result = gkx.solve(gkx.load(root / "examples/01_linear_tokamak/case.toml"))
    return {
        "gamma": float(np.asarray(result.gamma).ravel()[0]).hex(),
        "omega": float(np.asarray(result.omega).ravel()[0]).hex(),
        "gamma_sha": digest(np.asarray(result.gamma, dtype=float)),
        "omega_sha": digest(np.asarray(result.omega, dtype=float)),
        "eigenfunction_sha": digest(result.eigenfunction),
    }


def nonlinear(root: Path) -> dict[str, object]:
    deck = (root / "examples/03_nonlinear_tokamak/case.toml").read_text()
    deck = re.sub(r"(?m)^t_max = .*$", "t_max = 0.5", deck)
    deck = re.sub(r"(?m)^dt = .*$", "dt = 0.005", deck)
    path = Path(tempfile.mkdtemp()) / "case.toml"
    path.write_text(deck)
    trace = np.asarray(gkx.solve(gkx.load(path)).diagnostics.heat_flux_t)
    return {
        "heat_flux_sha": digest(trace),
        "n": int(trace.size),
        "last": float(trace.ravel()[-1]).hex(),
    }


def window_gradient() -> dict[str, object]:
    cfg = gkx.CycloneBaseCase(
        grid=gkx.GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, ky_layout="full")
    )
    grid = gkx.build_spectral_grid(cfg.grid)
    geom = gkx.geometry.ensure_flux_tube_geometry_data(
        gkx.SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base = gkx.LinearParams()
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = jnp.zeros((2, 2, 4, 4, 8), dtype=jnp.complex128)
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 0, :].set(0.25j * profile)
    state = state.at[0, 0, 1, 1, :].set((0.2 - 0.1j) * profile)

    def mean_heat_flux(rlt: jax.Array) -> jax.Array:
        return gkx.nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            replace(base, tprim=rlt),
            dt=0.01,
            steps=11,
            method="rk2",
            tail_steps=7,
            terms=gkx.TermConfig(nonlinear=1.0),
            compressed_real_fft=False,
        )

    value, gradient = jax.value_and_grad(mean_heat_flux)(jnp.asarray(6.9))
    return {"value": float(value).hex(), "grad": float(gradient).hex()}


def quasilinear(root: Path) -> dict[str, object]:
    case = gkx.load(root / "examples/08_quasilinear/case.toml")
    scan = gkx.scan(
        case, [0.2, 0.3], Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
    )
    flux = np.array(
        [p["saturated_heat_flux_total"] for p in scan.quasilinear], dtype=float
    )
    weight = np.array(
        [p["heat_flux_weight_total"] for p in scan.quasilinear], dtype=float
    )
    return {
        "gamma_sha": digest(np.asarray(scan.gamma, dtype=float)),
        "flux": [v.hex() for v in flux],
        "weight_sha": digest(weight),
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out: dict[str, object] = {"jax": jax.__version__, "device": str(jax.devices()[0])}
    for name, run in (
        ("linear", lambda: linear(root)),
        ("nonlinear", lambda: nonlinear(root)),
        ("window_gradient", window_gradient),
        ("quasilinear", lambda: quasilinear(root)),
    ):
        start = time.perf_counter()
        out[name] = {**run(), "seconds": round(time.perf_counter() - start, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
