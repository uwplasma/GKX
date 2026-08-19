"""One fixed windowed-adjoint case, reported exactly, for CPU/GPU comparison.

The release plan lists "CPU and GPU results agree within the selected
precision" as a gate, but the two profiles it rested on used different
platforms *and* different grids, so nothing was actually compared. This script
exists to make the comparison a like-for-like one: it fixes the case
completely -- grid, seed, dt, step count, integrator, term set, species -- so
the only thing that differs between two runs is the device (and, when the
environments differ, the jax version, which the output records).

Run it once per device and diff the JSON::

    python tools/profiling/profile_nonlinear_window_device_parity.py \\
        --output tools_out/window_parity_cpu.json
    # then, on the GPU host, same command with --output ..._gpu.json
    python tools/profiling/profile_nonlinear_window_device_parity.py \\
        --compare tools_out/window_parity_cpu.json tools_out/window_parity_gpu.json

``--compare`` refuses to report a difference when the two runs are not the same
case, which is the failure mode that produced the unsupported claim.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np

CASE = {
    "Nx": 16,
    "Ny": 16,
    "Nz": 8,
    "Lx": 6.0,
    "Ly": 6.0,
    "Nl": 2,
    "Nm": 4,
    "dt": 0.004,
    "steps": 8,
    "tail_steps": 5,
    "method": "rk3",
    "seed": 17,
    "amplitude": 1.0e-4,
    "beta": 0.02,
    "n_species": 2,
    "mass_ratio": 2.7e-4,
    "precision": "64",
}

#: Fields that must match for two runs to be comparable at all.
IDENTITY_KEYS = ("case", "state_checksum")


def evaluate(case: dict) -> dict:
    """Return the window value and its gradient for the fixed case."""

    import jax
    import jax.numpy as jnp
    import jaxlib

    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
    from gkx.operators.linear.params import Species, build_linear_params
    from gkx.solvers.nonlinear.state_integration import nonlinear_heat_flux_window
    from gkx.terms.config import TermConfig

    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=case["Nx"], Ny=case["Ny"], Nz=case["Nz"], Lx=case["Lx"], Ly=case["Ly"]
        )
    )
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    species = [
        Species(
            charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
        )
    ]
    if case["n_species"] == 2:
        species.append(
            Species(
                charge=-1.0,
                mass=case["mass_ratio"],
                density=1.0,
                temperature=1.0,
                tprim=2.49,
                fprim=0.8,
            )
        )
    params = build_linear_params(species, tau_e=1.0, beta=case["beta"], fapar=1.0)
    terms = TermConfig(nonlinear=1.0, apar=1.0, bpar=1.0)

    rng = np.random.default_rng(case["seed"])
    shape = (
        case["n_species"],
        case["Nl"],
        case["Nm"],
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    draw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    host_state = (case["amplitude"] * draw).astype(np.complex128)
    # The seed is drawn on the host, so the two devices differentiate the *same*
    # numbers rather than the same recipe for numbers.
    checksum = float(np.sum(np.abs(host_state) ** 2))
    state = jnp.asarray(host_state)

    base_drive = jnp.asarray(params.tprim)

    def objective(scale):
        return nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            replace(params, tprim=base_drive * scale),
            dt=case["dt"],
            steps=case["steps"],
            method=case["method"],
            tail_steps=case["tail_steps"],
            terms=terms,
            checkpoint=True,
            compressed_real_fft=True,
        )

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(1.0))
    return {
        "kind": "nonlinear_window_device_parity",
        "case": case,
        "state_checksum": checksum,
        "state_shape": list(shape),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "devices": [str(device) for device in jax.devices()],
        "default_backend": jax.default_backend(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "value": float(value),
        "gradient": float(gradient),
    }


def compare(left: Path, right: Path) -> int:
    """Report the relative gradient difference between two recorded runs."""

    a = json.loads(left.read_text())
    b = json.loads(right.read_text())
    for key in IDENTITY_KEYS:
        if a[key] != b[key]:
            raise SystemExit(
                f"refusing to compare: {key} differs between {left} and {right}. "
                "These are different cases, and their difference would not be a "
                "device-parity number."
            )
    report = {
        "kind": "nonlinear_window_device_parity_comparison",
        "left": {
            "path": str(left),
            "devices": a["devices"],
            "jax": a["jax"],
            "value": a["value"],
            "gradient": a["gradient"],
        },
        "right": {
            "path": str(right),
            "devices": b["devices"],
            "jax": b["jax"],
            "value": b["value"],
            "gradient": b["gradient"],
        },
        "value_relative_difference": abs(a["value"] - b["value"])
        / max(abs(b["value"]), 1.0e-300),
        "gradient_relative_difference": abs(a["gradient"] - b["gradient"])
        / max(abs(b["gradient"]), 1.0e-300),
        "same_jax": a["jax"] == b["jax"],
    }
    print(json.dumps(report, indent=2))
    if not report["same_jax"]:
        print(
            f"NOTE: jax {a['jax']} vs {b['jax']}; the difference below bounds "
            "device parity AND version drift together, not device parity alone.",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        default=None,
        metavar=("LEFT", "RIGHT"),
        help="compare two recorded runs instead of producing one",
    )
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    if args.compare is not None:
        return compare(*args.compare)

    import jax

    case = dict(CASE)
    if args.steps is not None:
        case["steps"] = int(args.steps)
        case["tail_steps"] = max(1, int(args.steps) - 3)
    jax.config.update("jax_enable_x64", case["precision"] == "64")

    summary = evaluate(case)
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
