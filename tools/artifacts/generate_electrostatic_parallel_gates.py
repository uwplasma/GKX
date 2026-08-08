#!/usr/bin/env python3
"""Generate electrostatic velocity-sharded identity-gate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIELD_PREFIX = REPO_ROOT / "docs" / "_static" / "electrostatic_field_reduce_gate"
DEFAULT_DRIFT_PREFIX = REPO_ROOT / "docs" / "_static" / "electrostatic_drift_gate"
DEFAULT_DIAMAGNETIC_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "electrostatic_diamagnetic_gate"
)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _configure_logical_cpu_devices(count: int) -> None:
    if int(count) <= 1:
        return
    flag = f"--xla_force_host_platform_device_count={int(count)}"
    current = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


def _block_until_ready(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _terms(
    *,
    mirror: float = 0.0,
    curvature: float = 0.0,
    gradb: float = 0.0,
    diamagnetic: float = 0.0,
) -> Any:
    from gkx.operators.linear.params import LinearTerms

    return LinearTerms(
        streaming=0.0,
        mirror=mirror,
        curvature=curvature,
        gradb=gradb,
        diamagnetic=diamagnetic,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def _field_only_terms() -> Any:
    return _terms()


def _diamagnetic_terms() -> Any:
    return _terms(diamagnetic=1.0)


def build_problem(
    *,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    include_drift_moments: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Build a periodic electrostatic identity-gate problem."""

    import jax.numpy as jnp

    from gkx.config import CycloneBaseCase, GridConfig
    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import SAlphaGeometry
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.operators.linear.params import LinearParams

    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=int(nx), Ny=int(ny), Nz=int(nz), Lx=6.0, Ly=6.0, boundary="periodic"
        )
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, int(nl), int(nm))
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros(
        (int(nl), int(nm), grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex64,
    )
    ky_index = min(1, grid.ky.size - 1)
    state = state.at[0, 0, ky_index, 0, :].set(0.2 * jnp.exp(1j * z))
    if int(nl) > 1:
        state = state.at[1, 0, ky_index, 0, :].set(0.08 * jnp.exp(2j * z))
    if include_drift_moments and int(nm) > 2:
        state = state.at[0, 2, ky_index, 0, :].set(0.06 * jnp.exp(2j * z))
        if int(nl) > 1 and int(nm) > 3:
            state = state.at[1, 3, ky_index, 0, :].set(0.04 * jnp.exp(3j * z))
    return state, cache, params, grid


def _device_list(requested_devices: int) -> list[Any]:
    import jax

    devices = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(devices) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(devices)} are available"
        )
    return devices


def _velocity_plan(state: Any, device_list: list[Any]) -> Any:
    from gkx.parallel.velocity import build_velocity_sharding_plan

    return build_velocity_sharding_plan(
        state.shape, num_devices=len(device_list), axes=("hermite",)
    )


def _electrostatic_phi(state: Any, cache: Any, params: Any, plan: Any, devices: list[Any]):
    from gkx.parallel.velocity import electrostatic_phi_shard_map

    return electrostatic_phi_shard_map(
        state,
        plan,
        Jl=cache.Jl,
        tau_e=params.tau_e,
        charge=params.charge_sign,
        density=params.density,
        tz=params.tz,
        mask0=cache.mask0,
        devices=devices,
    )


def build_electrostatic_field_reduce_gate(
    *,
    requested_devices: int,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare production ``phi`` with the Hermite-sharded density reduction."""

    import jax.numpy as jnp

    from gkx.operators.linear.rhs import linear_rhs_cached

    devices = _device_list(requested_devices)
    state, cache, params, grid = build_problem(nx=nx, ny=ny, nz=nz, nl=nl, nm=nm)
    _rhs, phi_serial = linear_rhs_cached(
        state,
        cache,
        params,
        terms=_field_only_terms(),
        use_jit=False,
        use_custom_vjp=False,
    )
    plan = _velocity_plan(state, devices)
    phi_sharded = _electrostatic_phi(state, cache, params, plan, devices)
    _block_until_ready((phi_serial, phi_sharded))

    abs_err = jnp.max(jnp.abs(phi_sharded - phi_serial))
    scale = jnp.max(jnp.abs(phi_serial))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    phi_norm = jnp.linalg.norm(phi_serial)
    _block_until_ready((abs_err, rel_err, phi_norm))
    max_abs_error = float(np.asarray(abs_err))
    max_rel_error = float(np.asarray(rel_err))
    phi_norm_float = float(np.asarray(phi_norm))
    identity_passed = bool(
        max_abs_error <= float(atol)
        and max_rel_error <= float(rtol)
        and phi_norm_float > 0.0
    )

    ky_index = min(1, grid.ky.size - 1)
    serial_trace = np.asarray(phi_serial[ky_index, 0, :])
    sharded_trace = np.asarray(phi_sharded[ky_index, 0, :])
    rows = [
        {
            "z_index": int(z_idx),
            "serial_abs": float(abs(serial_trace[z_idx])),
            "sharded_abs": float(abs(sharded_trace[z_idx])),
            "abs_error": float(abs(sharded_trace[z_idx] - serial_trace[z_idx])),
        }
        for z_idx in range(grid.z.size)
    ]

    return _json_clean(
        {
            "case": "Electrostatic Hermite-sharded field-reduction identity gate",
            "source": "gkx.parallel.velocity.electrostatic_phi_shard_map",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached production electrostatic field solve",
            "claim_scope": "single-species electrostatic phi field-reduction identity, not a full RHS or nonlinear speedup claim",
            "state_shape": tuple(int(x) for x in state.shape),
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Ny_actual": int(grid.ky.size),
                "Nz": int(grid.z.size),
            },
            "requested_devices": int(requested_devices),
            "actual_devices": len(devices),
            "plan": plan.to_dict(),
            "phi_norm": phi_norm_float,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "The sharded path selects the global m=0 density moment on the Hermite mesh, "
                "reduces it with lax.psum, and applies the electrostatic quasineutrality denominator."
            ),
        }
    )


def build_electrostatic_diamagnetic_gate(
    *,
    requested_devices: int,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare the sharded electrostatic diamagnetic drive with production."""

    import jax.numpy as jnp

    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.parallel.velocity import diamagnetic_drive_shard_map

    devices = _device_list(requested_devices)
    state, cache, params, grid = build_problem(nx=nx, ny=ny, nz=nz, nl=nl, nm=nm)
    plan = _velocity_plan(state, devices)
    serial, phi_serial = linear_rhs_cached(
        state,
        cache,
        params,
        terms=_diamagnetic_terms(),
        use_jit=False,
        use_custom_vjp=False,
    )
    phi_sharded = _electrostatic_phi(state, cache, params, plan, devices)
    sharded = diamagnetic_drive_shard_map(
        state,
        plan,
        phi=phi_sharded,
        Jl=cache.Jl,
        b=cache.b,
        l4=cache.l4,
        tprim=params.tprim,
        fprim=params.fprim,
        omega_star_scale=params.omega_star_scale,
        ky=cache.ky,
        devices=devices,
    )
    _block_until_ready((serial, phi_serial, phi_sharded, sharded))

    abs_err = jnp.linalg.norm(sharded - serial)
    rel_err = abs_err / jnp.maximum(
        jnp.linalg.norm(serial), jnp.asarray(1.0e-30, dtype=jnp.real(serial).dtype)
    )
    phi_abs_err = jnp.max(jnp.abs(phi_sharded - phi_serial))
    phi_norm = jnp.linalg.norm(phi_serial)
    _block_until_ready((abs_err, rel_err, phi_abs_err, phi_norm))
    max_abs_error = float(np.asarray(abs_err))
    max_rel_error = float(np.asarray(rel_err))
    max_phi_abs_error = float(np.asarray(phi_abs_err))
    identity_passed = bool(
        max_abs_error <= float(atol)
        and max_rel_error <= float(rtol)
        and max_phi_abs_error <= float(atol)
    )

    rows = []
    for m_idx in range(int(nm)):
        serial_norm = float(np.asarray(jnp.linalg.norm(serial[:, m_idx, ...])))
        sharded_norm = float(np.asarray(jnp.linalg.norm(sharded[:, m_idx, ...])))
        rows.append(
            {
                "m": int(m_idx),
                "serial_norm": serial_norm,
                "sharded_norm": sharded_norm,
                "abs_error": float(
                    np.asarray(
                        jnp.linalg.norm(sharded[:, m_idx, ...] - serial[:, m_idx, ...])
                    )
                ),
            }
        )

    return _json_clean(
        {
            "case": "Electrostatic Hermite-sharded diamagnetic-drive identity gate",
            "source": "gkx.parallel.velocity.diamagnetic_drive_shard_map",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached with diamagnetic term only",
            "claim_scope": "single-species periodic electrostatic diamagnetic-drive identity, not a full RHS or nonlinear speedup claim",
            "state_shape": tuple(int(x) for x in state.shape),
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Ny_actual": int(grid.ky.size),
                "Nz": int(grid.z.size),
            },
            "requested_devices": int(requested_devices),
            "actual_devices": len(devices),
            "plan": plan.to_dict(),
            "phi_norm": float(np.asarray(phi_norm)),
            "max_phi_abs_error": max_phi_abs_error,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "The sharded path uses the Hermite-sharded electrostatic field reduction, "
                "then applies the local m=0 and m=2 diamagnetic drive masks on each Hermite shard."
            ),
        }
    )


def _relative_error(abs_error: Any, reference: Any) -> Any:
    import jax.numpy as jnp

    scale = jnp.maximum(
        jnp.linalg.norm(reference),
        jnp.asarray(1.0e-30, dtype=jnp.real(reference).dtype),
    )
    return abs_error / scale


def build_electrostatic_drift_gate(
    *,
    requested_devices: int,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare sharded mirror/curvature/grad-B drift slices with production."""

    import jax.numpy as jnp

    from gkx.operators.linear.moments import build_H
    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.parallel.velocity import (
        curvature_gradb_drift_shard_map,
        mirror_drift_shard_map,
    )

    devices = _device_list(requested_devices)
    state, cache, params, grid = build_problem(
        nx=nx, ny=ny, nz=nz, nl=nl, nm=nm, include_drift_moments=True
    )
    plan = _velocity_plan(state, devices)
    phi = _electrostatic_phi(state, cache, params, plan, devices)
    H = build_H(state, cache.Jl, phi, jnp.asarray([params.tz], dtype=jnp.float32))
    mirror_sharded = mirror_drift_shard_map(
        H,
        plan,
        vth=jnp.asarray([params.vth], dtype=jnp.float32),
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        devices=devices,
    )
    curvgb_sharded = curvature_gradb_drift_shard_map(
        H,
        plan,
        tz=jnp.asarray([params.tz], dtype=jnp.float32),
        omega_d_scale=params.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        devices=devices,
    )
    total_sharded = mirror_sharded + curvgb_sharded
    mirror_serial, _ = linear_rhs_cached(
        state,
        cache,
        params,
        terms=_terms(mirror=1.0),
        use_jit=False,
        use_custom_vjp=False,
    )
    curvgb_serial, _ = linear_rhs_cached(
        state,
        cache,
        params,
        terms=_terms(curvature=1.0, gradb=1.0),
        use_jit=False,
        use_custom_vjp=False,
    )
    total_serial, phi_serial = linear_rhs_cached(
        state,
        cache,
        params,
        terms=_terms(mirror=1.0, curvature=1.0, gradb=1.0),
        use_jit=False,
        use_custom_vjp=False,
    )
    _block_until_ready(
        (
            mirror_sharded,
            curvgb_sharded,
            total_sharded,
            mirror_serial,
            curvgb_serial,
            total_serial,
            phi_serial,
        )
    )

    components = (
        ("mirror", mirror_serial, mirror_sharded),
        ("curvature_gradb", curvgb_serial, curvgb_sharded),
        ("total", total_serial, total_sharded),
    )
    rows = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    for name, serial, sharded in components:
        abs_error = jnp.linalg.norm(sharded - serial)
        rel_error = _relative_error(abs_error, serial)
        _block_until_ready((abs_error, rel_error))
        abs_float = float(np.asarray(abs_error))
        rel_float = float(np.asarray(rel_error))
        max_abs_error = max(max_abs_error, abs_float)
        max_rel_error = max(max_rel_error, rel_float)
        rows.append(
            {
                "component": name,
                "serial_norm": float(np.asarray(jnp.linalg.norm(serial))),
                "sharded_norm": float(np.asarray(jnp.linalg.norm(sharded))),
                "abs_error": abs_float,
                "rel_error": rel_float,
            }
        )

    identity_passed = bool(
        max_abs_error <= float(atol) and max_rel_error <= float(rtol)
    )
    return _json_clean(
        {
            "case": "Electrostatic Hermite-sharded mirror/curvature/grad-B drift identity gate",
            "source": "gkx.parallel.velocity mirror_drift_shard_map + curvature_gradb_drift_shard_map",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached with mirror/curvature/grad-B terms only",
            "claim_scope": "single-species periodic electrostatic drift-slice identity, not a full RHS or nonlinear speedup claim",
            "state_shape": tuple(int(x) for x in state.shape),
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Ny_actual": int(grid.ky.size),
                "Nz": int(grid.z.size),
            },
            "requested_devices": int(requested_devices),
            "actual_devices": len(devices),
            "plan": plan.to_dict(),
            "phi_norm": float(np.asarray(jnp.linalg.norm(phi_serial))),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "The sharded path uses the Hermite-sharded electrostatic field reduction, "
                "then applies mirror offset-1 and curvature offset-2 Hermite exchanges."
            ),
        }
    )


def _write_common(summary: dict[str, object], out_prefix: Path) -> dict[str, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_prefix.with_suffix(".json"),
        "csv": out_prefix.with_suffix(".csv"),
        "png": out_prefix.with_suffix(".png"),
        "pdf": out_prefix.with_suffix(".pdf"),
    }
    paths["json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return paths


def _plot_field(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    z = np.asarray([row["z_index"] for row in rows], dtype=float)
    serial = np.asarray([row["serial_abs"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_abs"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(z, serial, "s-", lw=1.8, label="serial phi")
    axes[0].plot(z, sharded, "^--", lw=1.8, label="sharded phi")
    axes[0].set_xlabel("z index")
    axes[0].set_ylabel("|phi|")
    axes[0].set_title("Electrostatic field reduction")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        z, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_xlabel("z index")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def _plot_moment(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    m = np.asarray([row["m"] for row in rows], dtype=float)
    serial = np.asarray([row["serial_norm"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_norm"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, serial, "s-", lw=1.8, label="serial")
    axes[0].plot(m, sharded, "^--", lw=1.8, label="sharded")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("RHS norm")
    axes[0].set_title("Electrostatic diamagnetic drive")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        m, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
    axes[1].set_xlabel("Hermite index m")
    axes[1].set_ylabel("absolute error")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def _plot_components(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    labels = [str(row["component"]).replace("_", "\n") for row in rows]
    serial = np.asarray([row["serial_norm"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_norm"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)
    x = np.arange(len(labels))

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, serial, width, label="serial")
    axes[0].bar(x + width / 2, sharded, width, label="sharded")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("RHS norm")
    axes[0].set_title("Electrostatic drift slices")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        x, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("absolute error")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for any electrostatic parallel gate."""

    paths = _write_common(summary, out_prefix)
    rows = list(summary["rows"])
    if rows and "z_index" in rows[0]:
        _plot_field(summary, paths)
    elif rows and "component" in rows[0]:
        _plot_components(summary, paths)
    elif rows and "m" in rows[0]:
        _plot_moment(summary, paths)
    else:
        raise ValueError("Unrecognized electrostatic gate row schema")
    return {name: str(path) for name, path in paths.items()}


def _add_common_args(parser: argparse.ArgumentParser, *, out_prefix: Path) -> None:
    parser.add_argument("--out-prefix", type=Path, default=out_prefix)
    parser.add_argument("--logical-devices", type=int, default=2)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=2.0e-6)


def _run_field(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_electrostatic_field_reduce_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=2.0e-6 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_drift(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_electrostatic_drift_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=2.0e-5 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_diamagnetic(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_electrostatic_diamagnetic_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=2.0e-5 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    field = subparsers.add_parser(
        "field-reduce", help="Gate the electrostatic quasineutrality reduction."
    )
    _add_common_args(field, out_prefix=DEFAULT_FIELD_PREFIX)
    field.set_defaults(func=_run_field)

    drift = subparsers.add_parser(
        "drift", help="Gate mirror/curvature/grad-B electrostatic slices."
    )
    _add_common_args(drift, out_prefix=DEFAULT_DRIFT_PREFIX)
    drift.set_defaults(func=_run_drift)

    diamagnetic = subparsers.add_parser(
        "diamagnetic", help="Gate the electrostatic diamagnetic-drive slice."
    )
    _add_common_args(diamagnetic, out_prefix=DEFAULT_DIAMAGNETIC_PREFIX)
    diamagnetic.set_defaults(func=_run_diamagnetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
