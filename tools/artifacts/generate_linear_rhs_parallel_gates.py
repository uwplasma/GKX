#!/usr/bin/env python3
"""Generate linear-RHS velocity-sharded identity-gate artifacts."""

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
DEFAULT_STREAMING_PREFIX = REPO_ROOT / "docs" / "_static" / "linear_rhs_streaming_gate"
DEFAULT_ELECTROSTATIC_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "linear_rhs_streaming_electrostatic_gate"
)
DEFAULT_SLICES_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "linear_rhs_electrostatic_slices_gate"
)
DEFAULT_ZERO_NORM_OUT_JSON = (
    REPO_ROOT / "docs" / "_static" / "linear_rhs_zero_norm_state_window_gate.json"
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


def _linear_terms(
    *,
    streaming: float = 0.0,
    mirror: float = 0.0,
    curvature: float = 0.0,
    gradb: float = 0.0,
    diamagnetic: float = 0.0,
) -> Any:
    from gkx.operators.linear.params import LinearTerms

    return LinearTerms(
        streaming=streaming,
        mirror=mirror,
        curvature=curvature,
        gradb=gradb,
        diamagnetic=diamagnetic,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def _streaming_only_terms() -> Any:
    return _linear_terms(streaming=1.0)


def _electrostatic_slice_terms() -> Any:
    return _linear_terms(
        streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0
    )


def build_problem(
    *,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    mode: str = "electrostatic_slices",
    return_geom: bool = False,
) -> tuple[Any, ...]:
    """Build a small periodic linear-RHS identity-gate problem."""

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

    if mode == "streaming_only":
        state = state.at[0, min(2, int(nm) - 1), ky_index, 0, :].set(jnp.exp(1j * z))
        if int(nm) > 4 and grid.ky.size > 2:
            state = state.at[1 % int(nl), 4, min(2, grid.ky.size - 1), 0, :].set(
                0.25 * jnp.exp(2j * z)
            )
    elif mode == "streaming_electrostatic":
        state = state.at[0, 0, ky_index, 0, :].set(0.2 * jnp.exp(1j * z))
        if int(nm) > 3 and int(nl) > 1:
            state = state.at[1, 2, ky_index, 0, :].set(0.1 * jnp.exp(2j * z))
    elif mode == "electrostatic_slices":
        state = state.at[0, 0, ky_index, 0, :].set(0.20 * jnp.exp(1j * z))
        if int(nl) > 1:
            state = state.at[1, 0, ky_index, 0, :].set(0.08 * jnp.exp(2j * z))
        if int(nm) > 2:
            state = state.at[0, 2, ky_index, 0, :].set(0.07 * jnp.exp(2j * z))
        if int(nm) > 3 and int(nl) > 1:
            state = state.at[1, 3, ky_index, 0, :].set(0.03 * jnp.exp(3j * z))
    else:
        raise ValueError(f"unknown linear-RHS gate problem mode: {mode}")

    if return_geom:
        return state, cache, params, grid, geom
    return state, cache, params, grid


def _device_list(requested_devices: int) -> list[Any]:
    import jax

    devices = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(devices) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(devices)} are available"
        )
    return devices


def _velocity_plan(state: Any, devices: list[Any]) -> Any:
    from gkx.parallel.velocity import build_velocity_sharding_plan

    return build_velocity_sharding_plan(
        state.shape, num_devices=len(devices), axes=("hermite",)
    )


def build_linear_rhs_streaming_gate(
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
    """Compare streaming-only ``linear_rhs_cached`` with the shard-map kernel."""

    import jax.numpy as jnp

    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.parallel.velocity import periodic_streaming_shard_map

    devices = _device_list(requested_devices)
    state, cache, params, grid, _geom = build_problem(
        nx=nx, ny=ny, nz=nz, nl=nl, nm=nm, mode="streaming_only", return_geom=True
    )
    terms = _streaming_only_terms()
    production, phi = linear_rhs_cached(state, cache, params, terms=terms, use_jit=True)
    plan = _velocity_plan(state, devices)
    # linked_streaming_contribution applies a leading minus sign before d/dz.
    sharded = -periodic_streaming_shard_map(
        state,
        plan,
        kz=cache.kz,
        vth=jnp.asarray([params.vth], dtype=jnp.float32),
        devices=devices,
    )
    _block_until_ready((production, phi, sharded))

    abs_err = jnp.max(jnp.abs(sharded - production))
    scale = jnp.max(jnp.abs(production))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    phi_norm = jnp.linalg.norm(phi)
    _block_until_ready((abs_err, rel_err, phi_norm))
    max_abs_error = float(np.asarray(abs_err))
    max_rel_error = float(np.asarray(rel_err))
    phi_norm_float = float(np.asarray(phi_norm))
    identity_passed = bool(
        max_abs_error <= float(atol) and max_rel_error <= float(rtol)
    )

    production_trace = np.asarray(production[0, :, min(1, grid.ky.size - 1), 0, 1])
    sharded_trace = np.asarray(sharded[0, :, min(1, grid.ky.size - 1), 0, 1])
    rows = [
        {
            "m": int(m_idx),
            "production_abs": float(abs(production_trace[m_idx])),
            "sharded_abs": float(abs(sharded_trace[m_idx])),
            "abs_error": float(abs(sharded_trace[m_idx] - production_trace[m_idx])),
        }
        for m_idx in range(int(nm))
    ]

    return _json_clean(
        {
            "case": "Full linear-RHS streaming-only shard-map identity gate",
            "source": "gkx.parallel.velocity.periodic_streaming_shard_map",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached with only streaming enabled",
            "claim_scope": "linear RHS identity gate with only streaming enabled, not a full-RHS or nonlinear speedup claim",
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
                "All non-streaming terms and electromagnetic channels are disabled. "
                "The test state uses non-density Hermite moments so the electrostatic field solve is zero."
            ),
        }
    )


def build_linear_rhs_streaming_electrostatic_gate(
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
    """Compare serial streaming+phi RHS with the explicit velocity route."""

    import jax.numpy as jnp

    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.parallel.velocity import build_velocity_sharding_plan
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    devices = _device_list(requested_devices)
    state, cache, params, grid = build_problem(
        nx=nx, ny=ny, nz=nz, nl=nl, nm=nm, mode="streaming_electrostatic"
    )
    terms = _streaming_only_terms()
    serial, phi_serial = linear_rhs_cached(
        state, cache, params, terms=terms, use_jit=True
    )
    parallel_cfg = RuntimeParallelConfig(
        strategy="velocity",
        backend="streaming_electrostatic",
        axis="hermite",
        num_devices=len(devices),
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel_cfg,
        use_custom_vjp=True,
    )
    _block_until_ready((serial, phi_serial, sharded, phi_sharded))

    abs_err = jnp.max(jnp.abs(sharded - serial))
    scale = jnp.max(jnp.abs(serial))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    phi_abs_err = jnp.max(jnp.abs(phi_sharded - phi_serial))
    phi_norm = jnp.linalg.norm(phi_serial)
    _block_until_ready((abs_err, rel_err, phi_abs_err, phi_norm))
    max_abs_error = float(np.asarray(abs_err))
    max_rel_error = float(np.asarray(rel_err))
    max_phi_abs_error = float(np.asarray(phi_abs_err))
    phi_norm_float = float(np.asarray(phi_norm))
    identity_passed = bool(
        max_abs_error <= float(atol)
        and max_rel_error <= float(rtol)
        and max_phi_abs_error <= float(atol)
    )

    ky_index = min(1, grid.ky.size - 1)
    serial_trace = np.asarray(serial[0, :, ky_index, 0, 1])
    sharded_trace = np.asarray(sharded[0, :, ky_index, 0, 1])
    rows = [
        {
            "m": int(m_idx),
            "serial_abs": float(abs(serial_trace[m_idx])),
            "sharded_abs": float(abs(sharded_trace[m_idx])),
            "abs_error": float(abs(sharded_trace[m_idx] - serial_trace[m_idx])),
        }
        for m_idx in range(int(nm))
    ]

    plan = build_velocity_sharding_plan(
        state.shape, num_devices=len(devices), axes=("hermite",)
    )
    return _json_clean(
        {
            "case": "Electrostatic streaming linear-RHS shard-map identity gate",
            "source": "gkx.solvers.linear.parallel.linear_rhs_parallel_cached backend=streaming_electrostatic",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached with only streaming enabled",
            "claim_scope": "streaming plus electrostatic field-solve call-graph identity, not a full-RHS or nonlinear speedup claim",
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
            "max_phi_abs_error": max_phi_abs_error,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "Only streaming is enabled. The state includes an m=0 density perturbation, so phi is nonzero. "
                "The electrostatic field solve uses the Hermite-sharded field-reduction path gated separately by "
                "tools/artifacts/generate_electrostatic_parallel_gates.py field-reduce."
            ),
        }
    )


def build_linear_rhs_electrostatic_slices_gate(
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
    """Compare serial production RHS with the composed velocity-sharded route."""

    import jax.numpy as jnp

    from gkx.operators.linear.rhs import linear_rhs_cached
    from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
    from gkx.parallel.velocity import build_velocity_sharding_plan
    from gkx.workflows.runtime.config import RuntimeParallelConfig

    devices = _device_list(requested_devices)
    state, cache, params, grid = build_problem(
        nx=nx, ny=ny, nz=nz, nl=nl, nm=nm, mode="electrostatic_slices"
    )
    terms = _electrostatic_slice_terms()
    serial, phi_serial = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
    )
    parallel_cfg = RuntimeParallelConfig(
        strategy="velocity",
        backend="electrostatic_linear_slices",
        axis="hermite",
        num_devices=len(devices),
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel_cfg,
        use_custom_vjp=False,
    )
    _block_until_ready((serial, phi_serial, sharded, phi_sharded))

    abs_err = jnp.max(jnp.abs(sharded - serial))
    scale = jnp.max(jnp.abs(serial))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
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
                        jnp.max(jnp.abs(sharded[:, m_idx, ...] - serial[:, m_idx, ...]))
                    )
                ),
            }
        )

    plan = build_velocity_sharding_plan(
        state.shape, num_devices=len(devices), axes=("hermite",)
    )
    return _json_clean(
        {
            "case": "Composed electrostatic linear-slices RHS identity gate",
            "source": "gkx.solvers.linear.parallel.linear_rhs_parallel_cached backend=electrostatic_linear_slices",
            "reference_source": "gkx.operators.linear.rhs.linear_rhs_cached with streaming/mirror/curvature/gradB/diamagnetic enabled",
            "claim_scope": (
                "single-species periodic electrostatic linear-RHS identity for the gated slices; "
                "not a linked-boundary, collision, electromagnetic, nonlinear, or speedup claim"
            ),
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
                "This is the composed call-graph gate for streaming, mirror, curvature, grad-B, "
                "and diamagnetic-drive slices after each primitive has passed its own identity gate."
            ),
        }
    )


def _block_tree(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        jax.block_until_ready(leaf)


def _float_norm(arr: Any) -> float:
    import jax.numpy as jnp

    return float(np.asarray(jnp.linalg.norm(arr)))


def _relative_error(reference: Any, candidate: Any) -> float:
    ref_norm = _float_norm(reference)
    diff_norm = _float_norm(reference - candidate)
    return diff_norm / max(ref_norm, 1.0e-30)


def _z_variation_norm(state: Any) -> float:
    import jax.numpy as jnp

    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return _float_norm(state - mean_z)


def _inject_z_wave(
    state: Any,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> Any:
    """Inject a deterministic resolved parallel wave into one Hermite mode."""

    import jax.numpy as jnp

    state = jnp.asarray(state)
    Nz = state.shape[-1]
    Nm = state.shape[-4] if state.ndim == 6 else state.shape[-4]
    m_index = min(max(1, Nm - 1), 3)
    z = jnp.arange(Nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(Nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - upstream assembly validates this before normal use.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _term_skip_config(term_cfg: Any, term: str) -> Any:
    from dataclasses import replace

    if term == "collisions":
        return replace(term_cfg, collisions=0.0)
    if term == "hypercollisions":
        return replace(term_cfg, hypercollisions=0.0)
    raise ValueError(f"unsupported zero-norm skip candidate: {term}")


def _evaluate_state(
    name: str,
    state: Any,
    *,
    cache: Any,
    params: Any,
    term_cfg: Any,
    candidates: tuple[str, ...] = ("collisions", "hypercollisions"),
) -> dict[str, Any]:
    from gkx.terms.assembly import assemble_rhs_cached, assemble_rhs_terms_cached

    rhs, _fields, contrib = assemble_rhs_terms_cached(
        state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
    )
    full_rhs, _ = assemble_rhs_cached(
        state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
    )
    _block_tree((rhs, full_rhs, contrib))
    term_norms = {key: _float_norm(value) for key, value in contrib.items()}
    skip_errors: dict[str, float] = {}
    for candidate in candidates:
        skipped_rhs, _ = assemble_rhs_cached(
            state,
            cache,
            params,
            terms=_term_skip_config(term_cfg, candidate),
            use_custom_vjp=False,
        )
        _block_tree(skipped_rhs)
        skip_errors[candidate] = _relative_error(full_rhs, skipped_rhs)
    return {
        "state": name,
        "state_norm": _float_norm(state),
        "z_variation_norm": _z_variation_norm(state),
        "term_norms": term_norms,
        "relative_skip_errors": skip_errors,
    }


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    config: str,
    ky: float,
    kx: float | None,
    nl: int,
    nm: int,
    identity_threshold: float,
    activation_threshold: float,
) -> dict[str, Any]:
    candidate_names = ("collisions", "hypercollisions")
    candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidate_names:
        skip_errors = [float(row["relative_skip_errors"][candidate]) for row in rows]
        term_norms = [float(row["term_norms"][candidate]) for row in rows]
        initial_skip_error = skip_errors[0] if skip_errors else math.nan
        max_skip_error = max(skip_errors, default=math.nan)
        max_term_norm = max(term_norms, default=math.nan)
        active_states = [
            str(row["state"])
            for row in rows
            if float(row["relative_skip_errors"][candidate]) > activation_threshold
            or float(row["term_norms"][candidate]) > activation_threshold
        ]
        candidates[candidate] = {
            "initial_relative_skip_error": initial_skip_error,
            "max_relative_skip_error": max_skip_error,
            "max_term_norm": max_term_norm,
            "active_states": active_states,
            "safe_to_disable_over_window": bool(max_skip_error <= identity_threshold),
        }

    collisions_identity_pass = bool(
        candidates["collisions"]["safe_to_disable_over_window"]
    )
    hyper_initial_zero = bool(
        candidates["hypercollisions"]["initial_relative_skip_error"]
        <= identity_threshold
    )
    hyper_rejected = bool(
        candidates["hypercollisions"]["max_relative_skip_error"] > activation_threshold
    )
    import jax

    return {
        "kind": "linear_rhs_zero_norm_state_window_gate",
        "case": Path(config).stem,
        "config": config,
        "backend": jax.default_backend(),
        "ky": float(ky),
        "kx": None if kx is None else float(kx),
        "Nl": int(nl),
        "Nm": int(nm),
        "identity_threshold": float(identity_threshold),
        "activation_threshold": float(activation_threshold),
        "rows": rows,
        "candidates": candidates,
        "passed": collisions_identity_pass and hyper_initial_zero and hyper_rejected,
        "policy": (
            "A term may only be disabled by a fast path when its relative skip error stays below "
            "the identity threshold over the whole sampled state window. A profiler-zero initial "
            "state is insufficient evidence for nonlinear states with resolved parallel variation."
        ),
        "recommendations": {
            "collisions": (
                "This Cyclone window supports a zero-collision fast path only when all collision "
                "frequencies are zero and the same identity gate passes."
            ),
            "hypercollisions": (
                "Do not skip kz hypercollisions based on the initial-state zero norm; the z-varying "
                "state activates the contribution."
            ),
        },
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )



def _add_zero_norm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--z-mode", type=int, default=1)
    parser.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    parser.add_argument("--kick-dt", type=float, default=1.0e-3)
    parser.add_argument("--identity-threshold", type=float, default=1.0e-10)
    parser.add_argument("--activation-threshold", type=float, default=1.0e-8)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_ZERO_NORM_OUT_JSON)


def _run_zero_norm_state_window(args: argparse.Namespace) -> int:
    import jax.numpy as jnp

    from gkx.core.grid import build_spectral_grid
    from gkx.geometry import apply_imported_geometry_grid_defaults
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.runtime import (
        _build_initial_condition,
        _select_nonlinear_mode_indices,
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_term_config,
    )
    from gkx.terms.assembly import assemble_rhs_cached
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=args.Nm, geom=geom)
    term_cfg = build_runtime_term_config(cfg)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    state0 = jnp.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=args.Nl,
            Nm=args.Nm,
            nspecies=len(cfg.species),
        )
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    z_wave = _inject_z_wave(
        state0,
        ky_index=int(ky_index),
        kx_index=int(kx_index),
        amplitude=float(args.z_wave_amplitude),
        z_mode=int(args.z_mode),
    )
    rhs0, _ = assemble_rhs_cached(
        state0, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    rhs_wave, _ = assemble_rhs_cached(
        z_wave, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    _block_tree((rhs0, rhs_wave))
    states = [
        ("initial", state0),
        ("initial_linear_kick", state0 + args.kick_dt * rhs0),
        ("z_wave", z_wave),
        ("z_wave_linear_kick", z_wave + args.kick_dt * rhs_wave),
    ]
    rows = [
        _evaluate_state(name, state, cache=cache, params=params, term_cfg=term_cfg)
        for name, state in states
    ]
    payload = _build_summary(
        rows,
        config=str(args.config),
        ky=args.ky,
        kx=args.kx,
        nl=args.Nl,
        nm=args.Nm,
        identity_threshold=args.identity_threshold,
        activation_threshold=args.activation_threshold,
    )
    _write_json(payload, args.out_json)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"saved {args.out_json}")
    return 0 if payload["passed"] else 1

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


def _plot_abs_rows(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    m = np.asarray([row["m"] for row in rows], dtype=float)
    left_key = "production_abs" if "production_abs" in rows[0] else "serial_abs"
    left_label = "linear_rhs streaming" if left_key == "production_abs" else "serial streaming+phi"
    right_label = "shard_map streaming" if left_key == "production_abs" else "velocity route"
    serial = np.asarray([row[left_key] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_abs"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)
    title = "Streaming-only linear RHS" if left_key == "production_abs" else "Electrostatic streaming RHS"

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, serial, "s-", lw=1.8, label=left_label)
    axes[0].plot(m, sharded, "^--", lw=1.8, label=right_label)
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("absolute value")
    axes[0].set_title(title)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        m, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_xlabel("Hermite index m")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def _plot_norm_rows(summary: dict[str, object], paths: dict[str, Path]) -> None:
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
    axes[0].set_title("Composed electrostatic linear RHS")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(m, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="max abs error")
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


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for any linear-RHS gate."""

    paths = _write_common(summary, out_prefix)
    rows = list(summary["rows"])
    if rows and "serial_norm" in rows[0]:
        _plot_norm_rows(summary, paths)
    elif rows and ("production_abs" in rows[0] or "serial_abs" in rows[0]):
        _plot_abs_rows(summary, paths)
    else:
        raise ValueError("Unrecognized linear-RHS gate row schema")
    return {name: str(path) for name, path in paths.items()}


def _add_common_args(parser: argparse.ArgumentParser, *, out_prefix: Path) -> None:
    parser.add_argument("--out-prefix", type=Path, default=out_prefix)
    parser.add_argument("--logical-devices", type=int, default=2)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-6)


def _run_streaming(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_linear_rhs_streaming_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_streaming_electrostatic(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_linear_rhs_streaming_electrostatic_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_electrostatic_slices(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_linear_rhs_electrostatic_slices_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    streaming = subparsers.add_parser(
        "streaming", help="Gate streaming-only linear_rhs_cached."
    )
    _add_common_args(streaming, out_prefix=DEFAULT_STREAMING_PREFIX)
    streaming.set_defaults(func=_run_streaming)

    streaming_electrostatic = subparsers.add_parser(
        "streaming-electrostatic", help="Gate streaming with nonzero electrostatic phi."
    )
    _add_common_args(streaming_electrostatic, out_prefix=DEFAULT_ELECTROSTATIC_PREFIX)
    streaming_electrostatic.set_defaults(func=_run_streaming_electrostatic)

    slices = subparsers.add_parser(
        "electrostatic-slices", help="Gate the composed electrostatic linear-slices route."
    )
    _add_common_args(slices, out_prefix=DEFAULT_SLICES_PREFIX)
    slices.set_defaults(func=_run_electrostatic_slices)

    zero_norm = subparsers.add_parser(
        "zero-norm-state-window",
        help="Gate profiler-zero linear RHS terms over an active state window.",
    )
    _add_zero_norm_args(zero_norm)
    zero_norm.set_defaults(func=_run_zero_norm_state_window)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
