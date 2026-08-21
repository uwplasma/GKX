#!/usr/bin/env python3
"""Generate velocity-space parallel identity-gate artifacts."""

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
DEFAULT_HERMITE_PREFIX = REPO_ROOT / "docs" / "_static" / "hermite_exchange_gate"
DEFAULT_REDUCE_PREFIX = REPO_ROOT / "docs" / "_static" / "velocity_field_reduce_gate"
DEFAULT_LADDER_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "hermite_streaming_ladder_gate"
)
DEFAULT_PERIODIC_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "periodic_streaming_microkernel_gate"
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


def _state(shape: tuple[int, ...]) -> Any:
    import jax.numpy as jnp

    if len(shape) == 6:
        nz = int(shape[-1])
        z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
        state = jnp.zeros(shape, dtype=jnp.complex64)
        state = state.at[0, 0, 1, 0, 0, :].set(jnp.exp(1j * z))
        state = state.at[0, 1, 3, 1, 0, :].set(0.3 * jnp.exp(2j * z))
        return state, z

    values = jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(shape)
    return values + 0.01j * (values + 1.0)


def build_hermite_exchange_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    atol: float,
) -> dict[str, object]:
    """Compare the sharded Hermite ghost exchange against the full-array shift."""

    import jax
    import jax.numpy as jnp

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        hermite_neighbor_reference,
        hermite_neighbor_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(device_list)} are available"
        )
    state = _state(shape)
    plan = build_velocity_sharding_plan(
        shape, num_devices=len(device_list), axes=("hermite",)
    )
    lower_ref, upper_ref = hermite_neighbor_reference(state)
    lower_sharded, upper_sharded = hermite_neighbor_shard_map(
        state, plan, devices=device_list
    )
    _block_until_ready((lower_sharded, upper_sharded))

    lower_err = jnp.max(jnp.abs(lower_sharded - lower_ref))
    upper_err = jnp.max(jnp.abs(upper_sharded - upper_ref))
    _block_until_ready((lower_err, upper_err))
    max_lower_abs_error = float(np.asarray(lower_err))
    max_upper_abs_error = float(np.asarray(upper_err))
    identity_passed = bool(max(max_lower_abs_error, max_upper_abs_error) <= float(atol))

    center_trace = np.asarray(state[0, :, 0, 0, 0])
    lower_trace = np.asarray(lower_sharded[0, :, 0, 0, 0])
    upper_trace = np.asarray(upper_sharded[0, :, 0, 0, 0])
    lower_ref_trace = np.asarray(lower_ref[0, :, 0, 0, 0])
    upper_ref_trace = np.asarray(upper_ref[0, :, 0, 0, 0])
    rows = []
    for m_idx in range(shape[1]):
        rows.append(
            {
                "m": int(m_idx),
                "center_real": float(np.real(center_trace[m_idx])),
                "lower_real": float(np.real(lower_trace[m_idx])),
                "upper_real": float(np.real(upper_trace[m_idx])),
                "lower_reference_real": float(np.real(lower_ref_trace[m_idx])),
                "upper_reference_real": float(np.real(upper_ref_trace[m_idx])),
                "lower_abs_error": float(
                    abs(lower_trace[m_idx] - lower_ref_trace[m_idx])
                ),
                "upper_abs_error": float(
                    abs(upper_trace[m_idx] - upper_ref_trace[m_idx])
                ),
            }
        )

    return _json_clean(
        {
            "case": "Hermite ghost-exchange shard_map identity gate",
            "source": "gkx.parallel.velocity.hermite_neighbor_shard_map",
            "claim_scope": "communication-kernel identity gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_lower_abs_error": max_lower_abs_error,
            "max_upper_abs_error": max_upper_abs_error,
            "atol": float(atol),
            "identity_passed": identity_passed,
            "rows": rows,
        }
    )


def build_velocity_field_reduce_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare the sharded Hermite field reduction against the full-array sum."""

    import jax
    import jax.numpy as jnp

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        velocity_field_reduce_reference,
        velocity_field_reduce_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(device_list)} are available"
        )
    state = _state(shape)
    plan = build_velocity_sharding_plan(
        shape, num_devices=len(device_list), axes=("hermite",)
    )
    reference = velocity_field_reduce_reference(state, axis="hermite")
    sharded = velocity_field_reduce_shard_map(
        state, plan, axis="hermite", devices=device_list
    )
    _block_until_ready(sharded)

    err = jnp.max(jnp.abs(sharded - reference))
    reference_norm = jnp.max(jnp.abs(reference))
    _block_until_ready((err, reference_norm))
    max_abs_error = float(np.asarray(err))
    max_abs_reference = float(np.asarray(reference_norm))
    max_allowed_error = float(atol) + float(rtol) * max(max_abs_reference, 1.0)
    max_rel_error = max_abs_error / max(max_abs_reference, 1.0e-30)
    identity_passed = bool(max_abs_error <= max_allowed_error)

    sharded_trace = np.asarray(sharded[0, :, 0, 0])
    reference_trace = np.asarray(reference[0, :, 0, 0])
    rows = []
    for ky_idx in range(shape[2]):
        rows.append(
            {
                "ky_index": int(ky_idx),
                "reduced_real": float(np.real(sharded_trace[ky_idx])),
                "reference_real": float(np.real(reference_trace[ky_idx])),
                "abs_error": float(
                    abs(sharded_trace[ky_idx] - reference_trace[ky_idx])
                ),
                "rel_error": float(
                    abs(sharded_trace[ky_idx] - reference_trace[ky_idx])
                    / max(abs(reference_trace[ky_idx]), 1.0e-30)
                ),
            }
        )

    return _json_clean(
        {
            "case": "Velocity-space field-reduction shard_map identity gate",
            "source": "gkx.parallel.velocity.velocity_field_reduce_shard_map",
            "claim_scope": "communication-kernel identity gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "reduction_axis": "hermite",
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_abs_error": max_abs_error,
            "max_abs_reference": max_abs_reference,
            "max_rel_error": max_rel_error,
            "max_allowed_error": max_allowed_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
        }
    )


def build_hermite_streaming_ladder_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    vth: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare the sharded Hermite streaming ladder against the reference."""

    import jax
    import jax.numpy as jnp

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        hermite_streaming_ladder_reference,
        hermite_streaming_ladder_shard_map,
        velocity_field_reduce_reference,
        velocity_field_reduce_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(device_list)} are available"
        )
    state = _state(shape)
    plan = build_velocity_sharding_plan(
        shape, num_devices=len(device_list), axes=("hermite",)
    )

    ladder_reference = hermite_streaming_ladder_reference(state, vth=float(vth))
    ladder_sharded = hermite_streaming_ladder_shard_map(
        state, plan, vth=float(vth), devices=device_list
    )
    reduction_reference = velocity_field_reduce_reference(state, axis="hermite")
    reduction_sharded = velocity_field_reduce_shard_map(
        state, plan, axis="hermite", devices=device_list
    )
    _block_until_ready((ladder_sharded, reduction_sharded))

    ladder_abs = jnp.max(jnp.abs(ladder_sharded - ladder_reference))
    ladder_scale = jnp.max(jnp.abs(ladder_reference))
    ladder_rel = ladder_abs / jnp.maximum(
        ladder_scale, jnp.asarray(1.0e-30, dtype=ladder_scale.dtype)
    )
    reduction_abs = jnp.max(jnp.abs(reduction_sharded - reduction_reference))
    _block_until_ready((ladder_abs, ladder_rel, reduction_abs))
    max_ladder_abs_error = float(np.asarray(ladder_abs))
    max_ladder_rel_error = float(np.asarray(ladder_rel))
    max_reduction_abs_error = float(np.asarray(reduction_abs))
    identity_passed = bool(
        max_ladder_abs_error <= float(atol) and max_ladder_rel_error <= float(rtol)
    )

    state_trace = np.asarray(state[0, :, 0, 0, 0])
    reference_trace = np.asarray(ladder_reference[0, :, 0, 0, 0])
    sharded_trace = np.asarray(ladder_sharded[0, :, 0, 0, 0])
    rows = []
    for m_idx in range(shape[1]):
        rows.append(
            {
                "m": int(m_idx),
                "state_real": float(np.real(state_trace[m_idx])),
                "ladder_real": float(np.real(sharded_trace[m_idx])),
                "reference_real": float(np.real(reference_trace[m_idx])),
                "abs_error": float(abs(sharded_trace[m_idx] - reference_trace[m_idx])),
            }
        )

    return _json_clean(
        {
            "case": "Hermite streaming-ladder shard_map identity gate",
            "source": "gkx.parallel.velocity.hermite_streaming_ladder_shard_map",
            "claim_scope": "Hermite streaming communication/coefficient gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "vth": float(vth),
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_ladder_abs_error": max_ladder_abs_error,
            "max_ladder_rel_error": max_ladder_rel_error,
            "max_reduction_abs_error": max_reduction_abs_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "This gate combines nearest-Hermite shard_map exchange with the sqrt(m+1)/sqrt(m) "
                "streaming-ladder coefficients. It also records the accompanying Hermite field-reduction "
                "error because production velocity-space sharding needs both communication primitives."
            ),
        }
    )


def _production_streaming_term(state: Any, *, kz: Any, vth: float) -> Any:
    import jax.numpy as jnp

    from gkx.core.velocity import hermite_ladder_coeffs
    from gkx.operators.linear.streaming import streaming_ladder_term

    nm = int(state.shape[-4])
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    return streaming_ladder_term(
        state,
        kz=kz,
        vth=jnp.asarray(vth, dtype=jnp.float32).reshape((1, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
    )


def build_periodic_streaming_microkernel_gate(
    *,
    shape: tuple[int, int, int, int, int, int],
    requested_devices: int,
    vth: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare the sharded periodic streaming microkernel to production."""

    import jax
    import jax.numpy as jnp

    from gkx.parallel.velocity import (
        build_velocity_sharding_plan,
        periodic_streaming_reference,
        periodic_streaming_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(device_list)} are available"
        )
    state, z = _state(shape)
    dz = float(np.asarray(z[1] - z[0]))
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(int(shape[-1]), d=dz)
    plan = build_velocity_sharding_plan(
        shape, num_devices=len(device_list), axes=("hermite",)
    )

    production = _production_streaming_term(state, kz=kz, vth=float(vth))
    reference = periodic_streaming_reference(
        state, kz=kz, vth=jnp.asarray([vth], dtype=jnp.float32)
    )
    sharded = periodic_streaming_shard_map(
        state,
        plan,
        kz=kz,
        vth=jnp.asarray([vth], dtype=jnp.float32),
        devices=device_list,
    )
    _block_until_ready((production, reference, sharded))

    ref_abs = jnp.max(jnp.abs(reference - production))
    sharded_abs = jnp.max(jnp.abs(sharded - production))
    scale = jnp.max(jnp.abs(production))
    sharded_rel = sharded_abs / jnp.maximum(
        scale, jnp.asarray(1.0e-30, dtype=scale.dtype)
    )
    _block_until_ready((ref_abs, sharded_abs, sharded_rel))
    max_reference_abs_error = float(np.asarray(ref_abs))
    max_sharded_abs_error = float(np.asarray(sharded_abs))
    max_sharded_rel_error = float(np.asarray(sharded_rel))
    identity_passed = bool(
        max_sharded_abs_error <= float(atol) and max_sharded_rel_error <= float(rtol)
    )

    production_trace = np.asarray(production[0, 0, :, 0, 0, 1])
    sharded_trace = np.asarray(sharded[0, 0, :, 0, 0, 1])
    state_trace = np.asarray(state[0, 0, :, 0, 0, 1])
    rows = []
    for m_idx in range(shape[2]):
        rows.append(
            {
                "m": int(m_idx),
                "state_abs": float(abs(state_trace[m_idx])),
                "production_abs": float(abs(production_trace[m_idx])),
                "sharded_abs": float(abs(sharded_trace[m_idx])),
                "abs_error": float(abs(sharded_trace[m_idx] - production_trace[m_idx])),
            }
        )

    return _json_clean(
        {
            "case": "Periodic streaming microkernel shard_map identity gate",
            "source": "gkx.parallel.velocity.periodic_streaming_shard_map",
            "reference_source": "gkx.operators.linear.streaming.streaming_ladder_term",
            "claim_scope": "linear streaming microkernel identity gate, not a full RHS or nonlinear speedup claim",
            "state_shape": shape,
            "vth": float(vth),
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_reference_abs_error": max_reference_abs_error,
            "max_sharded_abs_error": max_sharded_abs_error,
            "max_sharded_rel_error": max_sharded_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
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


def _plot_hermite(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    m = np.asarray([row["m"] for row in rows], dtype=float)
    center = np.asarray([row["center_real"] for row in rows], dtype=float)
    lower = np.asarray([row["lower_real"] for row in rows], dtype=float)
    upper = np.asarray([row["upper_real"] for row in rows], dtype=float)
    lower_err = np.asarray([row["lower_abs_error"] for row in rows], dtype=float)
    upper_err = np.asarray([row["upper_abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(m, center, "o-", lw=2.0, label="center")
    axes[0].plot(m, lower, "s--", lw=1.8, label="lower neighbor")
    axes[0].plot(m, upper, "^--", lw=1.8, label="upper neighbor")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("real value")
    axes[0].set_title("Nearest-Hermite exchange")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        m, np.maximum(lower_err, 1.0e-16), "s-", lw=2.0, label="lower error"
    )
    axes[1].semilogy(
        m, np.maximum(upper_err, 1.0e-16), "^-", lw=2.0, label="upper error"
    )
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="gate")
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


def _plot_reduce(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    ky = np.asarray([row["ky_index"] for row in rows], dtype=float)
    reduced = np.asarray([row["reduced_real"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_real"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(ky, reference, "o-", lw=2.0, label="reference reduction")
    axes[0].plot(ky, reduced, "s--", lw=1.8, label="shard_map reduction")
    axes[0].set_xlabel(r"$k_y$ index")
    axes[0].set_ylabel("real reduced field contribution")
    axes[0].set_title("Velocity reduction")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        ky, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(
        float(summary["max_allowed_error"]),
        ls=":",
        lw=1.2,
        color="0.25",
        label="gate",
    )
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_xlabel(r"$k_y$ index")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(paths["png"], dpi=220)
    fig.savefig(paths["pdf"])
    plt.close(fig)


def _plot_ladder(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    m = np.asarray([row["m"] for row in rows], dtype=float)
    state = np.asarray([row["state_real"] for row in rows], dtype=float)
    ladder = np.asarray([row["ladder_real"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_real"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, state, "o-", lw=2.0, label="state")
    axes[0].plot(m, reference, "s-", lw=1.8, label="reference ladder")
    axes[0].plot(m, ladder, "^--", lw=1.8, label="shard_map ladder")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("real value")
    axes[0].set_title("Streaming ladder")
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


def _plot_periodic(summary: dict[str, object], paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gkx.artifacts.plotting import set_plot_style

    rows = list(summary["rows"])
    m = np.asarray([row["m"] for row in rows], dtype=float)
    production = np.asarray([row["production_abs"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_abs"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, production, "s-", lw=1.8, label="production streaming")
    axes[0].plot(m, sharded, "^--", lw=1.8, label="shard_map streaming")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("absolute value")
    axes[0].set_title("Periodic streaming")
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


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for any velocity parallel gate."""

    paths = _write_common(summary, out_prefix)
    rows = list(summary["rows"])
    if rows and "center_real" in rows[0]:
        _plot_hermite(summary, paths)
    elif rows and "state_real" in rows[0]:
        _plot_ladder(summary, paths)
    elif rows and "production_abs" in rows[0]:
        _plot_periodic(summary, paths)
    elif rows and "ky_index" in rows[0]:
        _plot_reduce(summary, paths)
    else:
        raise ValueError("Unrecognized velocity parallel gate row schema")
    return {name: str(path) for name, path in paths.items()}


def _add_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--logical-devices", type=int, default=2)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--atol", type=float, default=None)


def _shape(args: argparse.Namespace) -> tuple[int, int, int, int, int]:
    return (int(args.nl), int(args.nm), int(args.ny), int(args.nx), int(args.nz))


def _shape6(args: argparse.Namespace) -> tuple[int, int, int, int, int, int]:
    return (
        int(args.ns),
        int(args.nl),
        int(args.nm),
        int(args.ny),
        int(args.nx),
        int(args.nz),
    )


def _run_hermite(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_hermite_exchange_gate(
        shape=_shape(args),
        requested_devices=int(args.logical_devices),
        atol=1.0e-7 if args.atol is None else float(args.atol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_reduce(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_velocity_field_reduce_gate(
        shape=_shape(args),
        requested_devices=int(args.logical_devices),
        atol=1.0e-5 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_ladder(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_hermite_streaming_ladder_gate(
        shape=_shape(args),
        requested_devices=int(args.logical_devices),
        vth=float(args.vth),
        atol=1.0e-5 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def _run_periodic(args: argparse.Namespace) -> int:
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_periodic_streaming_microkernel_gate(
        shape=_shape6(args),
        requested_devices=int(args.logical_devices),
        vth=float(args.vth),
        atol=1.0e-5 if args.atol is None else float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hermite = subparsers.add_parser(
        "hermite-exchange", help="Generate the Hermite ghost-exchange gate."
    )
    hermite.add_argument("--out-prefix", type=Path, default=DEFAULT_HERMITE_PREFIX)
    _add_shape_args(hermite)
    hermite.set_defaults(func=_run_hermite)

    reduce = subparsers.add_parser(
        "field-reduce", help="Generate the Hermite field-reduction gate."
    )
    reduce.add_argument("--out-prefix", type=Path, default=DEFAULT_REDUCE_PREFIX)
    _add_shape_args(reduce)
    reduce.add_argument("--rtol", type=float, default=1.0e-7)
    reduce.set_defaults(func=_run_reduce)

    ladder = subparsers.add_parser(
        "hermite-ladder", help="Generate the Hermite streaming-ladder gate."
    )
    ladder.add_argument("--out-prefix", type=Path, default=DEFAULT_LADDER_PREFIX)
    _add_shape_args(ladder)
    ladder.add_argument("--vth", type=float, default=1.7)
    ladder.add_argument("--rtol", type=float, default=1.0e-6)
    ladder.set_defaults(func=_run_ladder)

    periodic = subparsers.add_parser(
        "periodic-streaming", help="Generate the periodic streaming microkernel gate."
    )
    periodic.add_argument("--out-prefix", type=Path, default=DEFAULT_PERIODIC_PREFIX)
    periodic.add_argument("--logical-devices", type=int, default=2)
    periodic.add_argument("--ns", type=int, default=1)
    periodic.add_argument("--nl", type=int, default=2)
    periodic.add_argument("--nm", type=int, default=8)
    periodic.add_argument("--ny", type=int, default=2)
    periodic.add_argument("--nx", type=int, default=1)
    periodic.add_argument("--nz", type=int, default=16)
    periodic.add_argument("--vth", type=float, default=1.7)
    periodic.add_argument("--atol", type=float, default=None)
    periodic.add_argument("--rtol", type=float, default=1.0e-6)
    periodic.set_defaults(func=_run_periodic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
