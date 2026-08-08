#!/usr/bin/env python3
"""Profile the dominant cached linear RHS term kernels on a real runtime state."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.core.grid import build_spectral_grid
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import build_H
from gkx.operators.linear.params import _as_species_array
from gkx.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.terms.assembly import (
    _rhs_field_views,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from gkx.terms.linear_terms import (
    _hypercollision_kz_source,
    collisions_contribution,
    curvature_gradb_contribution,
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
    mirror_contribution,
    linked_streaming_contribution,
)
from gkx.operators.linear.streaming import abs_z_linked_fft, grad_z_linked_fft, shift_axis


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile per-term linear RHS kernels.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"),
    )
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--kx", type=float, default=None)
    p.add_argument("--Nl", type=int, default=4)
    p.add_argument("--Nm", type=int, default=8)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--state",
        choices=("initial", "z_wave", "z_wave_linear_kick"),
        default="initial",
        help="State to profile. z_wave variants activate resolved parallel variation.",
    )
    p.add_argument("--z-mode", type=int, default=1)
    p.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    p.add_argument("--kick-dt", type=float, default=1.0e-3)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to the CSV path with a .json suffix.",
    )
    return p.parse_args()


def _block_tree(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        jax.block_until_ready(leaf)


def _required_array(name: str, value: jnp.ndarray | None) -> jnp.ndarray:
    if value is None:
        raise RuntimeError(f"{name} must be built before linked profiling")
    return value


def _time_callable(fn, *args, repeats: int):
    out = fn(*args)
    _block_tree(out)
    t0 = time.perf_counter()
    last = out
    for _ in range(repeats):
        last = fn(*args)
        _block_tree(last)
    t1 = time.perf_counter()
    return (t1 - t0) / float(repeats), last


def _z_variation_norm(state: jnp.ndarray) -> float:
    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return float(np.asarray(jnp.linalg.norm(state - mean_z)))


def _inject_z_wave(
    state: jnp.ndarray,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> jnp.ndarray:
    """Inject a deterministic resolved parallel wave into one Hermite mode."""

    state = jnp.asarray(state)
    Nz = state.shape[-1]
    Nm = state.shape[-4]
    m_index = min(max(1, Nm - 1), 3)
    z = jnp.arange(Nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(Nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - runtime state builder controls dimensionality.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    ratio = float(numerator) / float(denominator)
    return ratio if math.isfinite(ratio) else None


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    config: str,
    ky: float,
    kx: float | None,
    nl: int,
    nm: int,
    repeats: int,
    backend: str,
    state: str = "initial",
    z_variation_norm: float | None = None,
    zero_norm_threshold: float = 1.0e-13,
) -> dict[str, Any]:
    """Build a machine-readable summary for the linear RHS split profile."""

    term_rows = {
        str(row["term"]): {
            "seconds": float(row["seconds"]),
            "norm": float(row["norm"]),
        }
        for row in rows
    }
    measured_terms = [
        (term, values)
        for term, values in term_rows.items()
        if term not in {"full_linear_rhs"} and math.isfinite(values["seconds"])
    ]
    measured_nonzero_terms = [
        (term, values)
        for term, values in measured_terms
        if abs(values["norm"]) > zero_norm_threshold
    ]
    dominant = max(
        measured_terms, default=(None, None), key=lambda item: float(item[1]["seconds"])
    )
    dominant_nonzero = max(
        measured_nonzero_terms,
        default=(None, None),
        key=lambda item: float(item[1]["seconds"]),
    )
    zero_norm_terms: list[dict[str, Any]] = [
        {
            "term": term,
            "seconds": values["seconds"],
            "norm": values["norm"],
        }
        for term, values in measured_terms
        if abs(values["norm"]) <= zero_norm_threshold
    ]
    zero_norm_terms.sort(key=lambda row: float(row["seconds"]), reverse=True)

    full_seconds = term_rows.get("full_linear_rhs", {}).get("seconds")
    measured_sum = sum(values["seconds"] for _, values in measured_terms)
    return {
        "kind": "linear_rhs_terms_profile_summary",
        "case": Path(config).stem,
        "config": config,
        "backend": backend,
        "ky": float(ky),
        "kx": None if kx is None else float(kx),
        "Nl": int(nl),
        "Nm": int(nm),
        "repeats": int(repeats),
        "state": state,
        "z_variation_norm": None
        if z_variation_norm is None
        else float(z_variation_norm),
        "zero_norm_threshold": float(zero_norm_threshold),
        "rows": term_rows,
        "dominant_measured_term": dominant[0],
        "dominant_nonzero_norm_term": dominant_nonzero[0],
        "zero_norm_terms_by_time": zero_norm_terms,
        "full_linear_rhs_seconds": full_seconds,
        "sum_independently_measured_components_seconds": measured_sum,
        "full_over_sum_independently_measured_components": _safe_ratio(
            full_seconds, measured_sum
        ),
        "claim_scope": (
            "Linear RHS split profile for hot-path localization. Zero-norm rows describe the "
            "profiled initial state only and must not be skipped in production unless a "
            "state-window identity gate proves the term remains inactive."
        ),
    }


def _write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
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
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=args.Nl,
        Nm=args.Nm,
        nspecies=len(cfg.species),
    )
    cache = build_linear_cache(grid, geom, params, args.Nl, args.Nm)
    G0 = jnp.asarray(G0)
    if args.state != "initial":
        G0 = _inject_z_wave(
            G0,
            ky_index=int(ky_index),
            kx_index=int(kx_index),
            amplitude=float(args.z_wave_amplitude),
            z_mode=int(args.z_mode),
        )
        if args.state == "z_wave_linear_kick":
            kick_rhs, _kick_fields = assemble_rhs_cached_jit(
                G0, cache, params, term_cfg
            )
            _block_tree(kick_rhs)
            G0 = G0 + float(args.kick_dt) * kick_rhs
    state_z_variation_norm = _z_variation_norm(G0)
    out_dtype = jnp.result_type(G0, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)
    ns = int(G0.shape[0])

    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.tprim, ns, "tprim").astype(real_dtype)
    fprim = _as_species_array(params.fprim, ns, "fprim").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)

    field_fn = jax.jit(
        lambda G: compute_fields_cached(G, cache, params, terms=term_cfg)
    )
    fields_time, fields = _time_callable(field_fn, G0, repeats=args.repeats)
    apar, bpar, h_apar, h_bpar = _rhs_field_views(fields, term_cfg)
    H = build_H(
        G0, cache.Jl, fields.phi, tz, apar=h_apar, vth=vth, bpar=h_bpar, JlB=cache.JlB
    )

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper = _as_species_array(nu_hyper, ns, "nu_hyper").astype(real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    hypercollisions_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    hypercollisions_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
    D_hyper = jnp.asarray(params.D_hyper, dtype=real_dtype)
    p_hyper_kperp = jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)

    streaming_rhs_pre_grad = None
    hyper_kz_source = None
    if cache.use_twist_shift:
        axis_m = -4
        G_p1 = shift_axis(G0, 1, axis=axis_m)
        G_m1 = shift_axis(G0, -1, axis=axis_m)
        vth_s = vth[:, None, None, None, None, None]
        streaming_rhs_pre_grad = -vth_s * (
            cache.sqrt_p * G_p1 + cache.sqrt_m_ladder * G_m1
        )

        tz_arr = tz[:, None, None, None, None, None]
        zt = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
        zt5 = zt[:, 0, 0, 0, 0, 0][:, None, None, None, None]
        vth5 = vth[:, None, None, None, None]
        phi_s = fields.phi[None, None, ...]
        Nm = streaming_rhs_pre_grad.shape[2]
        m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
        field_rhs = jnp.zeros_like(streaming_rhs_pre_grad)
        if h_apar is not None:
            apar_s = h_apar[None, None, ...]
            drive_m0 = zt5 * (vth5 * vth5) * cache.Jl * apar_s
            field_rhs = (
                field_rhs
                + (m_idx == 0).astype(field_rhs.dtype) * drive_m0[:, :, None, ...]
            )
        if Nm > 1:
            drive_m1 = -zt5 * vth5 * cache.Jl * phi_s
            if h_bpar is not None:
                drive_m1 = drive_m1 - vth5 * cache.JlB * h_bpar[None, None, ...]
            field_rhs = (
                field_rhs
                + (m_idx == 1).astype(field_rhs.dtype) * drive_m1[:, :, None, ...]
            )
        if Nm > 2 and h_apar is not None:
            drive_m2 = jnp.sqrt(2.0) * zt5 * (vth5 * vth5) * cache.Jl * apar_s
            field_rhs = (
                field_rhs
                + (m_idx == 2).astype(field_rhs.dtype) * drive_m2[:, :, None, ...]
            )
        streaming_rhs_pre_grad = kpar_scale * (streaming_rhs_pre_grad + field_rhs)

        hyper_kz_source = _hypercollision_kz_source(
            G0,
            weight=jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype),
            hypercollisions_kz=hypercollisions_kz,
            nu_hyper_m=nu_hyper_m,
            m_norm_kz_factor=cache.m_norm_kz_factor,
            vth=vth,
            kpar_scale=kpar_scale,
            mask_kz=cache.mask_kz,
            m_pow=cache.m_pow,
        )

    kernel_fns = {
        "build_H": jax.jit(
            lambda: build_H(
                G0,
                cache.Jl,
                fields.phi,
                tz,
                apar=h_apar,
                vth=vth,
                bpar=h_bpar,
                JlB=cache.JlB,
            )
        ),
        "streaming": jax.jit(
            lambda: linked_streaming_contribution(
                G0,
                phi=fields.phi,
                apar=h_apar,
                bpar=h_bpar,
                Jl=cache.Jl,
                JlB=cache.JlB,
                tz=tz,
                kz=cache.kz,
                dz=cache.dz,
                vth=vth,
                sqrt_p=cache.sqrt_p,
                sqrt_m=cache.sqrt_m_ladder,
                kpar_scale=kpar_scale,
                weight=jnp.asarray(term_cfg.streaming, dtype=real_dtype),
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
                use_twist_shift=cache.use_twist_shift,
            )
        ),
        "linked_grad_z": (
            jax.jit(
                lambda: grad_z_linked_fft(
                    _required_array("streaming_rhs_pre_grad", streaming_rhs_pre_grad),
                    dz=cache.dz,
                    linked_indices=cache.linked_indices,
                    linked_kz=cache.linked_kz,
                    linked_inverse_permutation=cache.linked_inverse_permutation,
                    linked_full_cover=cache.linked_full_cover,
                    linked_gather_map=cache.linked_gather_map,
                    linked_gather_mask=cache.linked_gather_mask,
                    linked_use_gather=cache.linked_use_gather,
                )
            )
            if cache.use_twist_shift
            else None
        ),
        "mirror": jax.jit(
            lambda: mirror_contribution(
                H,
                vth=vth,
                bgrad=cache.bgrad,
                ell=cache.l,
                sqrt_m=cache.sqrt_m,
                sqrt_m_p1=cache.sqrt_m_p1,
                weight=jnp.asarray(term_cfg.mirror, dtype=real_dtype),
            )
        ),
        "curvature_gradb": jax.jit(
            lambda: curvature_gradb_contribution(
                H,
                tz=tz,
                omega_d_scale=omega_d_scale,
                cv_d=cache.cv_d,
                gb_d=cache.gb_d,
                ell=cache.l,
                m=cache.m,
                imag=imag,
                weight_curv=jnp.asarray(term_cfg.curvature, dtype=real_dtype),
                weight_gradb=jnp.asarray(term_cfg.gradb, dtype=real_dtype),
            )
        ),
        "diamagnetic": jax.jit(
            lambda: diamagnetic_contribution(
                jnp.zeros_like(G0),
                phi=fields.phi,
                apar=h_apar,
                bpar=h_bpar,
                Jl=cache.Jl,
                b=cache.b,
                JlB=cache.JlB,
                l4=cache.l4,
                tprim=tprim,
                fprim=fprim,
                tz=tz,
                vth=vth,
                omega_star_scale=omega_star_scale,
                ky=cache.ky,
                imag=imag,
                weight=jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype),
            )
        ),
        "collisions": jax.jit(
            lambda: collisions_contribution(
                H,
                G=G0,
                Jl=cache.Jl,
                JlB=cache.JlB,
                b=cache.b,
                nu=nu,
                lb_lam=cache.lb_lam,
                weight=jnp.asarray(term_cfg.collisions, dtype=real_dtype),
            )
        ),
        "hypercollisions": jax.jit(
            lambda: hypercollisions_contribution(
                G0,
                vth=vth,
                nu_hyper=nu_hyper,
                nu_hyper_l=nu_hyper_l,
                nu_hyper_m=nu_hyper_m,
                nu_hyper_lm=nu_hyper_lm,
                hyper_ratio=cache.hyper_ratio,
                ratio_l=cache.ratio_l,
                ratio_m=cache.ratio_m,
                ratio_lm=cache.ratio_lm,
                mask_const=cache.mask_const,
                mask_kz=cache.mask_kz,
                m_pow=cache.m_pow,
                m_norm_kz_factor=cache.m_norm_kz_factor,
                kz=cache.kz,
                kpar_scale=kpar_scale,
                hypercollisions_const=hypercollisions_const,
                hypercollisions_kz=hypercollisions_kz,
                weight=jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype),
                linked_indices=cache.linked_indices,
                linked_kz=cache.linked_kz,
                linked_inverse_permutation=cache.linked_inverse_permutation,
                linked_full_cover=cache.linked_full_cover,
                linked_gather_map=cache.linked_gather_map,
                linked_gather_mask=cache.linked_gather_mask,
                linked_use_gather=cache.linked_use_gather,
            )
        ),
        "linked_abs_kz": (
            jax.jit(
                lambda: abs_z_linked_fft(
                    _required_array("hyper_kz_source", hyper_kz_source),
                    linked_indices=cache.linked_indices,
                    linked_kz=cache.linked_kz,
                    linked_inverse_permutation=cache.linked_inverse_permutation,
                    linked_full_cover=cache.linked_full_cover,
                    linked_gather_map=cache.linked_gather_map,
                    linked_gather_mask=cache.linked_gather_mask,
                    linked_use_gather=cache.linked_use_gather,
                )
            )
            if cache.use_twist_shift
            else None
        ),
        "hyperdiffusion": jax.jit(
            lambda: hyperdiffusion_contribution(
                G0,
                kx=cache.kx,
                ky=cache.ky,
                dealias_mask=cache.dealias_mask,
                D_hyper=D_hyper,
                p_hyper_kperp=p_hyper_kperp,
                weight=jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype),
            )
        ),
        "end_damping": jax.jit(
            lambda: end_damping_contribution(
                H,
                ky=cache.ky,
                damp_profile=cache.damp_profile,
                linked_damp_profile=cache.linked_damp_profile,
                damp_amp=damp_amp,
                weight=jnp.asarray(term_cfg.end_damping, dtype=real_dtype),
            )
        ),
        "full_linear_rhs": jax.jit(
            lambda: assemble_rhs_cached_jit(G0, cache, params, term_cfg)
        ),
    }

    rows = [
        {
            "term": "field_solve",
            "seconds": fields_time,
            "norm": float(np.asarray(jnp.linalg.norm(fields.phi))),
        }
    ]
    for name, fn in kernel_fns.items():
        if fn is None:
            continue
        seconds, out = _time_callable(fn, repeats=args.repeats)
        arr = out[0] if name == "full_linear_rhs" else out
        rows.append(
            {
                "term": name,
                "seconds": seconds,
                "norm": float(np.asarray(jnp.linalg.norm(arr))),
            }
        )

    for row in rows:
        print(f"{row['term']}: seconds={row['seconds']:.6f} norm={row['norm']:.6e}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["term", "seconds", "norm"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")
    summary_path = args.summary_json
    if summary_path is None and args.out is not None:
        summary_path = args.out.with_suffix(".json")
    if summary_path is not None:
        _write_summary_json(
            _build_summary(
                rows,
                config=str(args.config),
                ky=args.ky,
                kx=args.kx,
                nl=args.Nl,
                nm=args.Nm,
                repeats=args.repeats,
                backend=jax.default_backend(),
                state=args.state,
                z_variation_norm=state_z_variation_norm,
            ),
            summary_path,
        )
        print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
