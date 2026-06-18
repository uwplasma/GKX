"""Branch-locality gates for VMEC-JAX transport objectives."""

from __future__ import annotations

from typing import Any, cast

from spectraxgk.geometry.vmec_boozer_core import (
    flux_tube_geometry_from_vmec_boozer_state,
)
from spectraxgk.objectives.core import solver_linear_operator_matrix_from_geometry
from spectraxgk.objectives.eigen import dominant_eigenvalue_branch_locality_report
from spectraxgk.objectives.vmec_transport_config import (
    VMECJAXTransportObjectiveConfig,
    _pin_current_optional_backend_paths,
)
from spectraxgk.objectives.vmec_transport_tables import (
    _static_grid_options_from_ky_values,
)


def vmec_jax_transport_growth_branch_locality_report_from_states(
    base_state: Any,
    plus_state: Any,
    minus_state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig | None = None,
    *,
    step: float,
    gap_floor: float = 1.0e-8,
    slope_rtol: float = 1.0e-2,
    slope_atol: float = 1.0e-8,
    max_samples: int = 0,
) -> dict[str, object]:
    """Check dominant-growth eigenbranch locality for VMEC/Boozer samples.

    The optimizer-facing transport residual can only use the implicit
    dominant-eigenvalue gradient when the same eigenbranch is locally selected.
    This report evaluates the exact SPECTRAX-GK linear operator matrix at the
    base, plus, and minus VMEC final states for each configured
    surface/alpha/``k_y`` sample, then delegates branch classification to
    :func:`dominant_eigenvalue_branch_locality_report`.
    """

    _pin_current_optional_backend_paths()
    cfg = config or VMECJAXTransportObjectiveConfig(kind="growth")
    step_f = float(step)
    if step_f <= 0.0:
        raise ValueError("step must be positive")
    max_samples_int = int(max_samples)
    if max_samples_int < 0:
        raise ValueError("max_samples must be non-negative")

    samples = cfg.sample_set
    grid_options = _static_grid_options_from_ky_values(
        samples.ky_values,
        min_ny=int(cfg.ny),
    )
    selected_indices = cast(tuple[int, ...], grid_options["selected_ky_indices"])
    rows: list[dict[str, object]] = []
    total_sample_count = int(
        len(samples.surfaces) * len(samples.alphas) * len(samples.ky_values)
    )

    def geom_for(state: Any, *, torflux: float, alpha: float) -> Any:
        return flux_tube_geometry_from_vmec_boozer_state(
            state,
            static,
            indata,
            wout_reference,
            torflux=float(torflux),
            alpha=float(alpha),
            ntheta=int(cfg.ntheta),
            mboz=int(cfg.mboz),
            nboz=int(cfg.nboz),
            reference_length=cfg.reference_length,
            reference_b=cfg.reference_b,
            validate_finite=bool(cfg.validate_finite),
        )

    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            for ky_value, selected_ky_index in zip(
                samples.ky_values, selected_indices, strict=True
            ):
                metadata = {
                    "surface": float(torflux),
                    "alpha": float(alpha),
                    "ky": float(ky_value),
                    "selected_ky_index": int(selected_ky_index),
                }
                try:
                    base_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(
                            base_state, torflux=float(torflux), alpha=float(alpha)
                        ),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    plus_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(
                            plus_state, torflux=float(torflux), alpha=float(alpha)
                        ),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    minus_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(
                            minus_state, torflux=float(torflux), alpha=float(alpha)
                        ),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    branch = dominant_eigenvalue_branch_locality_report(
                        base_matrix,
                        plus_matrix,
                        minus_matrix,
                        step=step_f,
                        gap_floor=float(gap_floor),
                        slope_rtol=float(slope_rtol),
                        slope_atol=float(slope_atol),
                    )
                    rows.append(
                        {
                            **metadata,
                            "passed": bool(branch["passed"]),
                            "classification": str(branch["classification"]),
                            "branch_locality": branch,
                        }
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - exercised by optional backends.
                    rows.append(
                        {
                            **metadata,
                            "passed": False,
                            "classification": "branch_locality_evaluation_error",
                            "error": str(exc),
                        }
                    )
                if max_samples_int > 0 and len(rows) >= max_samples_int:
                    break
            if max_samples_int > 0 and len(rows) >= max_samples_int:
                break
        if max_samples_int > 0 and len(rows) >= max_samples_int:
            break

    finite = bool(rows and all("error" not in row for row in rows))
    passed = bool(
        finite
        and len(rows) == total_sample_count
        and all(bool(row["passed"]) for row in rows)
    )
    blockers: list[str] = []
    if not rows:
        blockers.append("no_branch_locality_samples")
    if len(rows) < total_sample_count:
        blockers.append("branch_locality_sample_set_truncated")
    if any(
        str(row.get("classification")) == "branch_locality_evaluation_error"
        for row in rows
    ):
        blockers.append("branch_locality_evaluation_error")
    if any(not bool(row.get("passed", False)) for row in rows):
        blockers.append("branch_locality_mismatch_or_underisolated")
    classifications = sorted({str(row.get("classification")) for row in rows})
    return {
        "kind": "vmec_jax_transport_growth_branch_locality_report",
        "claim_scope": (
            "VMEC/Boozer final-state perturbation -> SPECTRAX-GK linear operator "
            "dominant-growth branch locality; not a full transport-gradient promotion by itself"
        ),
        "passed": passed,
        "finite": finite,
        "classification": (
            "all_samples_dominant_growth_branch_locally_consistent"
            if passed
            else "growth_branch_locality_failed_or_incomplete"
        ),
        "step": step_f,
        "gap_floor": float(gap_floor),
        "slope_rtol": float(slope_rtol),
        "slope_atol": float(slope_atol),
        "sample_count": total_sample_count,
        "evaluated_sample_count": len(rows),
        "truncated": bool(len(rows) < total_sample_count),
        "classifications": classifications,
        "blockers": sorted(set(blockers)),
        "sample_set": samples.to_dict(),
        "spectrax_config": {
            "ntheta": int(cfg.ntheta),
            "mboz": int(cfg.mboz),
            "nboz": int(cfg.nboz),
            "n_laguerre": int(cfg.n_laguerre),
            "n_hermite": int(cfg.n_hermite),
            "nx": int(cfg.nx),
            "ny": int(grid_options["ny"]),
            "ly": float(grid_options["ly"]),
        },
        "rows": rows,
        "next_action": (
            "growth-branch locality is admissible for these samples"
            if passed
            else (
                "keep VMEC/SPECTRAX transport-gradient optimization fail-closed; "
                "reduce finite-difference steps, regularize branch selection, or "
                "use explicit branch tracking before promotion"
            )
        ),
    }


__all__ = ["vmec_jax_transport_growth_branch_locality_report_from_states"]
