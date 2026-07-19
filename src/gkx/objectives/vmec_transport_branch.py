"""Branch-locality gates for VMEC-JAX transport objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from gkx.geometry.vmec_boozer_core import (
    flux_tube_geometry_from_vmec_boozer_state,
)
from gkx.objectives.core import solver_linear_operator_matrix_from_geometry
from gkx.objectives.eigen import dominant_eigenvalue_branch_locality_report
from gkx.objectives.vmec_transport import (
    VMEXTransportObjectiveConfig,
    _pin_current_optional_backend_paths,
)
from gkx.objectives.vmec_transport import (
    _static_grid_options_from_ky_values,
)


@dataclass(frozen=True)
class _BranchStates:
    base: Any
    plus: Any
    minus: Any


@dataclass(frozen=True)
class _BranchLocalitySetup:
    cfg: VMEXTransportObjectiveConfig
    step: float
    max_samples: int
    grid_options: dict[str, Any]
    selected_ky_indices: tuple[int, ...]
    total_sample_count: int


def _branch_locality_setup(
    config: VMEXTransportObjectiveConfig | None,
    *,
    step: float,
    max_samples: int,
) -> _BranchLocalitySetup:
    _pin_current_optional_backend_paths()
    cfg = config or VMEXTransportObjectiveConfig(kind="growth")
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
    total_sample_count = int(
        len(samples.surfaces) * len(samples.alphas) * len(samples.ky_values)
    )
    return _BranchLocalitySetup(
        cfg=cfg,
        step=step_f,
        max_samples=max_samples_int,
        grid_options=grid_options,
        selected_ky_indices=selected_indices,
        total_sample_count=total_sample_count,
    )


def _geometry_for_state(
    state: Any,
    *,
    setup: _BranchLocalitySetup,
    static: Any,
    indata: Any,
    wout_reference: Any,
    torflux: float,
    alpha: float,
) -> Any:
    cfg = setup.cfg
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


def _linear_matrix_for_state(
    state: Any,
    *,
    setup: _BranchLocalitySetup,
    static: Any,
    indata: Any,
    wout_reference: Any,
    torflux: float,
    alpha: float,
    selected_ky_index: int,
) -> Any:
    cfg = setup.cfg
    return solver_linear_operator_matrix_from_geometry(
        _geometry_for_state(
            state,
            setup=setup,
            static=static,
            indata=indata,
            wout_reference=wout_reference,
            torflux=torflux,
            alpha=alpha,
        ),
        selected_ky_index=int(selected_ky_index),
        n_laguerre=int(cfg.n_laguerre),
        n_hermite=int(cfg.n_hermite),
        nx=int(cfg.nx),
        ny=int(setup.grid_options["ny"]),
        ly=float(setup.grid_options["ly"]),
    )


def _sample_metadata(
    *,
    torflux: float,
    alpha: float,
    ky_value: float,
    selected_ky_index: int,
) -> dict[str, Any]:
    return {
        "surface": float(torflux),
        "alpha": float(alpha),
        "ky": float(ky_value),
        "selected_ky_index": int(selected_ky_index),
    }


def _branch_locality_row(
    *,
    states: _BranchStates,
    setup: _BranchLocalitySetup,
    static: Any,
    indata: Any,
    wout_reference: Any,
    metadata: dict[str, Any],
    gap_floor: float,
    slope_rtol: float,
    slope_atol: float,
) -> dict[str, Any]:
    torflux = float(metadata["surface"])
    alpha = float(metadata["alpha"])
    selected_ky_index = int(metadata["selected_ky_index"])
    try:
        base_matrix = _linear_matrix_for_state(
            states.base,
            setup=setup,
            static=static,
            indata=indata,
            wout_reference=wout_reference,
            torflux=torflux,
            alpha=alpha,
            selected_ky_index=selected_ky_index,
        )
        plus_matrix = _linear_matrix_for_state(
            states.plus,
            setup=setup,
            static=static,
            indata=indata,
            wout_reference=wout_reference,
            torflux=torflux,
            alpha=alpha,
            selected_ky_index=selected_ky_index,
        )
        minus_matrix = _linear_matrix_for_state(
            states.minus,
            setup=setup,
            static=static,
            indata=indata,
            wout_reference=wout_reference,
            torflux=torflux,
            alpha=alpha,
            selected_ky_index=selected_ky_index,
        )
        branch = dominant_eigenvalue_branch_locality_report(
            base_matrix,
            plus_matrix,
            minus_matrix,
            step=setup.step,
            gap_floor=float(gap_floor),
            slope_rtol=float(slope_rtol),
            slope_atol=float(slope_atol),
        )
        return {
            **metadata,
            "passed": bool(branch["passed"]),
            "classification": str(branch["classification"]),
            "branch_locality": branch,
        }
    except Exception as exc:  # pragma: no cover - exercised by optional backends.
        return {
            **metadata,
            "passed": False,
            "classification": "branch_locality_evaluation_error",
            "error": str(exc),
        }


def _branch_locality_rows(
    *,
    states: _BranchStates,
    setup: _BranchLocalitySetup,
    static: Any,
    indata: Any,
    wout_reference: Any,
    gap_floor: float,
    slope_rtol: float,
    slope_atol: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples = setup.cfg.sample_set
    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            for ky_value, selected_ky_index in zip(
                samples.ky_values,
                setup.selected_ky_indices,
                strict=True,
            ):
                rows.append(
                    _branch_locality_row(
                        states=states,
                        setup=setup,
                        static=static,
                        indata=indata,
                        wout_reference=wout_reference,
                        metadata=_sample_metadata(
                            torflux=float(torflux),
                            alpha=float(alpha),
                            ky_value=float(ky_value),
                            selected_ky_index=int(selected_ky_index),
                        ),
                        gap_floor=gap_floor,
                        slope_rtol=slope_rtol,
                        slope_atol=slope_atol,
                    )
                )
                if setup.max_samples > 0 and len(rows) >= setup.max_samples:
                    return rows
    return rows


def _branch_locality_blockers(
    rows: list[dict[str, Any]],
    *,
    total_sample_count: int,
) -> list[str]:
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
    return sorted(set(blockers))


def _branch_locality_passed(
    rows: list[dict[str, Any]],
    *,
    total_sample_count: int,
) -> bool:
    finite = bool(rows and all("error" not in row for row in rows))
    return bool(
        finite
        and len(rows) == total_sample_count
        and all(bool(row["passed"]) for row in rows)
    )


def _gkx_config_payload(setup: _BranchLocalitySetup) -> dict[str, object]:
    cfg = setup.cfg
    return {
        "ntheta": int(cfg.ntheta),
        "mboz": int(cfg.mboz),
        "nboz": int(cfg.nboz),
        "n_laguerre": int(cfg.n_laguerre),
        "n_hermite": int(cfg.n_hermite),
        "nx": int(cfg.nx),
        "ny": int(setup.grid_options["ny"]),
        "ly": float(setup.grid_options["ly"]),
    }


def _pack_branch_locality_report(
    *,
    setup: _BranchLocalitySetup,
    rows: list[dict[str, Any]],
    gap_floor: float,
    slope_rtol: float,
    slope_atol: float,
) -> dict[str, object]:
    finite = bool(rows and all("error" not in row for row in rows))
    passed = _branch_locality_passed(
        rows,
        total_sample_count=setup.total_sample_count,
    )
    classifications = sorted({str(row.get("classification")) for row in rows})
    return {
        "kind": "vmex_transport_growth_branch_locality_report",
        "claim_scope": (
            "VMEC/Boozer final-state perturbation -> GKX linear operator "
            "dominant-growth branch locality; not a full transport-gradient promotion by itself"
        ),
        "passed": passed,
        "finite": finite,
        "classification": (
            "all_samples_dominant_growth_branch_locally_consistent"
            if passed
            else "growth_branch_locality_failed_or_incomplete"
        ),
        "step": setup.step,
        "gap_floor": float(gap_floor),
        "slope_rtol": float(slope_rtol),
        "slope_atol": float(slope_atol),
        "sample_count": setup.total_sample_count,
        "evaluated_sample_count": len(rows),
        "truncated": bool(len(rows) < setup.total_sample_count),
        "classifications": classifications,
        "blockers": _branch_locality_blockers(
            rows,
            total_sample_count=setup.total_sample_count,
        ),
        "sample_set": setup.cfg.sample_set.to_dict(),
        "gkx_config": _gkx_config_payload(setup),
        "rows": rows,
        "next_action": (
            "growth-branch locality is admissible for these samples"
            if passed
            else (
                "keep VMEC/GKX transport-gradient optimization fail-closed; "
                "reduce finite-difference steps, regularize branch selection, or "
                "use explicit branch tracking before promotion"
            )
        ),
    }


def vmex_transport_growth_branch_locality_report_from_states(
    base_state: Any,
    plus_state: Any,
    minus_state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMEXTransportObjectiveConfig | None = None,
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
    This report evaluates the exact GKX linear operator matrix at the
    base, plus, and minus VMEC final states for each configured
    surface/alpha/``k_y`` sample, then delegates branch classification to
    :func:`dominant_eigenvalue_branch_locality_report`.
    """

    setup = _branch_locality_setup(config, step=step, max_samples=max_samples)
    rows = _branch_locality_rows(
        states=_BranchStates(base=base_state, plus=plus_state, minus=minus_state),
        setup=setup,
        static=static,
        indata=indata,
        wout_reference=wout_reference,
        gap_floor=gap_floor,
        slope_rtol=slope_rtol,
        slope_atol=slope_atol,
    )
    return _pack_branch_locality_report(
        setup=setup,
        rows=rows,
        gap_floor=gap_floor,
        slope_rtol=slope_rtol,
        slope_atol=slope_atol,
    )


__all__ = ["vmex_transport_growth_branch_locality_report_from_states"]
