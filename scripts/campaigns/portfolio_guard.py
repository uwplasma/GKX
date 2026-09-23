"""Promotion guards for real VMEC/Boozer reduced objective-portfolio artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

from gkx.objectives.portfolio import (
    PortfolioReduction,
    aggregate_objective_portfolio,
    validate_objective_portfolio_contract,
)

_GROWTH_OBJECTIVE_NAMES = frozenset(("gamma", "growth", "growth_rate"))
_QUASILINEAR_OBJECTIVE_NAMES = frozenset(
    (
        "quasilinear_flux",
        "quasilinear_heat_flux",
        "mixing_length_heat_flux_proxy",
        "linear_heat_flux_weight",
    )
)


@dataclass(frozen=True)
class ReducedPortfolioArtifactGuardConfig:
    """Requirements for promoting real VMEC/Boozer reduced-portfolio rows."""

    min_alphas: int = 2
    min_ky: int = 2
    min_objectives: int = 1
    min_boozer_mode: int = 21
    require_growth_objective: bool = True
    require_quasilinear_objective: bool = True
    require_vmec_paths: bool = True
    value_rtol: float = 1.0e-8
    value_atol: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in ("min_alphas", "min_ky", "min_objectives", "min_boozer_mode"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if float(self.value_rtol) < 0.0 or float(self.value_atol) < 0.0:
            raise ValueError("value tolerances must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(frozen=True)
class _PortfolioArtifactInputs:
    sample_rows: np.ndarray
    sample_weights: np.ndarray
    surfaces: list[str]
    alphas: list[float]
    kys: list[float]
    samples: list[dict[str, Any]]
    full_table: np.ndarray
    objective_names: list[str]
    reduction: PortfolioReduction


def _artifact_text(payload: dict[str, Any], *keys: str) -> str:
    return " ".join(str(payload.get(key, "")) for key in keys).lower()


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nested(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(array.size > 0 and np.all(np.isfinite(array)))


def _sample_key(sample: dict[str, Any]) -> tuple[str, float, float]:
    surface = sample.get("surface_index", "__mid_surface__")
    if surface is None and sample.get("torflux") is not None:
        surface_key = f"torflux={_as_finite_float(sample.get('torflux'), name='sample torflux'):.16g}"
    elif surface is None and sample.get("surface") is not None:
        surface_key = f"surface={_as_finite_float(sample.get('surface'), name='sample surface'):.16g}"
    else:
        surface_key = "__mid_surface__" if surface is None else str(surface)
    alpha = _as_finite_float(sample.get("alpha"), name="sample alpha")
    ky_value = sample.get("ky", sample.get("ky_index", sample.get("selected_ky_index")))
    ky = _as_finite_float(ky_value, name="sample ky")
    return surface_key, alpha, ky


def _axis_indices(
    samples: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[float]]:
    surfaces = sorted({_sample_key(sample)[0] for sample in samples})
    alphas = sorted({_sample_key(sample)[1] for sample in samples})
    kys = sorted({_sample_key(sample)[2] for sample in samples})
    return surfaces, alphas, kys


def _artifact_sample_values(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    samples_raw = payload.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError("payload must contain a non-empty samples list")
    samples = [item for item in samples_raw if isinstance(item, dict)]
    if len(samples) != len(samples_raw):
        raise ValueError("all samples must be dictionaries")
    values = np.asarray(payload.get("base_sample_values"), dtype=float)
    if values.ndim != 1 or int(values.shape[0]) != len(samples):
        raise ValueError("base_sample_values must be a length-n_samples vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("base_sample_values must be finite")
    return samples, values


def _artifact_sample_value_tensor(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[float], list[float]]:
    samples, values = _artifact_sample_values(payload)
    surfaces, alphas, kys = _axis_indices(samples)
    shape = (len(surfaces), len(alphas), len(kys))
    table = np.full(shape, np.nan, dtype=float)
    weights = np.full(shape, np.nan, dtype=float)
    seen: set[tuple[int, int, int]] = set()
    surface_index = {value: index for index, value in enumerate(surfaces)}
    alpha_index = {value: index for index, value in enumerate(alphas)}
    ky_index = {value: index for index, value in enumerate(kys)}
    for sample, value in zip(samples, values, strict=True):
        surface, alpha, ky = _sample_key(sample)
        idx = (surface_index[surface], alpha_index[alpha], ky_index[ky])
        if idx in seen:
            raise ValueError("samples must not contain duplicate surface/alpha/ky rows")
        seen.add(idx)
        table[idx] = value
        weights[idx] = _as_finite_float(sample.get("weight", 1.0), name="sample weight")
    if len(seen) != int(np.prod(shape)) or not np.all(np.isfinite(table)):
        raise ValueError(
            "samples must form a complete rectangular surface/alpha/ky table"
        )
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError(
            "sample weights must be finite, non-negative, and have positive sum"
        )
    return table[..., None], weights, surfaces, alphas, kys


def _artifact_full_objective_table(
    payload: dict[str, Any], *, n_samples: int
) -> np.ndarray:
    table = np.asarray(payload.get("base_objective_table"), dtype=float)
    if table.ndim != 2:
        raise ValueError("base_objective_table must be a two-dimensional array")
    if int(table.shape[0]) != int(n_samples):
        raise ValueError("base_objective_table row count must match samples")
    if not np.all(np.isfinite(table)):
        raise ValueError("base_objective_table must be finite")
    return table


def _artifact_objective_name_gate(
    objective_names: list[str],
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    lowered = {name.lower() for name in objective_names}
    has_growth = bool(lowered & _GROWTH_OBJECTIVE_NAMES)
    has_quasilinear = bool(lowered & _QUASILINEAR_OBJECTIVE_NAMES)
    passed = bool(
        len(objective_names) >= int(config.min_objectives)
        and (has_growth or not config.require_growth_objective)
        and (has_quasilinear or not config.require_quasilinear_objective)
    )
    return {
        "passed": passed,
        "objective_names": objective_names,
        "n_objectives": len(objective_names),
        "min_objectives": int(config.min_objectives),
        "has_growth_objective": has_growth,
        "has_quasilinear_objective": has_quasilinear,
        "requires_growth_objective": bool(config.require_growth_objective),
        "requires_quasilinear_objective": bool(config.require_quasilinear_objective),
    }


def _artifact_provenance_gate(
    payload: dict[str, Any],
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    options_raw = payload.get("options")
    options: dict[str, Any] = options_raw if isinstance(options_raw, dict) else {}
    mboz = int(payload.get("mboz") or options.get("mboz") or 0)
    nboz = int(payload.get("nboz") or options.get("nboz") or 0)
    text = _artifact_text(
        payload, "kind", "artifact_kind", "source_scope", "claim_scope", "builder"
    )
    has_vmec_boozer_scope = "vmec_boozer" in text or "vmec/boozer" in text
    input_path = str(payload.get("input_path", ""))
    wout_path = str(payload.get("wout_path", ""))
    has_paths = bool(input_path and wout_path)
    mode_gate = bool(
        mboz >= int(config.min_boozer_mode) and nboz >= int(config.min_boozer_mode)
    )
    return {
        "passed": bool(
            has_vmec_boozer_scope
            and mode_gate
            and (has_paths or not config.require_vmec_paths)
        ),
        "has_vmec_boozer_scope": has_vmec_boozer_scope,
        "requires_vmec_paths": bool(config.require_vmec_paths),
        "has_input_and_wout_paths": has_paths,
        "input_path": input_path,
        "wout_path": wout_path,
        "mboz": mboz,
        "nboz": nboz,
        "min_boozer_mode": int(config.min_boozer_mode),
    }


def _artifact_claim_scope_gate(payload: dict[str, Any]) -> dict[str, object]:
    objective = str(payload.get("objective", payload.get("objective_kind", ""))).lower()
    text = _artifact_text(payload, "claim_scope", "next_action")
    has_safe_disclaimer = any(
        phrase in text
        for phrase in (
            "not a nonlinear",
            "not a production nonlinear",
            "not an optimized-equilibrium nonlinear",
            "nonlinear transport optimization still requires",
        )
    )
    no_nonlinear_objective = objective not in {
        "nonlinear_heat_flux",
        "nonlinear_window_heat_flux_mean",
        "production_nonlinear_heat_flux",
    }
    no_trace = payload.get("nonlinear_trace") in (None, {}, [])
    return {
        "passed": bool(no_nonlinear_objective and no_trace and has_safe_disclaimer),
        "objective": objective,
        "nonlinear_trace_absent": no_trace,
        "has_nonproduction_disclaimer": has_safe_disclaimer,
    }


def _artifact_fd_gate(payload: dict[str, Any]) -> dict[str, object]:
    finite_fields = {
        name: _finite_nested(payload.get(name))
        for name in (
            "minus_sample_values",
            "base_sample_values",
            "plus_sample_values",
            "minus_objective_table",
            "base_objective_table",
            "plus_objective_table",
        )
    }
    scalar_fields = {
        name: bool(np.isfinite(float(payload.get(name, np.nan))))
        for name in (
            "base_value",
            "minus_value",
            "plus_value",
            "central_derivative",
            "response_abs",
            "curvature_ratio",
        )
    }
    passed = bool(
        payload.get("passed", False)
        and payload.get("finite_values", True)
        and payload.get("finite_difference_consistent", False)
        and payload.get("response_resolved", True)
        and all(finite_fields.values())
        and all(scalar_fields.values())
    )
    return {
        "passed": passed,
        "artifact_passed": bool(payload.get("passed", False)),
        "finite_difference_consistent": bool(
            payload.get("finite_difference_consistent", False)
        ),
        "response_resolved": bool(payload.get("response_resolved", True)),
        "finite_array_fields": finite_fields,
        "finite_scalar_fields": scalar_fields,
    }


def _gradient_artifact_gate(
    gradient_artifacts: list[dict[str, Any]],
) -> dict[str, object]:
    objective_names: set[str] = set()
    n_gates = 0
    n_passed = 0
    finite = True
    for artifact in gradient_artifacts:
        for gate in (
            artifact.get("objective_gates", []) if isinstance(artifact, dict) else []
        ):
            if not isinstance(gate, dict):
                finite = False
                continue
            n_gates += 1
            objective_names.add(str(gate.get("objective", "")).lower())
            n_passed += int(bool(gate.get("passed", False)))
            finite = bool(
                finite
                and all(
                    _finite_nested(gate.get(name))
                    for name in (
                        "implicit",
                        "finite_difference",
                        "abs_error",
                        "rel_error",
                    )
                )
            )
    has_growth = bool(objective_names & _GROWTH_OBJECTIVE_NAMES)
    has_quasilinear = bool(objective_names & _QUASILINEAR_OBJECTIVE_NAMES)
    passed = bool(
        n_gates > 0
        and n_passed == n_gates
        and finite
        and has_growth
        and has_quasilinear
    )
    return {
        "passed": passed,
        "n_gradient_artifacts": len(gradient_artifacts),
        "n_objective_gates": n_gates,
        "n_passed_objective_gates": n_passed,
        "finite_ad_fd_values": finite,
        "objective_names": sorted(objective_names),
        "has_growth_ad_fd_gate": has_growth,
        "has_quasilinear_ad_fd_gate": has_quasilinear,
    }


def _artifact_reduction(payload: dict[str, Any]) -> PortfolioReduction:
    reduction = str(payload.get("reduction", "weighted_mean"))
    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("artifact reduction must be weighted_mean, mean, or max")
    return cast(PortfolioReduction, reduction)


def _portfolio_artifact_inputs(
    payload: dict[str, Any],
) -> _PortfolioArtifactInputs:
    sample_rows, sample_weights, surfaces, alphas, kys = _artifact_sample_value_tensor(
        payload
    )
    samples, _values = _artifact_sample_values(payload)
    full_table = _artifact_full_objective_table(payload, n_samples=len(samples))
    objective_names = [str(item) for item in payload.get("objective_names", [])]
    return _PortfolioArtifactInputs(
        sample_rows=sample_rows,
        sample_weights=sample_weights,
        surfaces=surfaces,
        alphas=alphas,
        kys=kys,
        samples=samples,
        full_table=full_table,
        objective_names=objective_names,
        reduction=_artifact_reduction(payload),
    )


def _portfolio_coverage_gate(
    inputs: _PortfolioArtifactInputs,
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    return {
        "passed": bool(
            len(inputs.alphas) >= int(config.min_alphas)
            and len(inputs.kys) >= int(config.min_ky)
        ),
        "surface_labels": inputs.surfaces,
        "alphas": inputs.alphas,
        "ky_values": inputs.kys,
        "n_surfaces": len(inputs.surfaces),
        "n_alphas": len(inputs.alphas),
        "n_ky": len(inputs.kys),
        "n_samples": len(inputs.samples),
        "min_alphas": int(config.min_alphas),
        "min_ky": int(config.min_ky),
    }


def _portfolio_full_table_gate(
    inputs: _PortfolioArtifactInputs,
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    table = inputs.full_table
    finite = bool(np.all(np.isfinite(table)))
    return {
        "passed": bool(
            table.shape[0] == len(inputs.samples)
            and table.shape[1] >= int(config.min_objectives)
            and finite
        ),
        "shape": list(table.shape),
        "finite": finite,
    }


def _portfolio_reducer_gate(
    payload: dict[str, Any],
    inputs: _PortfolioArtifactInputs,
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    """Recompute and validate the artifact's scalar portfolio reduction."""

    sample_weights = (
        inputs.sample_weights if inputs.reduction == "weighted_mean" else None
    )
    reduced_value = float(
        aggregate_objective_portfolio(
            inputs.sample_rows,
            sample_weights=sample_weights,
            reduction=inputs.reduction,
        )
    )
    contract = validate_objective_portfolio_contract(
        inputs.sample_rows,
        sample_weights=sample_weights,
        reduction=inputs.reduction,
    )
    artifact_value = _as_finite_float(payload.get("base_value"), name="base_value")
    reducer_matches = bool(
        np.isclose(
            reduced_value,
            artifact_value,
            rtol=config.value_rtol,
            atol=config.value_atol,
        )
    )
    return {
        "passed": reducer_matches,
        "reduction": inputs.reduction,
        "reduced_base_value": reduced_value,
        "artifact_base_value": artifact_value,
        "abs_error": float(abs(reduced_value - artifact_value)),
        "rtol": float(config.value_rtol),
        "atol": float(config.value_atol),
        "contract": contract.to_dict(),
    }


def reduced_portfolio_artifact_guard_report(
    row_artifact: dict[str, Any],
    *,
    gradient_artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    config: ReducedPortfolioArtifactGuardConfig | None = None,
) -> dict[str, object]:
    """Gate persisted portfolios on provenance, coverage, reduction, and AD/FD."""

    cfg = config or ReducedPortfolioArtifactGuardConfig()
    inputs = _portfolio_artifact_inputs(row_artifact)
    objective_name_gate = _artifact_objective_name_gate(
        inputs.objective_names, config=cfg
    )
    coverage_gate = _portfolio_coverage_gate(inputs, config=cfg)
    full_table_gate = _portfolio_full_table_gate(inputs, config=cfg)
    reducer_gate = _portfolio_reducer_gate(row_artifact, inputs, config=cfg)
    provenance_gate = _artifact_provenance_gate(row_artifact, config=cfg)
    claim_scope_gate = _artifact_claim_scope_gate(row_artifact)
    fd_gate = _artifact_fd_gate(row_artifact)
    ad_fd_gate = _gradient_artifact_gate(list(gradient_artifacts))
    passed = all(
        gate["passed"]
        for gate in (
            provenance_gate,
            coverage_gate,
            full_table_gate,
            objective_name_gate,
            reducer_gate,
            fd_gate,
            ad_fd_gate,
            claim_scope_gate,
        )
    )
    return {
        "kind": "vmec_boozer_reduced_portfolio_artifact_guard",
        "passed": passed,
        "claim_scope": (
            "artifact-level guard for real VMEC/Boozer reduced growth/QL portfolio rows; "
            "not a production nonlinear turbulent-transport optimization claim"
        ),
        "config": cfg.to_dict(),
        "provenance_gate": provenance_gate,
        "coverage_gate": coverage_gate,
        "full_objective_table_gate": full_table_gate,
        "objective_name_gate": objective_name_gate,
        "portfolio_reducer_gate": reducer_gate,
        "finite_difference_gate": fd_gate,
        "ad_fd_gradient_gate": ad_fd_gate,
        "claim_scope_gate": claim_scope_gate,
        "next_action": (
            "Use this guard to admit reduced VMEC/Boozer growth/QL portfolio artifacts. "
            "Production nonlinear claims still require held-out optimized-equilibrium "
            "nonlinear-window ensembles and transport audits."
        ),
    }


__all__ = [
    "ReducedPortfolioArtifactGuardConfig",
    "reduced_portfolio_artifact_guard_report",
]
