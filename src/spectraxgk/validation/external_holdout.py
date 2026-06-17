"""Planning utilities for external-VMEC nonlinear quasilinear holdouts.

The functions in this module do not promote a quasilinear absolute-flux model
and do not run simulations.  They turn the tracked holdout-gap metadata and a
linear candidate screen into a reproducible launch/runbook contract for the
next expensive nonlinear validation campaign.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class ExternalHoldoutScreenRow:
    """One row from an external-VMEC linear candidate screen."""

    case: str
    vmec_file: str
    returncode: int
    best_ky: float | None
    best_gamma: float | None
    best_omega: float | None
    log: str = ""

    @property
    def family(self) -> str:
        """Geometry family inferred from case and source path."""

        return external_vmec_family(self.case, self.vmec_file)

    @property
    def unstable(self) -> bool:
        """Whether the screen row is finite, successful, and linearly unstable."""

        return (
            self.returncode == 0
            and self.best_ky is not None
            and self.best_gamma is not None
            and self.best_gamma > 0.0
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        payload = asdict(self)
        payload["family"] = self.family
        payload["unstable"] = self.unstable
        return payload


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def external_vmec_family(case: str, source: str = "") -> str:
    """Return a stable family label for an external-VMEC candidate."""

    text = f"{case} {source}".lower()
    if "updown" in text or "up-down" in text or "up_down" in text:
        return "updown_asym_external_vmec"
    if "itermodel" in text:
        return "itermodel_external_vmec"
    if "li383" in text:
        return "li383_external_vmec"
    if "qi_stel" in text or "quasi-isodynamic" in text or "nfp3_qi" in text:
        return "qi_external_vmec"
    if "qa" in text and ("landremanpaul" in text or "quasi-axisymmetric" in text):
        return "qa_external_vmec"
    if "dshape" in text or "d-shaped" in text or "d_shaped" in text:
        return "dshape_external_vmec"
    if "circular" in text:
        return "circular_external_vmec"
    if "cth" in text:
        return "cth_like_external_vmec"
    if "qh" in text or "nfp4" in text:
        return "qh_external_vmec"
    if "basic_non_stellsym" in text or "non_stellsym" in text:
        return "non_stellsym_external_vmec"
    if "purely_toroidal" in text:
        return "purely_toroidal_external_vmec"
    if "solovev" in text:
        return "solovev_external_vmec"
    if "shaped_tokamak" in text:
        return "shaped_tokamak_external_vmec"
    return "external_vmec"


def read_external_holdout_screen(path: str | Path) -> list[ExternalHoldoutScreenRow]:
    """Load a candidate-screen CSV produced by the linear external-VMEC sweep."""

    rows: list[ExternalHoldoutScreenRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ExternalHoldoutScreenRow(
                    case=str(row.get("case", "")).strip(),
                    vmec_file=str(row.get("vmec_file", "")).strip(),
                    returncode=int(str(row.get("returncode", "1")).strip() or "1"),
                    best_ky=_finite_float(row.get("best_ky")),
                    best_gamma=_finite_float(row.get("best_gamma")),
                    best_omega=_finite_float(row.get("best_omega")),
                    log=str(row.get("log", "")).strip(),
                )
            )
    return rows


def _families_from_rows(rows: Iterable[dict[str, Any]]) -> set[str]:
    families: set[str] = set()
    for row in rows:
        family = str(row.get("geometry", ""))
        if "external_vmec" in family:
            families.add(family)
    return families


def _failed_external_families(gap_report: dict[str, Any]) -> set[str]:
    """Return external-VMEC families with tracked failed convergence gates."""

    families: set[str] = set()
    excluded = gap_report.get("excluded_candidates", [])
    if not isinstance(excluded, list):
        return families
    for row in excluded:
        if not isinstance(row, dict):
            continue
        family = str(row.get("geometry", ""))
        if "external_vmec" not in family:
            continue
        status = str(row.get("status", ""))
        failed = row.get("gate_passed") is False or status == "excluded_failed_external_gate"
        if failed:
            families.add(family)
    return families


def _passed_training_audit_families(gap_report: dict[str, Any]) -> set[str]:
    """Return represented families with a passed same-family audit gate."""

    families: set[str] = set()
    excluded = gap_report.get("excluded_candidates", [])
    if not isinstance(excluded, list):
        return families
    for row in excluded:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")) != "excluded_same_family_training_audit":
            continue
        family = str(row.get("geometry", ""))
        if "external_vmec" in family and row.get("gate_passed") is True:
            families.add(family)
    return families


def _first_nearest_gap(gap_report: dict[str, Any]) -> dict[str, Any]:
    needed = gap_report.get("next_actual_nonlinear_holdout_needed", {})
    if not isinstance(needed, dict):
        return {}
    nearest = needed.get("nearest_tracked_gap", {})
    return nearest if isinstance(nearest, dict) else {}


def _preferred_family(gap_report: dict[str, Any]) -> str | None:
    needed = gap_report.get("next_actual_nonlinear_holdout_needed", {})
    if not isinstance(needed, dict):
        return None
    family = needed.get("preferred_family")
    return str(family) if family else None


def _tracked_horizon(case: str) -> float | None:
    match = re.search(r"\bt(\d+(?:p\d+)?)\b", str(case))
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def _recommended_horizons(nearest_gap: dict[str, Any]) -> list[float]:
    horizon = _tracked_horizon(str(nearest_gap.get("case", "")))
    if horizon is None:
        return [150.0, 250.0, 350.0]
    return [horizon, horizon + 100.0, horizon + 200.0]


def _validated_horizons(horizons: Iterable[float]) -> list[float]:
    values = [float(value) for value in horizons]
    if not values:
        raise ValueError("at least one nonlinear holdout horizon is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("nonlinear holdout horizons must be finite and positive")
    if values != sorted(values):
        raise ValueError("nonlinear holdout horizons must be sorted increasingly")
    return values


def _candidate_status(
    row: ExternalHoldoutScreenRow,
    *,
    preferred_family: str | None,
    represented_families: set[str],
    failed_external_families: set[str],
    passed_training_audit_families: set[str],
    min_launch_gamma: float,
    allow_modified_protocol_families: set[str],
    modified_protocol_note: str,
) -> tuple[str, float, str]:
    if not row.unstable:
        return ("screen_rejected_stable_or_failed", 9.0, "screen row did not finish with positive growth")
    if float(row.best_gamma or 0.0) < float(min_launch_gamma):
        return (
            "screen_marginal_needs_linear_refinement",
            7.0,
            "positive growth is below the nonlinear-launch threshold; refine the linear branch before launching a transport holdout",
        )
    if (
        row.family in failed_external_families
        and row.family in allow_modified_protocol_families
        and row.family not in passed_training_audit_families
    ):
        return (
            "modified_protocol_failed_family_candidate",
            1.5,
            "this external-VMEC family has a tracked failed convergence gate, but an explicit modified-protocol rerun was requested; "
            f"protocol change: {modified_protocol_note}",
        )
    if row.family in failed_external_families and row.family != preferred_family:
        return (
            "recent_family_failed_external_gate",
            6.0,
            "this external-VMEC family has a tracked failed convergence gate; rerun only with a modified higher-resolution protocol",
        )
    if preferred_family and row.family == preferred_family:
        if row.family in represented_families:
            if row.family in passed_training_audit_families:
                return (
                    "preferred_family_audit_already_passed",
                    5.0,
                    "same-family audit already passed; relaunch only with a different independent geometry or a materially changed protocol",
                )
            return (
                "preferred_family_already_represented",
                3.0,
                "preferred family is already used in the current calibration portfolio; use only with an independent split/window",
            )
        return ("preferred_family_new_holdout", 0.0, "matches preferred gap-report family")
    if row.family not in represented_families:
        return ("new_family_holdout_candidate", 2.0, "unstable electrostatic-compatible VMEC family not yet represented")
    return ("represented_family_audit_candidate", 4.0, "family already represented; useful as an audit, not first holdout leverage")


def build_external_holdout_runbook(
    *,
    gap_report: dict[str, Any],
    screen_rows: Iterable[ExternalHoldoutScreenRow],
    out_dir: str = "tools_out/external_vmec_holdouts",
    grids: tuple[str, ...] = ("n48:48:48:32:32", "n64:64:64:40:40"),
    dt: float = 0.05,
    min_launch_gamma: float = 0.02,
    max_candidates: int = 6,
    horizons: tuple[float, ...] | None = None,
    allow_modified_protocol_families: tuple[str, ...] = (),
    modified_protocol_note: str = "",
) -> dict[str, Any]:
    """Build a JSON-ready runbook for the next nonlinear holdout campaign.

    The ranking first preserves the gap-report priority, then prefers unstable
    families absent from the current train/holdout portfolio.  The runbook is a
    launch plan only; promotion still requires the generated nonlinear traces to
    pass grid/window convergence and then enter calibration metadata as
    ``split=holdout``.
    """

    preferred = _preferred_family(gap_report)
    admitted = gap_report.get("admitted_holdouts", [])
    training = gap_report.get("training_references", [])
    represented = _families_from_rows([*admitted, *training])
    failed_external = _failed_external_families(gap_report)
    passed_training_audits = _passed_training_audit_families(gap_report)
    nearest_gap = _first_nearest_gap(gap_report)
    recommended_horizons = _validated_horizons(horizons) if horizons is not None else _recommended_horizons(nearest_gap)
    grid_args = " ".join(f"--grid {grid}" for grid in grids)
    allowed_modified = {str(family).strip() for family in allow_modified_protocol_families if str(family).strip()}
    if allowed_modified and not str(modified_protocol_note).strip():
        raise ValueError("modified_protocol_note is required when allowing failed-family modified-protocol reruns")

    ranked: list[dict[str, Any]] = []
    for row in screen_rows:
        status, priority, reason = _candidate_status(
            row,
            preferred_family=preferred,
            represented_families=represented,
            failed_external_families=failed_external,
            passed_training_audit_families=passed_training_audits,
            min_launch_gamma=float(min_launch_gamma),
            allow_modified_protocol_families=allowed_modified,
            modified_protocol_note=str(modified_protocol_note).strip(),
        )
        gamma_key = -(row.best_gamma if row.best_gamma is not None else -math.inf)
        ky_key = -(row.best_ky if row.best_ky is not None else -math.inf)
        ranked.append(
            {
                **row.to_dict(),
                "status": status,
                "priority": priority,
                "reason": reason,
                "_sort_key": [priority, gamma_key, ky_key, row.case],
            }
        )
    ranked = sorted(ranked, key=lambda item: tuple(item["_sort_key"]))
    for idx, ranked_row in enumerate(ranked, start=1):
        ranked_row["rank"] = idx
        ranked_row.pop("_sort_key", None)
    selected_new = next(
        (
            row
            for row in ranked
            if row["status"]
            in {"preferred_family_new_holdout", "new_family_holdout_candidate", "modified_protocol_failed_family_candidate"}
        ),
        None,
    )
    selected_preferred_audit = next(
        (row for row in ranked if row["status"] == "preferred_family_already_represented"),
        None,
    )

    launch_commands: list[str] = []
    if selected_new is not None:
        launch_commands.append(
            "python tools/write_external_vmec_holdout_configs.py "
            f"--case {str(selected_new['case']).replace('_nc', '')}_holdout "
            f"--vmec-file {selected_new['vmec_file']} "
            f"--out-dir {out_dir}/{str(selected_new['case']).replace('_nc', '')} "
            f"--ky {float(selected_new['best_ky']):.12g} "
            f"--dt {float(dt):.12g} "
            f"--horizons {','.join(f'{value:.12g}' for value in recommended_horizons)} "
            f"{grid_args}"
        )
    if selected_preferred_audit is not None:
        launch_commands.append(
            "python tools/write_external_vmec_holdout_configs.py "
            f"--case {str(selected_preferred_audit['case']).replace('_nc', '')}_independent_audit "
            f"--vmec-file {selected_preferred_audit['vmec_file']} "
            f"--out-dir {out_dir}/{str(selected_preferred_audit['case']).replace('_nc', '')}_audit "
            f"--ky {float(selected_preferred_audit['best_ky']):.12g} "
            f"--dt {float(dt):.12g} "
            f"--horizons {','.join(f'{value:.12g}' for value in recommended_horizons)} "
            f"{grid_args}"
        )

    return {
        "kind": "external_vmec_holdout_runbook",
        "claim_level": "nonlinear_holdout_launch_plan_not_transport_validation",
        "passed": bool(selected_new is not None or selected_preferred_audit is not None),
        "absolute_flux_promoted": False,
        "preferred_family": preferred,
        "represented_families": sorted(represented),
        "failed_external_families": sorted(failed_external),
        "allow_modified_protocol_families": sorted(allowed_modified),
        "modified_protocol_note": str(modified_protocol_note).strip(),
        "nearest_tracked_gap": nearest_gap,
        "recommended_horizons": recommended_horizons,
        "recommended_grids": list(grids),
        "dt": float(dt),
        "min_launch_gamma": float(min_launch_gamma),
        "selected_new_family_candidate": selected_new,
        "selected_preferred_family_audit": selected_preferred_audit,
        "ranked_candidates": ranked[: int(max_candidates)],
        "launch_commands": launch_commands,
        "acceptance_gate": {
            "required_split": "holdout",
            "required_claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "requires_grid_window_convergence": True,
            "requires_post_transient_window": True,
            "requires_independent_from_training_reference": True,
            "requires_explicit_modified_protocol_note_for_failed_families": True,
            "minimum_screen_growth_rate_for_launch": float(min_launch_gamma),
        },
        "notes": (
            "Run the selected configurations on the large-run host, build a convergence gate with "
            "tools/plot_external_vmec_nonlinear_convergence_gate.py, and admit the resulting transport "
            "window to quasilinear calibration only if the gate passes and the split is holdout."
        ),
    }
