"""File readers for nonlinear transport-window convergence metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np

from spectraxgk.validation.quasilinear.window_config import (
    NonlinearWindowConvergenceConfig,
)
from spectraxgk.validation.quasilinear.window_statistics import (
    nonlinear_window_convergence_report,
)


def _resolve_summary_artifact(summary_path: Path, source: object) -> Path:
    diag_path = Path(str(source))
    if diag_path.is_absolute():
        return diag_path
    candidates = (
        (summary_path.parent / diag_path).resolve(),
        (summary_path.parent.parent / diag_path).resolve(),
        (Path.cwd() / diag_path).resolve(),
    )
    return next(
        (candidate for candidate in candidates if candidate.exists()), candidates[0]
    )


def nonlinear_window_convergence_from_csv(
    csv_path: str | Path,
    *,
    time_column: str = "t",
    value_column: str = "heat_flux",
    case: str | None = None,
    config: NonlinearWindowConvergenceConfig | None = None,
    summary_artifact: str | None = None,
) -> dict[str, Any]:
    """Build a convergence report from a diagnostics CSV."""

    path = Path(csv_path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.shape == ():
        data = np.asarray([data], dtype=data.dtype)
    names = set(data.dtype.names or ())
    if time_column not in names:
        raise ValueError(f"{path} is missing time column '{time_column}'")
    if value_column not in names:
        raise ValueError(f"{path} is missing observable column '{value_column}'")
    return nonlinear_window_convergence_report(
        np.asarray(data[time_column], dtype=float),
        np.asarray(data[value_column], dtype=float),
        case=str(case or path.stem),
        observable=str(value_column),
        source_artifact=str(path),
        summary_artifact=summary_artifact,
        config=config,
    )


def nonlinear_window_convergence_from_summary(
    summary_json: str | Path,
    *,
    diagnostics_source: str = "spectrax",
    time_column: str = "t",
    value_column: str = "heat_flux",
    case: str | None = None,
    config: NonlinearWindowConvergenceConfig | None = None,
) -> dict[str, Any]:
    """Build a convergence report from a window summary and diagnostics CSV."""

    summary_path = Path(summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get(diagnostics_source)
    if source is None:
        raise ValueError(
            f"summary does not contain diagnostics source '{diagnostics_source}'"
        )
    cfg = config or NonlinearWindowConvergenceConfig(
        tmin=summary.get("tmin"),
        tmax=summary.get("tmax"),
    )
    diag_path = _resolve_summary_artifact(summary_path, source)
    if diag_path.suffix.lower() != ".csv":
        raise NotImplementedError(
            "nonlinear window convergence currently reads diagnostics CSV files"
        )
    return nonlinear_window_convergence_from_csv(
        diag_path,
        time_column=time_column,
        value_column=value_column,
        case=str(case or summary.get("case", summary_path.stem)),
        config=cfg,
        summary_artifact=str(summary_path),
    )


__all__ = [
    "nonlinear_window_convergence_from_csv",
    "nonlinear_window_convergence_from_summary",
]
