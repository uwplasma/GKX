"""Deck-relative file overrides and field-line length checks for a runtime case."""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path
from typing import Any


def anchor_case_paths(case: Any, source_dir: str | None) -> Any:
    """Resolve relative file overrides against the deck directory."""

    if source_dir is None:
        return case

    def anchor(value: str | None) -> str | None:
        if not value:
            return value
        path = Path(os.path.expanduser(os.path.expandvars(value)))
        if "$" in str(path) or path.is_absolute():
            return value
        return str((Path(source_dir) / path).resolve())

    geometry, init = case.geometry, case.init
    return dataclasses.replace(
        case,
        geometry=dataclasses.replace(
            geometry,
            vmec_file=anchor(geometry.vmec_file),
            geometry_file=anchor(geometry.geometry_file),
        ),
        init=dataclasses.replace(init, init_file=anchor(init.init_file)),
    )


def field_line_errors(case: Any) -> list[str]:
    """Reject a VMEC deck whose ``nperiod`` disagrees with ``geometry.npol``."""

    npol, nperiod = case.geometry.npol, case.grid.nperiod
    if case.geometry.model != "vmec" or npol is None or nperiod is None:
        return []
    if 2 * nperiod - 1 == npol:
        return []
    return [
        f"grid.nperiod = {nperiod} is ignored for model = 'vmec'; the field-line "
        f"length is geometry.npol = {npol:g}: set geometry.npol and drop grid.nperiod"
    ]


def warn_short_field_line(case: Any) -> None:
    """Warn below npol = 3 (VMEC) or nperiod = 2 (analytic geometry)."""

    model, npol, nperiod = case.geometry.model, case.geometry.npol, case.grid.nperiod
    if model == "vmec":
        if npol is None and nperiod is not None:
            npol = 2 * nperiod - 1
        detail = None if npol is None or npol >= 3 else f"geometry.npol = {npol:g}"
    elif model != "imported-netcdf":
        detail = (
            None if nperiod is None or nperiod >= 2 else f"grid.nperiod = {nperiod}"
        )
    else:
        detail = None
    if detail:
        warnings.warn(
            f"short field line ({detail}): growth rates and fluxes are not converged "
            "below npol = 3 (VMEC) or nperiod = 2 (analytic); repeat at a longer "
            "line and compare",
            RuntimeWarning,
            stacklevel=3,
        )
