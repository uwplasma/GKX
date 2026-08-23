"""Resolution-estimator contract against the 2026-08 y0=14 ladder.

The corrected ladder (dky*rho = 0.071; Nx=Ny in {64, 96, 128}; Nz=48, Nl=4,
Nm=8, t_max=400 with saturation auto-stop) showed DIII-D converged by 96^2
(64^2 only ~8% high) while every stellarator case was still falling at
128^2. It also falsified the earlier per-case anisotropy ordering (QHS has
the lowest stellarator anisotropy yet the steepest continuing decline), so
the estimator's contract is CLASS-based: nfp splits tokamak from
stellarator, anisotropy only corroborates, and the tiers land on the
calibrated rungs 64/96/128 (tokamak) and 96/128/192 (stellarator) at the
ladder's box, with stellarator tiers annotated as upper estimates.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gkx.workflows.runtime.resolution import (
    PERP_LADDER,
    GeometryFeatures,
    geometry_class,
    ky_max_target,
    perp_points_for,
    resolution_from_features,
)
from gkx.workflows.runtime import wout as runtime_wout

LADDER_DKY = 1.0 / 14.0

# case -> (wout file, anisotropy, |B| wells, geometry class). Features were
# recorded from the ladder geometry (torflux=0.64, ntheta=48).
SCAN_CASES = {
    "tok_diiid": ("wout_DIII-D_lasym_false.nc", 0.1381, 1, "tokamak"),
    "qhs": ("wout_NuhrenbergZille_1988_QHS.nc", 0.5522, 3, "stellarator"),
    "qi": ("wout_QI_stel_seed_3127.nc", 0.5909, 19, "stellarator"),
    "qa_b0p5": (
        "wout_LandremanPaul2021_QA_beta0p5_bootstrap.nc",
        0.6193,
        1,
        "stellarator",
    ),
    "qh": (
        "wout_LandremanPaul2021_QH_reactorScale_lowres.nc",
        0.6744,
        2,
        "stellarator",
    ),
    "qa_b2p5": (
        "wout_LandremanPaul2021_QA_beta2p5_bootstrap.nc",
        0.6931,
        1,
        "stellarator",
    ),
    "qa_vac": ("wout_LandremanPaul2021_QA_lowres.nc", 0.7964, 1, "stellarator"),
}

#: Calibrated rungs per (class, tier) at the ladder's dky*rho = 0.071.
TIER_RUNGS = {
    "tokamak": {"preview": 64, "standard": 96, "cautious": 128},
    "stellarator": {"preview": 96, "standard": 128, "cautious": 192},
}


def _features(
    anisotropy: float, wells: int = 1, shat: float = 1.0, nfp: int = 2
) -> GeometryFeatures:
    return GeometryFeatures(
        anisotropy=anisotropy, shat=shat, q=2.0, nfp=nfp, bmag_wells=wells, zp=1.0
    )


def test_class_split_is_nfp_first_anisotropy_second() -> None:
    assert geometry_class(_features(0.14, nfp=1)) == "tokamak"
    # nfp > 1 is a stellarator no matter how tokamak-like the metric looks.
    assert geometry_class(_features(0.14, nfp=2)) == "stellarator"
    # An nfp=1 equilibrium with a stellarator-band metric rounds up in cost.
    assert geometry_class(_features(0.55, nfp=1)) == "stellarator"


def test_tiers_land_on_the_calibrated_rungs() -> None:
    for klass, nfp in (("tokamak", 1), ("stellarator", 2)):
        anisotropy = 0.14 if klass == "tokamak" else 0.62
        for tier, rung in TIER_RUNGS[klass].items():
            est = resolution_from_features(
                _features(anisotropy, nfp=nfp), dky=LADDER_DKY, target_error=tier
            )
            assert (est["nx"], est["ny"]) == (rung, rung), (klass, tier)
            assert est["geometry_class"] == klass


def test_target_error_tiers_are_monotone() -> None:
    for nfp in (1, 2):
        for anisotropy in (0.14, 0.55, 0.80):
            rungs = [
                PERP_LADDER.index(
                    int(
                        resolution_from_features(
                            _features(anisotropy, nfp=nfp),
                            dky=LADDER_DKY,
                            target_error=t,
                        )["nx"]
                    )
                )
                for t in ("preview", "standard", "cautious")
            ]
            assert rungs[0] <= rungs[1] <= rungs[2]


def test_invalid_target_error_raises() -> None:
    with pytest.raises(ValueError, match="target_error"):
        ky_max_target(_features(0.5), "fast")


def test_perp_points_reproduce_ladder_reaches() -> None:
    # At the ladder's box the class tiers land exactly on the ladder rungs.
    for target, rung in ((1.5, 64), (2.2, 96), (2.9, 128), (4.4, 192)):
        assert perp_points_for(LADDER_DKY, target) == rung


def test_velocity_floors() -> None:
    base = resolution_from_features(_features(0.5), dky=LADDER_DKY)
    assert (base["nl"], base["nm"]) == (4, 8)
    no_hyper = resolution_from_features(
        _features(0.5), dky=LADDER_DKY, hypercollisions=False
    )
    assert (no_hyper["nl"], no_hyper["nm"]) == (6, 12)
    kin_e = resolution_from_features(
        _features(0.5), dky=LADDER_DKY, kinetic_electrons=True
    )
    assert kin_e["nm"] >= 16


def test_parallel_floor_notes_and_cautious_nz() -> None:
    # QI-like tube: 19 deep |B| wells ask Nz >= 114 -> 120 after rounding.
    feats = _features(0.59, wells=19, shat=-0.34)
    std = resolution_from_features(feats, dky=LADDER_DKY)
    assert std["nz"] == 48  # ladder default kept; the floor is advisory
    assert any("Nz >= 120" in note for note in std["notes"])
    caut = resolution_from_features(feats, dky=LADDER_DKY, target_error="cautious")
    assert caut["nz"] == 120


def test_stellarator_upper_estimate_note_and_cheaper_box() -> None:
    est = resolution_from_features(_features(0.7964, shat=0.02), dky=1.0 / 21.0)
    # A wider box than the calibrated one earns the cheaper-box pointer.
    assert any("y0 = 14" in note for note in est["notes"])
    assert any("upper estimate" in note for note in est["notes"])
    assert any("low-shear" in note for note in est["notes"])
    at_ladder = resolution_from_features(_features(0.62), dky=LADDER_DKY)
    assert not any("cheaper box" in note for note in at_ladder["notes"])
    assert not any(
        "upper estimate" in note
        for note in resolution_from_features(
            _features(0.14, nfp=1), dky=LADDER_DKY
        )["notes"]
    )


def test_estimate_flag_short_circuits_before_any_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str | None, str]] = []

    def _fake_print(wout_path: Path, config_arg: str | None, *, target_error: str) -> None:
        calls.append((wout_path, config_arg, target_error))

    monkeypatch.setattr(runtime_wout, "_print_resolution_estimate", _fake_print)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        runtime_wout.wout_shorthand_args("wout_case.nc", None, ["--estimate=preview"])
    assert excinfo.value.code == 0
    assert calls and calls[0][2] == "preview"
    assert list(tmp_path.iterdir()) == []  # no resolved deck, no output dir


def test_estimate_flag_defaults_to_standard() -> None:
    args = ["--out", "x", "--estimate", "--progress"]
    assert runtime_wout._pop_estimate_flag(args) == "standard"
    assert args == ["--out", "x", "--progress"]
    assert runtime_wout._pop_estimate_flag(args) is None


def _scan_wouts_dir() -> Path | None:
    candidates = [os.environ.get("GKX_RESOLUTION_SCAN_WOUTS")]
    candidates.append(str(Path.home() / "gkx-runs/resolution_scan/wouts"))
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand)
    return None


@pytest.mark.integration
def test_estimator_end_to_end_on_scan_equilibria() -> None:
    """Full wout -> geometry -> estimate path against the recorded features."""

    wouts = _scan_wouts_dir()
    if wouts is None:
        pytest.skip("resolution-scan wout files not present on this machine")
    from gkx.workflows.runtime.resolution import estimate_resolution

    checked = 0
    for _case, (fname, anisotropy, wells, klass) in SCAN_CASES.items():
        path = wouts / fname
        if not path.is_file():
            continue
        checked += 1
        est = estimate_resolution(path, torflux=0.64)
        feats = est["features"]
        assert feats.anisotropy == pytest.approx(anisotropy, abs=0.02)
        assert feats.bmag_wells == wells
        assert est["geometry_class"] == klass
        assert 0.0 < est["dt"] <= 0.1
    if checked == 0:
        pytest.skip("no scan wout files found in the scan directory")
