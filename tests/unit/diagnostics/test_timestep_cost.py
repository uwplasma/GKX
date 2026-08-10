"""Time-step cost and CFL-attribution diagnostics.

Every expected number here is hand-computable from the synthetic series, so a
regression in the reported cost per unit of simulated time shows up as a
concrete arithmetic mismatch rather than a shifted baseline.
"""

import json

import numpy as np
import pytest

from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.analysis import (
    CFL_TERM_NAMES,
    CFL_TERM_UNRESOLVED,
    CFLScales,
    cfl_limiter_report,
    cfl_limiting_term,
    cfl_scales_from_array,
    cfl_term_contributions,
)
from gkx.diagnostics.metadata import CFL_SCALE_LABELS
from gkx.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    timestep_cost_payload,
    timestep_cost_report,
)


def _series(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, dt)`` with the runtime convention ``t[i] - t[i-1] == dt[i]``."""

    return np.cumsum(dt), dt


def _diag(t: np.ndarray, dt: np.ndarray, **kwargs: object) -> SimulationDiagnostics:
    zeros = np.zeros_like(t)
    return SimulationDiagnostics(
        t=t,
        dt_t=dt,
        dt_mean=np.asarray(np.mean(dt)),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=zeros,
        Wphi_t=zeros,
        Wapar_t=zeros,
        heat_flux_t=zeros,
        particle_flux_t=zeros,
        energy_t=zeros,
        **kwargs,
    )


# --- cost per unit of simulated time --------------------------------------


def test_steps_and_dt_collapse_match_a_hand_computed_series() -> None:
    """Reported cost reproduces arithmetic done by hand on a known series."""

    # dt halves every step: four samples, so three gaps, each of which is
    # exactly one step. t runs 1.0, 1.5, 1.75, 1.875 -> span 0.875.
    t, dt = _series(np.array([1.0, 0.5, 0.25, 0.125]))
    report = timestep_cost_report(t, dt, wall_seconds=8.75)

    assert report.n_samples == 4
    assert report.t_start == pytest.approx(1.0)
    assert report.t_end == pytest.approx(1.875)
    assert report.t_span == pytest.approx(0.875)
    assert report.steps == pytest.approx(3.0)
    assert report.steps_are_exact is True
    assert report.steps_per_unit_time == pytest.approx(3.0 / 0.875)
    assert report.wall_seconds_per_unit_time == pytest.approx(8.75 / 0.875)
    assert report.dt_initial == pytest.approx(1.0)
    assert report.dt_min == pytest.approx(0.125)
    assert report.dt_final == pytest.approx(0.125)
    # The headline collapse number: the first step was eight times the smallest.
    assert report.dt_collapse_ratio == pytest.approx(8.0)


def test_constant_step_run_reports_one_step_per_dt_and_no_collapse() -> None:
    """A run that never adapts costs exactly ``1 / dt`` steps per unit time."""

    t, dt = _series(np.full(101, 0.25))
    report = timestep_cost_report(t, dt)

    assert report.steps == pytest.approx(100.0)
    assert report.steps_per_unit_time == pytest.approx(4.0)
    assert report.dt_collapse_ratio == pytest.approx(1.0)
    assert report.cost_growth_ratio == pytest.approx(1.0)
    assert report.wall_seconds is None
    assert report.wall_seconds_per_unit_time is None


def test_cost_growth_ratio_compares_the_first_and_last_quarter_of_a_run() -> None:
    """Within-run degradation is measured over time, not sample index."""

    # First half at dt=1 (1 step per unit t), second half at dt=0.25 (4 per
    # unit t). The quarters therefore differ by exactly a factor of four.
    dt = np.concatenate([np.full(40, 1.0), np.full(160, 0.25)])
    t, dt = _series(dt)
    report = timestep_cost_report(t, dt)

    assert report.steps_per_unit_time_first_quarter == pytest.approx(1.0)
    assert report.steps_per_unit_time_last_quarter == pytest.approx(4.0)
    assert report.cost_growth_ratio == pytest.approx(4.0)


def test_strided_series_is_reported_as_an_estimate_not_a_count() -> None:
    """Striding makes the step count approximate, and the report says so."""

    t_full, dt_full = _series(np.full(101, 0.25))
    report = timestep_cost_report(t_full[::5], dt_full[::5])

    assert report.steps_are_exact is False
    assert any("strided" in note for note in report.notes)
    # Each retained gap spans five steps, so the estimate recovers the truth
    # here precisely because dt is constant.
    assert report.steps == pytest.approx(100.0)


def test_report_rejects_mismatched_and_unusable_series() -> None:
    """A malformed series raises rather than reporting a meaningless number."""

    with pytest.raises(ValueError, match="same length"):
        timestep_cost_report(np.arange(3.0), np.ones(4))
    with pytest.raises(ValueError, match="usable sample"):
        timestep_cost_report(np.array([np.nan]), np.array([np.nan]))


# --- which CFL term is limiting --------------------------------------------


def test_term_contributions_are_additive_and_reproduce_the_integrator_sum() -> None:
    """Contributions sum to the frequency the integrator actually forms."""

    contributions = cfl_term_contributions(
        magnetic_drift_radial=3.0,
        magnetic_drift_binormal=5.0,
        parallel_streaming=2.0,
        exb_radial=11.0,
        exb_binormal=1.0,
    )

    # max(3, 11) + max(5, 1) + 2 == 18
    assert sum(contributions.values()) == pytest.approx(18.0)
    assert contributions["exb"] == pytest.approx(8.0)
    assert set(contributions) == set(CFL_TERM_NAMES)


@pytest.mark.parametrize(
    ("dominant", "speeds"),
    [
        ("exb", {"exb_binormal": 100.0}),
        ("parallel_streaming", {"parallel_streaming": 100.0}),
        ("magnetic_drift_radial", {"magnetic_drift_radial": 100.0}),
        ("magnetic_drift_binormal", {"magnetic_drift_binormal": 100.0}),
    ],
)
def test_limiting_term_names_the_speed_that_dominates_by_construction(
    dominant: str, speeds: dict[str, float]
) -> None:
    """One speed set 100x the rest is the term the report names."""

    args = {
        "magnetic_drift_radial": 1.0,
        "magnetic_drift_binormal": 1.0,
        "parallel_streaming": 1.0,
        "exb_radial": 1.0,
        "exb_binormal": 1.0,
    }
    args.update(speeds)
    term, share = cfl_limiting_term(cfl_term_contributions(**args))

    assert term == dominant
    assert share > 0.9


def test_matched_exb_does_not_displace_the_drift_it_equals() -> None:
    """An ExB speed that merely matches a drift is not reported as limiting."""

    term, _share = cfl_limiting_term(
        cfl_term_contributions(
            magnetic_drift_radial=4.0,
            magnetic_drift_binormal=1.0,
            parallel_streaming=1.0,
            exb_radial=4.0,
            exb_binormal=1.0,
        )
    )
    assert term == "magnetic_drift_radial"


def test_dt_trajectory_inverts_to_a_growing_exb_share() -> None:
    """A collapsing dt is attributed to the ExB excess over the linear floor."""

    # Linear floor 1 + 2 + 3 = 6, numerator 12, so dt = 12 / omega_total.
    scales = CFLScales(1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0)
    # omega_total = 6 (pure linear floor) then 12, 24, 48.
    report = cfl_limiter_report(np.array([2.0, 1.0, 0.5, 0.25]), scales)

    assert report.omega_linear_floor == pytest.approx(6.0)
    assert report.omega_total_initial == pytest.approx(6.0)
    assert report.omega_total_final == pytest.approx(48.0)
    assert report.exb_share_initial == pytest.approx(0.0)
    # At the end ExB supplies 48 - 6 = 42 of the 48.
    assert report.exb_share_final == pytest.approx(42.0 / 48.0)
    assert report.limiting_term_initial == "parallel_streaming"
    assert report.limiting_term_final == "exb"
    assert report.limiting_term_final_share == pytest.approx(42.0 / 48.0)
    assert report.limiting_term_sample_fractions["exb"] == pytest.approx(0.75)
    assert report.samples_attributed == 4


def test_a_run_pinned_at_the_dt_ceiling_is_reported_as_capped_not_limited() -> None:
    """A capped step only bounds the frequency, so no term is named."""

    scales = CFLScales(1.0, 2.0, 3.0, 12.0, 1.0e-3, 2.0)
    report = cfl_limiter_report(np.full(5, 2.0), scales)

    assert report.samples_at_dt_ceiling == 5
    assert report.samples_attributed == 0
    assert report.limiting_term_final == CFL_TERM_UNRESOLVED
    assert np.isnan(report.omega_total_final)


def test_scales_decode_round_trips_and_rejects_unusable_vectors() -> None:
    """The recorded vector decodes positionally and fails closed."""

    values = [1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0]
    assert len(CFL_SCALE_LABELS) == len(values)
    assert cfl_scales_from_array(np.asarray(values)) == CFLScales(*values)
    assert cfl_scales_from_array(None) is None
    assert cfl_scales_from_array(np.asarray(values[:-1])) is None
    assert cfl_scales_from_array(np.asarray([1.0, 2.0, 3.0, 0.0, 1e-3, 10.0])) is None
    assert cfl_scales_from_array(np.asarray([np.nan, *values[1:]])) is None


# --- what a run's own summary reports --------------------------------------


def test_summary_payload_is_json_safe_and_carries_the_cost_block() -> None:
    """The block a user reads is valid JSON even where ratios are undefined."""

    t, dt = _series(np.array([2.0, 1.0, 0.5, 0.25]))
    scales = np.asarray([1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0])
    payload = timestep_cost_payload(_diag(t, dt, cfl_scales=scales), wall_seconds=42.0)

    assert payload["dt_collapse_ratio"] == pytest.approx(8.0)
    assert payload["wall_seconds_per_unit_time"] == pytest.approx(42.0 / 1.75)
    assert payload["cfl"]["limiting_term_final"] == "exb"
    # exb_growth_ratio is infinite here (the run starts on the linear floor);
    # strict JSON has no Infinity, so it must serialise as null.
    assert payload["cfl"]["exb_growth_ratio"] is None
    json.loads(json.dumps(payload, allow_nan=False))


def test_summary_payload_survives_a_run_without_recorded_cfl_scales() -> None:
    """Fixed-step and reloaded runs still report cost, without attribution."""

    t, dt = _series(np.full(5, 0.5))
    payload = timestep_cost_payload(_diag(t, dt))

    assert payload["cfl"] is None
    assert payload["steps_per_unit_time"] == pytest.approx(2.0)


def test_empty_diagnostics_yield_an_empty_block_rather_than_a_write_failure() -> None:
    """A summary artifact never fails to write because of the cost block."""

    empty = np.asarray([], dtype=float)
    assert timestep_cost_payload(_diag(empty, empty)) == {}


def test_cfl_scales_survive_slicing_striding_and_concatenation() -> None:
    """Run-constant scales are not per-sample, so reshaping must not drop them."""

    t, dt = _series(np.full(8, 0.5))
    scales = np.asarray([1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0])
    diag = _diag(t, dt, cfl_scales=scales)

    for reshaped in (
        slice_runtime_diagnostics(diag, 4),
        stride_runtime_diagnostics(diag, stride=2),
        concat_runtime_diagnostics([diag, diag]),
    ):
        assert reshaped.cfl_scales is not None
        np.testing.assert_allclose(np.asarray(reshaped.cfl_scales), scales)
