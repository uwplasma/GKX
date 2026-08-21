"""Unit contracts: vmec transport objectives."""

from __future__ import annotations


# ---- test_vmex_boundary_chain.py ----

import json
import math

import pytest

import gkx
import gkx.geometry.vmec_boundary_chain as boundary_chain
from gkx.geometry.vmec_boundary_chain import (
    boundary_chain_summary_from_probe,
    build_boundary_chain_collection_summary,
    build_boundary_chain_summary,
)


def test_boundary_chain_error_and_pass_helpers_handle_optional_linear_values() -> None:
    errors = boundary_chain._boundary_chain_error_metrics(
        exact=1.0,
        final=1.02,
        frozen_jvp=0.91,
        frozen_vjp=0.91 + 1.0e-12,
        frozen_linear_jvp=None,
        frozen_linear_vjp=None,
        tangent_diff_abs=None,
        tangent_diff_rel=None,
        raw=None,
        absolute_tolerance=1.0e-10,
    )

    assert errors["final_state_vs_exact_fd_rel"] == pytest.approx(0.01960784313725492)
    assert errors["frozen_axis_fd_jvp_vs_linear_jvp_abs"] is None
    assert errors["frozen_axis_fd_vjp_vs_linear_vjp_rel"] is None
    assert errors["raw_initial_vs_exact_fd_abs"] is None

    passes = boundary_chain._boundary_chain_passes(
        errors,
        raw=None,
        exact_relative_tolerance=0.1,
        internal_relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-10,
    )

    assert passes["final_state_matches_exact_fd"] is True
    assert passes["frozen_axis_matches_exact_fd"] is True
    assert passes["frozen_axis_jvp_vjp_consistent"] is True
    assert passes["frozen_axis_convention_verified"] is False
    assert passes["raw_initial_matches_exact_fd"] is False


def test_boundary_chain_collection_helpers_have_canonical_owner() -> None:
    for helper in (
        boundary_chain._empty_boundary_chain_counts,
        boundary_chain._boundary_chain_collection_counts,
        boundary_chain._boundary_chain_collection_decision,
    ):
        assert helper.__module__ == "gkx.geometry.vmec_boundary_chain"


def test_boundary_chain_summary_classifies_frozen_axis_branch_sensitivity() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.5936446584,
        final_cot_dot_exact_final_fd=0.5238431385,
        frozen_axis_replay_cost_gradient=0.10354209293,
        frozen_axis_vjp_cost_gradient=0.10354209294,
        raw_initial_replay_cost_gradient=0.46125484804,
        raw_initial_fd_norm=209.00303407,
        frozen_axis_initial_fd_norm=1.81142209,
    )

    assert summary["finite"] is True
    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(
        115.379, rel=1.0e-3
    )
    assert "magnetic-axis initialization branch" in summary["next_action"]
    json.dumps(summary, allow_nan=False)


def test_boundary_chain_summary_verifies_frozen_axis_convention_despite_raw_exact_branch_sensitivity() -> (
    None
):
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=1.0,
        final_cot_dot_exact_final_fd=0.98,
        frozen_axis_replay_cost_gradient=0.5,
        frozen_axis_vjp_cost_gradient=0.5 + 1.0e-13,
        frozen_axis_linear_replay_cost_gradient=0.5 + 2.0e-13,
        frozen_axis_linear_vjp_cost_gradient=0.5,
        frozen_axis_initial_fd_vs_linear_abs_norm=1.0e-13,
        frozen_axis_initial_fd_vs_linear_rel=1.0e-13,
        raw_initial_replay_cost_gradient=3.0,
        raw_initial_fd_norm=120.0,
        frozen_axis_initial_fd_norm=1.0,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == (
        "frozen_axis_convention_verified_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_fd_matches_linear_tangent"] is True
    assert summary["passes"]["frozen_axis_fd_jvp_matches_linear_jvp"] is True
    assert summary["passes"]["frozen_axis_linear_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_fd_vjp_matches_linear_vjp"] is True
    assert summary["passes"]["frozen_axis_convention_verified"] is True
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(
        120.0
    )
    assert "explicit tangent column" in summary["next_action"]


def test_boundary_chain_summary_anchors_latest_rc14_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.07727649731974727,
        final_cot_dot_exact_final_fd=0.07362260082651849,
        frozen_axis_replay_cost_gradient=0.11282399953461382,
        frozen_axis_vjp_cost_gradient=0.11282399953439505,
        raw_initial_replay_cost_gradient=0.10166856394763844,
        raw_initial_fd_norm=70.29927985025522,
        frozen_axis_initial_fd_norm=1.8114220932737295,
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
    )
    assert summary["passes"]["final_state_matches_exact_fd"] is True
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.04728341242111478
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.31507039602829146
    )


def test_boundary_chain_summary_anchors_latest_zs13_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=-0.11873119671939811,
        final_cot_dot_exact_final_fd=-0.11900005540528957,
        frozen_axis_replay_cost_gradient=-0.14192363954706916,
        frozen_axis_vjp_cost_gradient=-0.14192363954695417,
        raw_initial_replay_cost_gradient=-0.1419716858450226,
        raw_initial_fd_norm=1.8888557263064212,
        frozen_axis_initial_fd_norm=1.8114220932737053,
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "frozen_axis_replay_consistent_but_exact_fd_inconsistent"
    )
    assert summary["passes"]["final_state_matches_exact_fd"] is True
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.0022593156362472975
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.16341493849570599
    )
    assert summary["metrics"]["raw_to_frozen_initial_norm_ratio"] == pytest.approx(
        1.0427474266325047
    )


def test_boundary_chain_summary_anchors_coefficient_local_rc14_probe() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=-0.008248825352711719,
        final_cot_dot_exact_final_fd=-0.0073983155214745865,
        frozen_axis_replay_cost_gradient=0.11282399953752609,
        frozen_axis_vjp_cost_gradient=0.11282399953729642,
        raw_initial_replay_cost_gradient=0.11313073901185483,
        raw_initial_fd_norm=1.939974024843866,
        frozen_axis_initial_fd_norm=1.8114220932738312,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == "final_state_cotangent_mismatch"
    assert summary["passes"]["final_state_matches_exact_fd"] is False
    assert summary["passes"]["frozen_axis_jvp_vjp_consistent"] is True
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is False
    assert summary["errors"]["final_state_vs_exact_fd_rel"] == pytest.approx(
        0.10310678125310722
    )
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        1.0731123288176654
    )


def test_boundary_chain_summary_accepts_converged_sparse_fd_window() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=0.02245397716,
        final_cot_dot_exact_final_fd=0.0221,
        frozen_axis_replay_cost_gradient=0.02079152741,
        frozen_axis_vjp_cost_gradient=0.02079152740,
        raw_initial_replay_cost_gradient=0.0208,
        raw_initial_fd_norm=1.9,
        frozen_axis_initial_fd_norm=1.8,
        exact_relative_tolerance=0.1,
    )

    assert summary["classification"] == "exact_fd_and_frozen_axis_replay_consistent"
    assert summary["passes"]["frozen_axis_matches_exact_fd"] is True
    assert summary["errors"]["frozen_axis_vs_exact_fd_rel"] == pytest.approx(
        0.0740380977
    )


def test_boundary_chain_summary_reports_nonfinite_payload() -> None:
    summary = build_boundary_chain_summary(
        exact_fd_cost_gradient=math.nan,
        final_cot_dot_exact_final_fd=1.0,
        frozen_axis_replay_cost_gradient=1.0,
        frozen_axis_vjp_cost_gradient=1.0,
    )

    assert summary["finite"] is False
    assert summary["classification"] == "nonfinite_boundary_chain_probe"
    assert summary["errors"] == {}


def test_boundary_chain_summary_from_probe_and_public_api() -> None:
    payload = {
        "exact_fd_cost_gradient": 0.02245397716,
        "final_cot_dot_exact_final_fd": 0.0221,
        "final_cot_dot_tape_jvp_frozen_axis_fd": 0.02079152741,
        "initial_cot_dot_frozen_axis_fd": 0.02079152740,
        "final_cot_dot_tape_jvp_frozen_axis_linear": 0.02079152741001,
        "initial_cot_dot_frozen_axis_linear": 0.02079152740001,
        "frozen_axis_initial_fd_vs_linear_abs_norm": 1.0e-13,
        "frozen_axis_initial_fd_vs_linear_rel": 1.0e-13,
        "final_cot_dot_tape_jvp_raw_initial_fd": 0.0208,
        "raw_initial_fd_norm": 1.9,
        "frozen_axis_initial_fd_norm": 1.8,
    }

    assert gkx.build_boundary_chain_summary is build_boundary_chain_summary
    assert (
        gkx.boundary_chain_summary_from_probe
        is boundary_chain_summary_from_probe
    )
    summary = boundary_chain_summary_from_probe(payload, exact_relative_tolerance=0.1)
    assert summary["classification"] == "exact_fd_and_frozen_axis_replay_consistent"
    assert summary["passes"]["frozen_axis_convention_verified"] is True


def test_boundary_chain_collection_count_helper_tracks_gate_categories() -> None:
    rows = [
        {
            "finite": True,
            "frozen_axis_jvp_vjp_consistent": True,
            "frozen_axis_convention_verified": False,
            "frozen_axis_matches_exact_fd": True,
            "classification": "exact_fd_and_frozen_axis_replay_consistent",
            "growth_branch_locality_checked": True,
            "growth_branch_locality_passed": True,
        },
        {
            "finite": True,
            "frozen_axis_jvp_vjp_consistent": True,
            "frozen_axis_convention_verified": True,
            "frozen_axis_matches_exact_fd": False,
            "classification": (
                "frozen_axis_convention_verified_but_exact_fd_branch_sensitive"
            ),
            "growth_branch_locality_checked": True,
            "growth_branch_locality_passed": False,
        },
    ]

    assert boundary_chain._boundary_chain_collection_counts(rows) == {
        "n_total": 2,
        "n_finite": 2,
        "n_frozen_axis_internal_pass": 2,
        "n_frozen_axis_convention_verified": 1,
        "n_exact_fd_consistent": 1,
        "n_branch_sensitive": 1,
        "n_growth_branch_locality_checked": 2,
        "n_growth_branch_locality_passed": 1,
    }
    assert boundary_chain._empty_boundary_chain_counts()["n_total"] == 0


def test_boundary_chain_collection_decision_policy_is_explicit() -> None:
    base = boundary_chain._empty_boundary_chain_counts() | {
        "n_total": 2,
        "n_finite": 2,
        "n_frozen_axis_internal_pass": 2,
    }

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_finite": 1}
    )
    assert finite is False
    assert classification == "nonfinite_boundary_chain_collection"
    assert "repair nonfinite" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_frozen_axis_internal_pass": 1}
    )
    assert finite is True
    assert classification == "internal_replay_failure"
    assert "exact-tape replay" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_exact_fd_consistent": 2}
    )
    assert finite is True
    assert classification == "all_components_exact_fd_and_frozen_axis_consistent"
    assert "sparse-FD gates" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_exact_fd_consistent": 1, "n_branch_sensitive": 1}
    )
    assert finite is True
    assert classification == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    assert "regularize branch-sensitive modes" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_frozen_axis_convention_verified": 2}
    )
    assert finite is True
    assert classification == "all_components_frozen_axis_convention_verified"
    assert "nonlinear-audit gates" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base
        | {
            "n_exact_fd_consistent": 1,
            "n_frozen_axis_convention_verified": 1,
            "n_branch_sensitive": 1,
        }
    )
    assert finite is True
    assert classification == "mixed_exact_or_frozen_axis_convention_verified"
    assert "unresolved branch-sensitive modes" in action

    finite, classification, action = boundary_chain._boundary_chain_collection_decision(
        base | {"n_branch_sensitive": 1}
    )
    assert finite is True
    assert classification == "branch_sensitive_boundary_chain_collection"
    assert "better-conditioned finite-difference protocol" in action


def test_boundary_chain_collection_summary_counts_mixed_modes() -> None:
    branch_sensitive = {
        "index": 22,
        "name": "rc11",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=-0.0135,
            final_cot_dot_exact_final_fd=-0.014,
            frozen_axis_replay_cost_gradient=-0.0172,
            frozen_axis_vjp_cost_gradient=-0.0172,
            raw_initial_replay_cost_gradient=-0.092,
            raw_initial_fd_norm=120.0,
            frozen_axis_initial_fd_norm=1.0,
            exact_relative_tolerance=0.1,
        ),
    }
    consistent = {
        "index": 28,
        "name": "rc14",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=0.02245397716,
            final_cot_dot_exact_final_fd=0.0221,
            frozen_axis_replay_cost_gradient=0.02079152741,
            frozen_axis_vjp_cost_gradient=0.02079152740,
            raw_initial_replay_cost_gradient=0.0208,
            raw_initial_fd_norm=1.9,
            frozen_axis_initial_fd_norm=1.8,
            exact_relative_tolerance=0.1,
        ),
        "growth_branch_locality": {
            "enabled": True,
            "passed": True,
            "classification": "all_samples_dominant_growth_branch_locally_consistent",
        },
    }

    summary = build_boundary_chain_collection_summary(
        [branch_sensitive, consistent],
        exact_relative_tolerance=0.1,
    )

    assert (
        summary["classification"]
        == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    )
    assert summary["counts"] == {
        "n_total": 2,
        "n_finite": 2,
        "n_frozen_axis_internal_pass": 2,
        "n_frozen_axis_convention_verified": 0,
        "n_exact_fd_consistent": 1,
        "n_branch_sensitive": 1,
        "n_growth_branch_locality_checked": 1,
        "n_growth_branch_locality_passed": 1,
    }
    assert summary["rows"][0]["name"] == "rc11"
    assert summary["rows"][1]["frozen_axis_matches_exact_fd"] is True
    assert summary["rows"][1]["exact_fd_consistent"] is True
    assert summary["rows"][1]["growth_branch_locality_passed"] is True
    assert (
        summary["rows"][1]["growth_branch_locality_classification"]
        == "all_samples_dominant_growth_branch_locally_consistent"
    )
    assert "exclude or regularize branch-sensitive modes" in summary["next_action"]


def test_boundary_chain_collection_counts_verified_frozen_axis_convention_separately() -> (
    None
):
    verified = {
        "index": 3,
        "name": "rc11",
        "summary": build_boundary_chain_summary(
            exact_fd_cost_gradient=1.0,
            final_cot_dot_exact_final_fd=0.99,
            frozen_axis_replay_cost_gradient=0.5,
            frozen_axis_vjp_cost_gradient=0.5,
            frozen_axis_linear_replay_cost_gradient=0.5,
            frozen_axis_linear_vjp_cost_gradient=0.5,
            frozen_axis_initial_fd_vs_linear_abs_norm=0.0,
            frozen_axis_initial_fd_vs_linear_rel=0.0,
            raw_initial_replay_cost_gradient=4.0,
            raw_initial_fd_norm=100.0,
            frozen_axis_initial_fd_norm=1.0,
        ),
    }

    summary = build_boundary_chain_collection_summary([verified])

    assert summary["classification"] == "all_components_frozen_axis_convention_verified"
    assert summary["counts"]["n_exact_fd_consistent"] == 0
    assert summary["counts"]["n_frozen_axis_convention_verified"] == 1
    assert summary["rows"][0]["frozen_axis_matches_exact_fd"] is False
    assert summary["rows"][0]["exact_fd_consistent"] is False
    assert summary["rows"][0]["frozen_axis_convention_verified"] is True
    assert "growth-branch" in summary["next_action"]


def test_boundary_chain_collection_summary_fails_closed_when_empty() -> None:
    summary = build_boundary_chain_collection_summary([])

    assert summary["finite"] is False
    assert summary["classification"] == "empty_boundary_chain_collection"
    assert summary["counts"]["n_total"] == 0
    assert summary["counts"]["n_frozen_axis_convention_verified"] == 0
    assert summary["counts"]["n_growth_branch_locality_checked"] == 0


# ---- test_vmex_candidate_gate.py ----

import sys
from types import SimpleNamespace

import numpy as np

from gkx.objectives.vmec_candidate_admission import (
    build_authoritative_wout_candidate_gate,
    build_solved_vmec_candidate_gate,
    build_wout_reproducibility_gate,
    final_iota_profiles_from_vmec_result,
)


POLICY = {
    "target_aspect": 6.0,
    "aspect_atol": 5.0e-2,
    "min_abs_mean_iota": 0.41,
    "qs_residual_max": 5.0e-2,
    "iota_profile_floor": 0.41,
}


def test_candidate_gate_rejects_bad_history_without_nonfinite_json() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": "bad", "iota_final": np.inf, "qs_final": np.nan},
        **POLICY,
        iota_profiles=(np.asarray([0.0, 0.412]), np.asarray([0.413])),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["value"] is None
    assert report["checks"]["aspect"]["passed"] is False
    assert report["checks"]["mean_iota"]["value"] is None
    assert report["checks"]["quasisymmetry"]["value"] is None
    json.dumps(report, allow_nan=False)


def test_candidate_gate_requires_iota_profiles_when_floor_is_enabled() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        **POLICY,
    )

    assert report["passed"] is False
    assert report["checks"]["iota_profile"]["source"] == "missing"
    assert report["checks"]["iota_profile"]["passed"] is False


def test_candidate_gate_can_disable_profile_floor_for_fast_diagnostic_use() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["iota_profile"]["floor"] is None
    assert report["checks"]["iota_profile"]["passed"] is True


def test_final_iota_profiles_from_vmec_result_returns_none_without_solved_state() -> (
    None
):
    assert final_iota_profiles_from_vmec_result(SimpleNamespace(history={})) is None


def test_candidate_gate_extracts_iota_profiles_from_vmex_state(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vmex", SimpleNamespace())
    wout = SimpleNamespace(
        iotas=np.asarray([0.0, 0.411, 0.415]),
        iotaf=np.asarray([0.412, 0.416]),
    )
    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": -0.42, "qs_final": 0.02},
        final_equilibrium=SimpleNamespace(wout=wout),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == 0.42
    assert report["checks"]["iota_profile"]["source"] == "vmex_state"
    assert report["checks"]["iota_profile"]["minimum_iotas_excluding_axis"] == 0.411


def test_candidate_gate_prefers_independent_state_qs_over_history(monkeypatch) -> None:
    class FakeQS:
        def __init__(self, surfaces, *, helicity_m=1, helicity_n=0, **_kwargs):
            assert np.asarray(surfaces).shape[0] == 11
            assert (helicity_m, helicity_n) == (1, 0)

        def total_state(self, state, runtime):
            assert state == "state"
            assert runtime == "runtime"
            return 0.013

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)
    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.013
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_state"


def test_candidate_gate_uses_standalone_qs_not_assembled_transport_block(
    monkeypatch,
) -> None:
    class FakeQS:
        def __init__(self, surfaces, *, helicity_m=1, helicity_n=0, **_kwargs):
            assert np.asarray(surfaces).shape[0] == 11
            assert (helicity_m, helicity_n) == (1, 0)

        def total_state(self, state, runtime):
            assert state == "state"
            assert runtime == "runtime"
            return 0.009

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)

    class FakeOptimizer:
        def quasisymmetry_objective(self, _params):
            raise AssertionError("assembled optimizer objective must not be used")

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_params=(1.0, 2.0),
        final_optimizer=FakeOptimizer(),
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.009
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_state"


def test_candidate_gate_state_qs_falls_back_to_optimizer_method(monkeypatch) -> None:
    class FailingQS:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("qs residual unavailable")

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FailingQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)

    class FakeOptimizer:
        def quasisymmetry_objective(self, params):
            assert params == (1.0, 2.0)
            return 0.017

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_params=(1.0, 2.0),
        final_optimizer=FakeOptimizer(),
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.017


def test_final_iota_profiles_from_vmec_result_handles_vmex_failure() -> None:
    class BrokenEquilibrium:
        @property
        def wout(self):
            raise RuntimeError("not converged")

    result = SimpleNamespace(final_equilibrium=BrokenEquilibrium())

    assert final_iota_profiles_from_vmec_result(result) is None


def test_wout_reproducibility_gate_rejects_iota_drift() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.000154,
            "mean_iota": 0.40851,
            "min_iotas_excluding_axis": 0.39598,
            "min_iotaf": 0.39581,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
    )

    assert report["passed"] is False
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"][
        "absolute_drift"
    ] == pytest.approx(0.00169)
    json.dumps(report, allow_nan=False)


def test_wout_reproducibility_gate_accepts_matching_rerun() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.0001542,
            "mean_iota": 0.41010,
            "min_iotas_excluding_axis": 0.40561,
            "min_iotaf": 0.40545,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
        aspect_repro_atol=5.0e-7,
        profile_repro_atol=5.0e-4,
    )

    assert report["passed"] is True
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is True


def test_authoritative_wout_candidate_gate_accepts_mapping_with_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": -0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
            "qs_residual": 2.0e-3,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == pytest.approx(0.411)
    assert report["checks"]["quasisymmetry"]["source"] == "mapping"
    json.dumps(report, allow_nan=False)


def test_authoritative_wout_candidate_gate_rejects_missing_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": 0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is False
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["error"] == "missing_qs_residual"


def test_authoritative_wout_candidate_gate_reads_wout_file_with_profile_floor(
    tmp_path, monkeypatch
) -> None:
    class FakeVar:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, _key):
            return np.asarray(self.value)

    class FakeDataset:
        def __init__(self, path):
            assert path == tmp_path / "wout_final_rerun.nc"
            self.variables = {
                "aspect": FakeVar(5.0002),
                "iotas": FakeVar([0.0, 0.412, 0.418]),
                "iotaf": FakeVar([0.413, 0.419]),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_read_wout(path):
        assert path == tmp_path / "wout_final_rerun.nc"
        return "loaded-wout"

    class FakeQS:
        def __init__(self, surfaces, *, helicity_m, helicity_n, ntheta, nphi):
            assert tuple(np.asarray(surfaces, dtype=float)) == (0.0, 0.5, 1.0)
            assert (helicity_m, helicity_n, ntheta, nphi) == (1, 0, 31, 32)

        def total(self, wout):
            assert wout == "loaded-wout"
            return 0.003

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules,
        "vmex",
        SimpleNamespace(
            read_wout=fake_read_wout,
            optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS),
        ),
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "wout_final_rerun.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.411,
        qs_surfaces=(0.0, 0.5, 1.0),
        qs_ntheta=31,
        qs_nphi=32,
    )

    assert report["passed"] is True
    assert report["authoritative_wout"]["mean_iota"] == pytest.approx(0.415)
    assert report["checks"]["iota_profile"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_wout"


def test_authoritative_wout_candidate_gate_reports_wout_load_errors(
    tmp_path, monkeypatch
) -> None:
    def broken_dataset(_path):
        raise OSError("missing variable")

    def broken_read_wout(_path):
        raise RuntimeError("bad wout")

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=broken_dataset))
    monkeypatch.setitem(
        sys.modules, "vmex", SimpleNamespace(read_wout=broken_read_wout)
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "bad_wout.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
    )

    assert report["passed"] is False
    assert report["authoritative_wout"]["aspect"] is None
    assert report["checks"]["iota_profile"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_wout_error"


# ---- test_vmex_transport_admission.py ----


import gkx.diagnostics.stellarator_transport_reports as transport_reports
from gkx.diagnostics.stellarator_transport_reports import (
    build_nonlinear_audit_redesign_report,
    build_nonlinear_campaign_admission_report,
    build_nonlinear_landscape_admission_report,
    build_reduced_nonlinear_audit_prelaunch_report,
)
from gkx.objectives.vmec_transport_admission import (
    VMEXNonlinearAuditPolicy,
    VMEXNonlinearCampaignPolicy,
    VMEXReducedPrelaunchPolicy,
    VMEXTransportAdmissionPolicy,
)
from gkx.objectives.vmec_transport_admission import (
    candidate_transport_metric,
    transport_objective_sample_summary,
)
from gkx.objectives.vmec_transport_admission import (
    build_transport_admission_report,
    select_admitted_transport_candidate,
)


def _candidate(
    label: str,
    *,
    objective: float,
    weight: float | None = None,
    passed: bool = True,
    authoritative: bool = True,
    baseline: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "baseline": baseline,
        "transport_weight": weight,
        "passed": passed and authoritative,
        "gate_reported_passed": passed,
        "gate_is_authoritative": authoritative,
        "gate_checks": {
            "aspect": passed,
            "mean_iota": True,
            "quasisymmetry": passed,
            "iota_profile": passed,
        },
        "objective_final": objective,
    }


def test_transport_metric_prefers_explicit_transport_metric_over_total_objective() -> (
    None
):
    metric = candidate_transport_metric(
        {
            "objective_final": 4.0,
            "gkx_objective_final": 2.0,
            "transport_objective_final": 1.0,
        }
    )

    assert metric["available"] is True
    assert metric["source"] == "transport_objective_final"
    assert metric["value"] == 1.0
    assert metric["uses_total_objective_proxy"] is False


def test_transport_admission_selects_largest_physical_improving_weight() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("low", objective=0.8, weight=0.001),
        _candidate("high", objective=0.7, weight=0.005),
        _candidate("failed", objective=0.1, weight=0.01, passed=False),
    ]

    report = build_transport_admission_report(summaries)

    assert report["transport_candidate_admitted"] is True
    assert report["promoted_candidate"]["label"] == "high"
    assert report["promoted_candidate"]["transport_weight"] == 0.005
    assert report["admitted_transport_candidates"] == ["low", "high"]


def test_transport_admission_blocks_worse_transport_metric_even_if_gate_passes() -> (
    None
):
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("worse", objective=1.1, weight=0.001),
    ]

    report = build_transport_admission_report(summaries)
    worse = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert report["promoted_candidate"]["label"] == "baseline"
    assert worse["relative_transport_improvement"] < 0.0
    assert "insufficient_transport_improvement" in worse["admission_blockers"]


def test_transport_admission_blocks_non_authoritative_gate() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("legacy", objective=0.5, weight=0.001, authoritative=False),
    ]

    report = build_transport_admission_report(summaries)
    legacy = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert "non_authoritative_gate" in legacy["admission_blockers"]
    assert report["promoted_candidate"]["label"] == "baseline"


def test_transport_admission_can_require_stronger_relative_improvement() -> None:
    policy = VMEXTransportAdmissionPolicy(minimum_relative_improvement=0.25)
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("small", objective=0.9, weight=0.001),
        _candidate("large", objective=0.7, weight=0.002),
    ]

    report = build_transport_admission_report(summaries, policy=policy)

    assert report["admitted_transport_candidates"] == ["large"]
    assert report["promoted_candidate"]["label"] == "large"


def test_transport_admission_exports_public_api() -> None:
    assert gkx.VMEXTransportAdmissionPolicy is VMEXTransportAdmissionPolicy
    assert (
        gkx.build_transport_admission_report is build_transport_admission_report
    )
    assert gkx.candidate_transport_metric is candidate_transport_metric
    assert (
        gkx.select_admitted_transport_candidate
        is select_admitted_transport_candidate
    )


def _matched_comparison(
    *,
    relative_reduction: float,
    z_score: float,
    passed: bool,
) -> dict[str, object]:
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "case": "qa_projected_transport_step1e3",
        "passed": passed,
        "baseline": {"passed": True, "ensemble_mean": 9.833},
        "candidate": {"passed": True, "ensemble_mean": 9.891},
        "statistics": {
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": z_score,
        },
    }


def _ensemble(
    mean: float, sem: float, *, passed: bool = True, n_reports: int = 3
) -> dict[str, object]:
    return {
        "case": f"ensemble_mean_{mean}",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "n_reports": n_reports,
        },
    }


def test_nonlinear_landscape_admission_selects_uncertainty_resolved_candidate() -> None:
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [
            _ensemble(6.274543846475065, 0.04213243251063571),
            _ensemble(6.42653555490751, 0.04399590111876854),
        ],
        candidate_labels=("+3%", "+6%"),
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "+3%"
    assert report["selected_candidate"]["relative_reduction"] > 0.26
    assert report["selected_candidate"]["uncertainty_z_score"] > 17.0
    assert all(row["admitted"] for row in report["candidates"])
    assert (
        gkx.build_nonlinear_landscape_admission_report
        is build_nonlinear_landscape_admission_report
    )
    assert (
        build_nonlinear_landscape_admission_report
        is transport_reports.build_nonlinear_landscape_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_nonlinear_landscape_admission_fails_closed_for_noisy_or_unresolved_candidates() -> (
    None
):
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.5),
        [
            _ensemble(7.95, 0.5),
            _ensemble(6.0, 2.0, n_reports=2),
            _ensemble(5.0, 0.1, passed=False),
        ],
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.2,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is False
    assert report["selected_candidate"] is None
    blockers = [set(row["admission_blockers"]) for row in report["candidates"]]
    assert "insufficient_relative_reduction" in blockers[0]
    assert "insufficient_uncertainty_separation" in blockers[0]
    assert "candidate_combined_sem_rel_too_large" in blockers[1]
    assert "candidate_insufficient_replicates" in blockers[1]
    assert "candidate_ensemble_failed" in blockers[2]


@pytest.mark.parametrize("invalid", ["false", "true", 1, None])
def test_nonlinear_landscape_admission_rejects_nonboolean_pass_flags(invalid) -> None:
    baseline = _ensemble(8.0, 0.1)
    candidate = _ensemble(6.0, 0.1)
    baseline["passed"] = invalid

    report = build_nonlinear_landscape_admission_report(baseline, [candidate])

    assert report["passed"] is False
    assert "baseline_ensemble_failed" in report["candidates"][0]["admission_blockers"]


@pytest.mark.parametrize("invalid", ["three", 2.5, -3, float("nan")])
def test_nonlinear_landscape_admission_rejects_invalid_replicate_counts(
    invalid,
) -> None:
    candidate = _ensemble(6.0, 0.1)
    candidate["statistics"]["n_reports"] = invalid

    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.1), [candidate]
    )

    assert report["passed"] is False
    assert (
        "candidate_insufficient_replicates"
        in report["candidates"][0]["admission_blockers"]
    )


def test_nonlinear_landscape_admission_validates_candidate_labels() -> None:
    try:
        build_nonlinear_landscape_admission_report(
            _ensemble(8.0, 0.1),
            [_ensemble(7.0, 0.1)],
            candidate_labels=("one", "two"),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched candidate labels were accepted")


def test_reduced_nonlinear_audit_prelaunch_passes_calibrated_landscape_margin() -> None:
    baseline = 0.06558065223919245
    candidate = 0.06251277500404685

    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=baseline,
        candidate_metric=candidate,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )

    assert report["passed"] is True
    assert report["relative_reduced_reduction"] > 0.046
    assert report["required_relative_reduced_reduction"] == 0.04
    assert report["blockers"] == []
    assert report["gates"][0]["passed"] is True
    assert report["reduced_cross_sample_statistics"]["passed"] is True
    assert report["gates"][2]["metric"] == "reduced_cross_sample_dispersion"
    assert gkx.VMEXReducedPrelaunchPolicy is VMEXReducedPrelaunchPolicy
    assert (
        gkx.build_reduced_nonlinear_audit_prelaunch_report
        is build_reduced_nonlinear_audit_prelaunch_report
    )
    assert (
        build_reduced_nonlinear_audit_prelaunch_report
        is transport_reports.build_reduced_nonlinear_audit_prelaunch_report
    )


def test_reduced_nonlinear_audit_prelaunch_blocks_weak_failed_transfer_margin() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.08010670290,
        candidate_metric=0.07827418221,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMEXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            failed_reference_safety_factor=1.5,
        ),
    )

    assert report["passed"] is False
    assert "insufficient_reduced_margin_for_nonlinear_audit" in report["blockers"]
    assert (
        report["relative_reduced_reduction"]
        < report["required_relative_reduced_reduction"]
    )


def test_reduced_prelaunch_blocks_excessive_reduced_cross_sample_spread() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.067,
            "weighted_standard_error": 0.03,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.064,
            "weighted_standard_error": 0.04,
        },
        policy=VMEXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            maximum_cross_sample_sem_rel=0.35,
        ),
    )

    assert report["passed"] is False
    assert "candidate_cross_sample_sem_rel_too_large" in report["blockers"]
    assert report["gates"][2]["passed"] is False


def test_campaign_admission_combines_reduced_and_replicated_landscape_gates() -> None:
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [_ensemble(6.274543846475065, 0.04213243251063571)],
        candidate_labels=("+3% RBC(0,1)",),
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
    )

    assert report["campaign_admitted"] is True
    assert report["blockers"] == []
    assert report["selected_landscape_candidate"]["label"] == "+3% RBC(0,1)"
    assert report["claim_scope"].startswith(
        "next nonlinear optimizer-campaign admission"
    )
    assert gkx.VMEXNonlinearCampaignPolicy is VMEXNonlinearCampaignPolicy
    assert (
        gkx.build_nonlinear_campaign_admission_report
        is build_nonlinear_campaign_admission_report
    )
    assert (
        build_nonlinear_campaign_admission_report
        is transport_reports.build_nonlinear_campaign_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_campaign_admission_fails_closed_without_cross_sample_gate_or_landscape_margin() -> (
    None
):
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=1.0,
        candidate_metric=0.95,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.3),
        [_ensemble(7.4, 0.3)],
        candidate_labels=("weak",),
        policy=VMEXNonlinearAuditPolicy(minimum_relative_reduction=0.02),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
        policy=VMEXNonlinearCampaignPolicy(
            minimum_landscape_relative_reduction=0.10,
            minimum_landscape_uncertainty_z_score=3.0,
        ),
    )

    assert report["campaign_admitted"] is False
    assert "reduced_cross_sample_statistics_missing" in report["blockers"]
    assert "selected_landscape_reduction_too_small" in report["blockers"]
    assert "selected_landscape_uncertainty_separation_too_small" in report["blockers"]


def test_campaign_admission_rejects_corrupt_persisted_gate_fields() -> None:
    prelaunch = {
        "passed": "true",
        "objective_sample_summary": {"passed": "true", "sample_count": 18},
        "reduced_cross_sample_statistics": {
            "available": "true",
            "passed": "true",
            "rows": [],
        },
    }
    landscape = {
        "passed": "true",
        "selected_candidate": {
            "relative_reduction": 0.2,
            "uncertainty_z_score": 4.0,
            "combined_sem_rel": 0.01,
            "n_reports": "three",
        },
    }

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
    )

    assert report["campaign_admitted"] is False
    assert "reduced_prelaunch_gate_failed" in report["blockers"]
    assert "reduced_objective_sample_coverage_failed" in report["blockers"]
    assert "reduced_cross_sample_statistics_missing" in report["blockers"]
    assert "replicated_landscape_admission_failed" in report["blockers"]
    assert "selected_landscape_insufficient_replicates" in report["blockers"]


def test_transport_sample_summary_requires_surface_alpha_and_ky_coverage() -> None:
    summary = transport_objective_sample_summary(
        {"surfaces": [0.5], "alphas": [0.0], "ky_values": [0.3]}
    )

    assert summary["passed"] is False
    assert summary["sample_count"] == 1
    assert "insufficient_surface_coverage" in summary["blockers"]
    assert "insufficient_field_line_coverage" in summary["blockers"]
    assert "insufficient_ky_coverage" in summary["blockers"]


def test_nonlinear_audit_redesign_blocks_negative_transfer_and_recommends_multisample_design() -> (
    None
):
    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=-0.00585, z_score=-0.20, passed=False),
        objective_sample_set={"surfaces": [0.64], "alphas": [0.0], "ky_values": [0.3]},
    )

    assert report["nonlinear_audit_promoted"] is False
    assert report["requires_objective_redesign"] is True
    assert "insufficient_matched_reduction" in report["blockers"]
    assert "insufficient_uncertainty_separation" in report["blockers"]
    assert "insufficient_total_sample_count" in report["blockers"]
    assert report["recommended_sample_set"]["sample_count"] == 18
    assert report["gates"][0]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_nonlinear_audit_redesign_promotes_only_when_audit_and_sample_coverage_pass() -> (
    None
):
    policy = VMEXNonlinearAuditPolicy(
        minimum_relative_reduction=0.02,
        minimum_uncertainty_z_score=1.0,
        minimum_surface_count=3,
        minimum_alpha_count=2,
        minimum_ky_count=3,
        minimum_sample_count=12,
    )
    sample_set = {
        "surfaces": [0.45, 0.64, 0.78],
        "alphas": [0.0, 0.7853981633974483],
        "ky_values": [0.1, 0.3, 0.5],
    }

    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=0.08, z_score=2.5, passed=True),
        objective_sample_set=sample_set,
        policy=policy,
    )

    assert report["nonlinear_audit_promoted"] is True
    assert report["requires_objective_redesign"] is False
    assert report["blockers"] == []
    assert report["objective_sample_summary"]["sample_count"] == 18
    assert all(gate["passed"] for gate in report["gates"])
    assert gkx.VMEXNonlinearAuditPolicy is VMEXNonlinearAuditPolicy
    assert (
        gkx.build_nonlinear_audit_redesign_report
        is build_nonlinear_audit_redesign_report
    )
    assert (
        build_nonlinear_audit_redesign_report
        is transport_reports.build_nonlinear_audit_redesign_report
    )
    assert (
        gkx.transport_objective_sample_summary
        is transport_objective_sample_summary
    )


def test_nonlinear_audit_redesign_rejects_nonboolean_persisted_pass_flags() -> None:
    comparison = _matched_comparison(
        relative_reduction=0.08,
        z_score=2.5,
        passed=True,
    )
    comparison["passed"] = "true"
    comparison["baseline"]["passed"] = "true"

    report = build_nonlinear_audit_redesign_report(
        comparison,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, np.pi / 4.0],
            "ky_values": [0.1, 0.3, 0.5],
        },
    )

    assert report["nonlinear_audit_promoted"] is False
    assert "baseline_ensemble_failed" in report["blockers"]
    assert "matched_comparison_not_passed" in report["blockers"]


def test_transport_sample_summary_rejects_ky_values_not_supported_by_single_solver_grid() -> (
    None
):
    summary = transport_objective_sample_summary(
        {
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.19, 0.3, 0.476],
        }
    )

    assert summary["passed"] is False
    assert "ky_values_not_single_grid_compatible" in summary["blockers"]


# ---- test_vmex_transport_gradient.py ----

from dataclasses import dataclass


from gkx.objectives.vmec_transport_optimization import (
    boundary_spec_record,
    build_boundary_transport_gradient_report,
    write_boundary_transport_gradient_report,
)


@dataclass(frozen=True)
class FakeSpec:
    name: str
    kind: str
    index: int
    m: int
    n: int


class FakeOptimizer:
    _specs = (
        FakeSpec("rc01", "rc", 0, 0, 1),
        FakeSpec("zs10", "zs", 1, 1, 0),
        FakeSpec("rc11", "rc", 2, 1, 1),
    )

    def residual_fun(self, params):
        params = np.asarray(params, dtype=float)
        return np.asarray([0.4 + params[0] - 2.0 * params[1]])

    def objective_and_gradient_fun(self, params):
        residual = self.residual_fun(params)[0]
        jac = np.asarray([1.0, -2.0, 0.0])
        return 0.5 * residual**2, residual * jac

    def jacobian_fun(self, params):
        return np.asarray([[1.0, -2.0, 0.0]])


def test_boundary_spec_record_uses_vmex_fields() -> None:
    row = boundary_spec_record(FakeSpec("zs10", "zs", 4, 1, 0), fallback_index=9)

    assert row == {
        "name": "zs10",
        "kind": "zs",
        "mode_index": 4,
        "m": 1,
        "n": 0,
    }


def test_transport_gradient_report_ranks_boundary_directions() -> None:
    report = build_boundary_transport_gradient_report(
        FakeOptimizer(),
        top_n=2,
        include_jacobian=True,
    )

    assert report["finite"] is True
    assert report["transport_sensitivity_detected"] is True
    assert report["classification"] == "sensitive_boundary_transport_objective"
    assert report["parameter_count"] == 3
    assert report["residual_count"] == 1
    assert report["objective_value"] == pytest.approx(0.08)
    assert report["gradient_norm_l2"] == pytest.approx(np.sqrt(0.16 + 0.64))
    assert [row["name"] for row in report["top_gradient_components"]] == [
        "zs10",
        "rc01",
    ]
    assert report["jacobian"]["available"] is True
    assert report["jacobian"]["shape"] == [1, 3]
    assert report["jacobian"]["frobenius_norm"] == pytest.approx(np.sqrt(5.0))
    json.dumps(report, allow_nan=False)


def test_transport_gradient_report_classifies_flat_response() -> None:
    class FlatOptimizer(FakeOptimizer):
        def residual_fun(self, params):
            return np.asarray([0.0])

        def objective_and_gradient_fun(self, params):
            return 0.0, np.zeros(3)

    report = build_boundary_transport_gradient_report(
        FlatOptimizer(),
        sensitivity_atol=1.0e-10,
    )

    assert report["finite"] is True
    assert report["transport_sensitivity_detected"] is False
    assert (
        report["classification"]
        == "locally_flat_or_underconditioned_transport_objective"
    )
    assert "do not launch another blind scalar-weight ladder" in report["next_action"]


def test_transport_gradient_report_requires_params_without_specs() -> None:
    class NoSpecOptimizer:
        def residual_fun(self, params):
            return np.asarray([1.0])

        def objective_and_gradient_fun(self, params):
            return 0.5, np.ones(1)

    with pytest.raises(ValueError, match="params must be provided"):
        build_boundary_transport_gradient_report(NoSpecOptimizer())


def test_transport_gradient_report_writer_and_public_api(tmp_path) -> None:
    assert (
        gkx.build_boundary_transport_gradient_report
        is build_boundary_transport_gradient_report
    )
    report = build_boundary_transport_gradient_report(FakeOptimizer(), top_n=1)
    out = write_boundary_transport_gradient_report(report, tmp_path / "gradient.json")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "vmex_transport_gradient_diagnostic"
    assert payload["top_gradient_components"][0]["name"] == "zs10"


# ---- test_vmex_transport_line_search.py ----


from gkx.objectives.vmec_transport_optimization import (
    ProjectedLineSearchPolicy,
    boundary_chain_accepted_parameter_indices,
    projected_line_search_input_manifest,
    select_projected_line_search_candidate,
    sparse_descent_direction_from_gradient_report,
)


def _gradient_report() -> dict[str, object]:
    return {
        "parameter_count": 4,
        "top_gradient_components": [
            {"parameter_index": 1, "gradient": -3.0, "name": "zs10"},
            {"parameter_index": 3, "gradient": 4.0, "name": "rc11"},
            {"parameter_index": 0, "gradient": 12.0, "name": "rc01"},
        ],
    }


def test_sparse_descent_direction_uses_ranked_gradient_components() -> None:
    direction = sparse_descent_direction_from_gradient_report(
        _gradient_report(), top_n=2
    )

    assert direction.shape == (4,)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[1] == pytest.approx(3.0 / 5.0)
    assert direction[3] == pytest.approx(-4.0 / 5.0)
    assert direction[0] == 0.0


def test_projected_line_search_input_manifest_is_json_safe() -> None:
    manifest = projected_line_search_input_manifest(
        _gradient_report(), steps=(0.1, 0.2), top_n=2
    )

    assert manifest["kind"] == "vmex_projected_transport_line_search_input_manifest"
    assert manifest["parameter_count"] == 4
    assert manifest["direction_l2_norm"] == pytest.approx(1.0)
    assert manifest["steps"][0]["parameter_l2_norm"] == pytest.approx(0.1)
    assert manifest["steps"][0]["parameter_linf_norm"] == pytest.approx(0.08)
    json.dumps(manifest, allow_nan=False)


def test_sparse_descent_direction_rejects_bad_gradient_report() -> None:
    with pytest.raises(ValueError, match="zero descent direction"):
        sparse_descent_direction_from_gradient_report(
            {
                "parameter_count": 2,
                "top_gradient_components": [{"parameter_index": 0, "gradient": 0.0}],
            }
        )


def test_sparse_descent_direction_rejects_malformed_gradient_rows() -> None:
    with pytest.raises(ValueError, match="parameter_count must be positive"):
        sparse_descent_direction_from_gradient_report({"top_gradient_components": []})
    with pytest.raises(ValueError, match="top_gradient_components"):
        sparse_descent_direction_from_gradient_report(
            {"parameter_count": 2, "top_gradient_components": []}
        )
    with pytest.raises(ValueError, match="rows must be mappings"):
        sparse_descent_direction_from_gradient_report(
            {"parameter_count": 2, "top_gradient_components": [object()]}
        )
    with pytest.raises(ValueError, match="outside parameter_count"):
        sparse_descent_direction_from_gradient_report(
            {
                "parameter_count": 2,
                "top_gradient_components": [{"parameter_index": 3, "gradient": 1.0}],
            }
        )
    with pytest.raises(ValueError, match="not finite"):
        sparse_descent_direction_from_gradient_report(
            {
                "parameter_count": 2,
                "top_gradient_components": [{"parameter_index": 1, "gradient": "bad"}],
            }
        )


def _boundary_chain_collection() -> dict[str, object]:
    return {
        "kind": "vmex_boundary_chain_collection_summary",
        "classification": "mixed_exact_fd_consistency_with_branch_sensitive_modes",
        "rows": [
            {
                "index": 1,
                "name": "zs10",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": True,
                "frozen_axis_matches_exact_fd": True,
                "frozen_axis_convention_verified": False,
                "growth_branch_locality_checked": True,
                "growth_branch_locality_passed": True,
            },
            {
                "index": 3,
                "name": "rc11",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": True,
                "frozen_axis_matches_exact_fd": False,
                "frozen_axis_convention_verified": True,
                "growth_branch_locality_checked": True,
                "growth_branch_locality_passed": False,
            },
            {
                "index": 0,
                "name": "rc01",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": False,
                "frozen_axis_matches_exact_fd": True,
                "frozen_axis_convention_verified": False,
                "growth_branch_locality_checked": False,
                "growth_branch_locality_passed": False,
            },
        ],
    }


def test_boundary_chain_filter_keeps_only_exact_fd_consistent_components() -> None:
    collection = _boundary_chain_collection()

    assert boundary_chain_accepted_parameter_indices(collection) == (1,)
    direction = sparse_descent_direction_from_gradient_report(
        _gradient_report(),
        top_n=3,
        boundary_chain_collection=collection,
    )

    assert direction[1] == pytest.approx(1.0)
    assert direction[3] == 0.0
    assert direction[0] == 0.0


def test_boundary_chain_filter_can_admit_internal_replay_diagnostics() -> None:
    collection = _boundary_chain_collection()

    assert boundary_chain_accepted_parameter_indices(
        collection, require_exact_fd=False
    ) == (1, 3)
    direction = sparse_descent_direction_from_gradient_report(
        _gradient_report(),
        top_n=3,
        boundary_chain_collection=collection,
        require_boundary_chain_exact_fd=False,
    )

    assert direction[1] == pytest.approx(3.0 / 5.0)
    assert direction[3] == pytest.approx(-4.0 / 5.0)
    assert direction[0] == 0.0


def test_boundary_chain_filter_excludes_unverified_branch_sensitive_rows() -> None:
    collection = _boundary_chain_collection()
    rows = collection["rows"]
    assert isinstance(rows, list)
    rows[1]["frozen_axis_convention_verified"] = False

    assert boundary_chain_accepted_parameter_indices(
        collection, require_exact_fd=False
    ) == (1,)


def test_boundary_chain_filter_can_require_growth_branch_locality() -> None:
    collection = _boundary_chain_collection()

    assert boundary_chain_accepted_parameter_indices(
        collection,
        require_exact_fd=False,
        require_growth_branch_locality=True,
    ) == (1,)
    direction = sparse_descent_direction_from_gradient_report(
        _gradient_report(),
        top_n=3,
        boundary_chain_collection=collection,
        require_boundary_chain_exact_fd=False,
        require_growth_branch_locality=True,
    )

    assert direction[1] == pytest.approx(1.0)
    assert direction[3] == 0.0
    assert direction[0] == 0.0


def test_boundary_chain_filter_rejects_malformed_rows_and_skips_missing_index() -> None:
    with pytest.raises(ValueError, match="must contain rows"):
        boundary_chain_accepted_parameter_indices({"rows": None})
    with pytest.raises(ValueError, match="rows must be mappings"):
        boundary_chain_accepted_parameter_indices({"rows": [object()]})

    accepted = boundary_chain_accepted_parameter_indices(
        {
            "rows": [
                {"name": "missing-index", "frozen_axis_jvp_vjp_consistent": True},
                {
                    "index": 2,
                    "frozen_axis_jvp_vjp_consistent": True,
                    "frozen_axis_matches_exact_fd": True,
                },
                {
                    "index": 2,
                    "frozen_axis_jvp_vjp_consistent": True,
                    "frozen_axis_matches_exact_fd": True,
                },
            ]
        }
    )

    assert accepted == (2,)


def test_projected_line_search_manifest_records_boundary_chain_filter() -> None:
    manifest = projected_line_search_input_manifest(
        _gradient_report(),
        steps=(0.1,),
        top_n=3,
        boundary_chain_collection=_boundary_chain_collection(),
        require_growth_branch_locality=True,
    )

    assert manifest["boundary_chain_filter"] == {
        "enabled": True,
        "require_exact_fd": True,
        "require_frozen_axis_convention_when_exact_fd_missing": False,
        "require_growth_branch_locality": True,
        "collection_classification": "mixed_exact_fd_consistency_with_branch_sensitive_modes",
        "accepted_parameter_indices": [1],
    }
    assert manifest["steps"][0]["parameter_linf_norm"] == pytest.approx(0.1)
    json.dumps(manifest, allow_nan=False)


def test_projected_line_search_manifest_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        projected_line_search_input_manifest(_gradient_report(), steps=(0.0,), top_n=2)


def test_projected_line_search_admission_selects_best_gate_passing_candidate() -> None:
    report = select_projected_line_search_candidate(
        {"transport_metric_final": 10.0, "gate_passed": True},
        [
            {
                "label": "small",
                "step": 0.1,
                "transport_metric_final": 9.5,
                "gate_passed": True,
            },
            {
                "label": "failed",
                "step": 0.2,
                "transport_metric_final": 8.0,
                "gate_passed": False,
            },
            {
                "label": "best",
                "step": 0.15,
                "transport_metric_final": 9.0,
                "gate_passed": True,
            },
        ],
        policy=ProjectedLineSearchPolicy(minimum_relative_improvement=0.02),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "best"
    failed = next(row for row in report["candidates"] if row["label"] == "failed")
    assert "gate_failed" in failed["admission_blockers"]
    assert failed["admitted"] is False
    json.dumps(report, allow_nan=False)


def test_projected_line_search_admission_fails_closed_without_improvement() -> None:
    report = select_projected_line_search_candidate(
        {"transport_metric_final": 10.0, "gate_passed": True},
        [
            {
                "label": "worse",
                "step": 0.1,
                "transport_metric_final": 10.1,
                "gate_passed": True,
            }
        ],
    )

    assert report["passed"] is False
    assert report["selected_candidate"] is None
    assert (
        "insufficient_transport_improvement"
        in report["candidates"][0]["admission_blockers"]
    )


def test_projected_line_search_admission_reports_missing_metrics_and_higher_is_better() -> (
    None
):
    missing = select_projected_line_search_candidate(
        {},
        [{"label": "candidate", "step": 0.1, "gate_passed": True}],
    )
    assert missing["passed"] is False
    blockers = missing["candidates"][0]["admission_blockers"]
    assert "missing_transport_metric" in blockers
    assert "missing_baseline_transport_metric" in blockers

    higher = select_projected_line_search_candidate(
        {"transport_metric_final": 10.0, "gate_passed": True},
        [
            {
                "label": "higher",
                "step": 0.1,
                "gkx_objective_final": 11.0,
                "gate_passed": False,
            }
        ],
        policy=ProjectedLineSearchPolicy(
            minimum_relative_improvement=0.05,
            lower_is_better=False,
            require_gate_passed=False,
        ),
    )
    assert higher["passed"] is True
    assert higher["selected_candidate"]["label"] == "higher"
    assert higher["selected_candidate"][
        "relative_transport_improvement"
    ] == pytest.approx(0.1)


def test_projected_line_search_public_api_exports() -> None:
    assert gkx.ProjectedLineSearchPolicy is ProjectedLineSearchPolicy
    assert (
        gkx.boundary_chain_accepted_parameter_indices
        is boundary_chain_accepted_parameter_indices
    )
    assert (
        gkx.sparse_descent_direction_from_gradient_report
        is sparse_descent_direction_from_gradient_report
    )
    assert (
        gkx.projected_line_search_input_manifest
        is projected_line_search_input_manifest
    )
    assert (
        gkx.select_projected_line_search_candidate
        is select_projected_line_search_candidate
    )


# ---- test_vmex_transport_objective.py ----

"""Tests for VMEC-JAX to GKX transport objective plumbing."""


from types import ModuleType

import jax.numpy as jnp

from gkx import (
    StellaratorITGSampleSet,
    VMEXGKXTransportObjective,
    VMEXTransportObjectiveConfig,
    vmex_transport_growth_branch_locality_report_from_states,
    vmex_transport_objective_from_state,
)
from gkx.objectives.core import SOLVER_OBJECTIVE_NAMES
import gkx.objectives.vmec_transport_branch as transport_branch
import gkx.objectives.vmec_transport as transport_config
import gkx.objectives.vmec_transport as transport_tables


def _fake_geometry() -> SimpleNamespace:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    return SimpleNamespace(
        theta=theta,
        bmag_profile=1.0 + 0.05 * jnp.cos(theta),
        jacobian_profile=jnp.ones_like(theta),
        gds2_profile=1.2 + 0.1 * jnp.cos(theta),
        gds21_profile=0.05 * jnp.sin(theta),
        gds22_profile=1.0 + 0.08 * jnp.cos(2.0 * theta),
        cv_profile=0.03 * jnp.sin(theta),
        gb_profile=0.04 * jnp.cos(theta),
        cv0_profile=0.02 * jnp.sin(2.0 * theta),
        gb0_profile=0.02 * jnp.cos(2.0 * theta),
    )


def _fake_solver_rows(scale: float = 1.0) -> jnp.ndarray:
    rows = []
    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    for gamma in (0.08, 0.10, 0.12, 0.14):
        row = np.zeros(len(SOLVER_OBJECTIVE_NAMES), dtype=float)
        row[idx["gamma"]] = scale * gamma
        row[idx["omega"]] = -0.2
        row[idx["kperp_eff2"]] = 0.42
        row[idx["linear_heat_flux_weight"]] = 1.5
        row[idx["linear_particle_flux_weight"]] = 0.3
        row[idx["mixing_length_heat_flux_proxy"]] = scale * 0.04
        rows.append(row)
    return jnp.asarray(rows)


def test_vmex_transport_objective_reduces_fake_solver_rows(monkeypatch) -> None:

    calls: list[dict[str, object]] = []
    growth_calls: list[dict[str, object]] = []
    rows = _fake_solver_rows()
    row_counter = {"i": 0}

    def fake_geom(state, static, indata, wout, **kwargs):
        calls.append(
            {"state": state, "static": static, "indata": indata, "wout": wout, **kwargs}
        )
        return _fake_geometry()

    def fake_growth(_geom, **kwargs):
        growth_calls.append(kwargs)
        value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
        row_counter["i"] += 1
        return value

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4)
    )
    cfg = VMEXTransportObjectiveConfig(kind="growth", sample_set=samples, ny=4)

    value = vmex_transport_objective_from_state(
        object(),
        object(),
        object(),
        SimpleNamespace(signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])),
        cfg,
    )

    assert np.isclose(float(value), np.mean([0.08, 0.10, 0.12, 0.14]))
    assert calls[0]["mboz"] == 21
    assert calls[0]["nboz"] == 21
    assert [call["torflux"] for call in calls] == list(samples.surfaces)
    assert [call["selected_ky_index"] for call in growth_calls] == [1, 2, 1, 2]
    assert np.isclose(growth_calls[0]["ly"], 2.0 * np.pi / min(samples.ky_values))
    assert int(growth_calls[0]["ny"]) >= 6


def test_vmex_transport_surface_chunking_matches_unchunked_weighted_mean(
    monkeypatch,
) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    rows = _fake_solver_rows()

    def evaluate(*, chunk_size: int) -> float:
        row_counter = {"i": 0}

        def fake_growth(_geom, **_kwargs):
            value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
            row_counter["i"] += 1
            return value

        monkeypatch.setattr(
            transport_tables, "solver_growth_rate_from_geometry", fake_growth
        )
        samples = StellaratorITGSampleSet(
            surfaces=(0.5, 0.7),
            alphas=(0.0,),
            ky_values=(0.2, 0.4),
            surface_weights=(3.0, 1.0),
        )
        cfg = VMEXTransportObjectiveConfig(
            kind="growth",
            sample_set=samples,
            ny=4,
            objective_transform="log1p",
            surface_chunk_size=chunk_size,
        )
        value = vmex_transport_objective_from_state(
            object(),
            object(),
            object(),
            SimpleNamespace(
                signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])
            ),
            cfg,
        )
        assert row_counter["i"] == 4
        return float(value)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )

    assert evaluate(chunk_size=1) == pytest.approx(evaluate(chunk_size=0))


def test_vmex_transport_growth_branch_locality_report_accepts_consistent_branch(
    monkeypatch,
) -> None:

    def fake_geom(state, *_args, **kwargs):
        return SimpleNamespace(state=state, theta=jnp.ones(2), kwargs=kwargs)

    def fake_matrix(geom, **_kwargs):
        if geom.state == "base":
            return jnp.diag(jnp.asarray([1.0 + 0.0j, 0.5 + 0.0j]))
        if geom.state == "plus":
            return jnp.diag(jnp.asarray([1.02 + 0.0j, 0.48 + 0.0j]))
        return jnp.diag(jnp.asarray([0.98 + 0.0j, 0.52 + 0.0j]))

    monkeypatch.setattr(
        transport_branch, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_branch, "solver_linear_operator_matrix_from_geometry", fake_matrix
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMEXTransportObjectiveConfig(kind="growth", sample_set=samples)

    report = vmex_transport_growth_branch_locality_report_from_states(
        "base",
        "plus",
        "minus",
        "static",
        "indata",
        object(),
        cfg,
        step=1.0e-2,
    )

    assert (
        gkx.vmex_transport_growth_branch_locality_report_from_states
        is vmex_transport_growth_branch_locality_report_from_states
    )
    assert report["passed"] is True
    assert (
        report["classification"]
        == "all_samples_dominant_growth_branch_locally_consistent"
    )
    assert report["sample_count"] == 1
    assert report["evaluated_sample_count"] == 1
    assert report["rows"][0]["classification"] == "dominant_branch_locally_consistent"


def test_vmex_transport_growth_branch_locality_report_fails_on_branch_switch(
    monkeypatch,
) -> None:

    def fake_geom(state, *_args, **kwargs):
        return SimpleNamespace(state=state, theta=jnp.ones(2), kwargs=kwargs)

    def fake_matrix(geom, **_kwargs):
        if geom.state == "base":
            return jnp.diag(jnp.asarray([1.0 + 0.0j, 0.8 + 0.0j]))
        if geom.state == "plus":
            return jnp.diag(jnp.asarray([1.02 + 0.0j, 1.05 + 0.0j]))
        return jnp.diag(jnp.asarray([0.98 + 0.0j, 0.65 + 0.0j]))

    monkeypatch.setattr(
        transport_branch, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_branch, "solver_linear_operator_matrix_from_geometry", fake_matrix
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMEXTransportObjectiveConfig(kind="growth", sample_set=samples)

    report = vmex_transport_growth_branch_locality_report_from_states(
        "base",
        "plus",
        "minus",
        "static",
        "indata",
        object(),
        cfg,
        step=1.0e-2,
    )

    assert report["passed"] is False
    assert report["classification"] == "growth_branch_locality_failed_or_incomplete"
    assert report["blockers"] == ["branch_locality_mismatch_or_underisolated"]
    assert (
        report["rows"][0]["classification"]
        == "dominant_branch_differs_from_nearest_branch"
    )


def test_vmex_transport_objective_nonlinear_proxy_is_positive_and_exported(
    monkeypatch,
) -> None:

    scale = {"value": 1.0}

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.1 * scale["value"])

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4)
    )
    cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux", sample_set=samples
    )

    low = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )
    scale["value"] = 2.0
    high = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )

    assert gkx.VMEXTransportObjectiveConfig is VMEXTransportObjectiveConfig
    assert (
        gkx.VMEXGKXTransportObjective
        is VMEXGKXTransportObjective
    )
    assert float(low) > 0.0
    assert float(high) > float(low)


def test_vmex_transport_objective_transform_scales_large_residuals(
    monkeypatch,
) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(20.0)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    raw_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="raw",
    )
    scaled_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="scaled",
        objective_scale=10.0,
    )
    log_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="log1p",
        objective_scale=10.0,
    )

    raw = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), raw_cfg
    )
    scaled = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), scaled_cfg
    )
    logged = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), log_cfg
    )

    assert float(raw) > 1.0
    assert float(scaled) == pytest.approx(float(raw) / 10.0)
    assert float(logged) == pytest.approx(float(jnp.log1p(jnp.abs(scaled))))
    assert float(logged) < float(scaled)


def test_vmex_transport_objective_vmec_callback_builds_reference_wout(
    monkeypatch,
) -> None:
    import gkx.objectives.vmec_transport as mod

    captured: dict[str, object] = {}

    def fake_eval(state, static, indata, wout_reference, config):
        captured["state"] = state
        captured["static"] = static
        captured["indata"] = indata
        captured["wout"] = wout_reference
        captured["config"] = config
        return jnp.asarray(0.125)

    monkeypatch.setattr(mod, "vmex_transport_objective_from_state", fake_eval)
    objective = VMEXGKXTransportObjective()
    ctx = SimpleNamespace(
        static=SimpleNamespace(cfg=SimpleNamespace(nfp=3)), indata="indata", signgs=-1
    )

    value = objective.J(ctx, "state")

    assert float(value) == 0.125
    assert captured["state"] == "state"
    assert captured["indata"] == "indata"
    assert captured["wout"].nfp == 3
    assert captured["wout"].signgs == -1


def test_vmex_transport_config_rejects_underresolved_boozer_modes() -> None:
    assert (
        VMEXTransportObjectiveConfig(kind="growth").gradient_scope
        == "eigenvalue_growth_ad"
    )
    assert (
        VMEXTransportObjectiveConfig(kind="quasilinear_flux").gradient_scope
        == "eigenvalue_growth_ad_with_geometry_transport_weights"
    )
    try:
        VMEXTransportObjectiveConfig(mboz=12, nboz=21)
    except ValueError as exc:
        assert "at least 21" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("underresolved Boozer mode count should fail")
    with pytest.raises(ValueError, match="objective_scale"):
        VMEXTransportObjectiveConfig(objective_scale=0.0)
    with pytest.raises(ValueError, match="objective transform"):
        VMEXTransportObjectiveConfig(objective_transform="bad")  # type: ignore[arg-type]


def test_vmex_transport_objective_pins_imported_backend_paths(
    monkeypatch, tmp_path
) -> None:

    vmec_root = tmp_path / "vmex_repo"
    vmec_pkg = vmec_root / "vmex"
    vmec_pkg.mkdir(parents=True)
    vmec_file = vmec_pkg / "__init__.py"
    vmec_file.write_text("", encoding="utf-8")

    booz_root = tmp_path / "booz_xform_jax_repo" / "src"
    booz_pkg = booz_root / "booz_xform_jax"
    booz_pkg.mkdir(parents=True)
    booz_file = booz_pkg / "__init__.py"
    booz_file.write_text("", encoding="utf-8")

    vmec_module = ModuleType("vmex")
    vmec_module.__file__ = str(vmec_file)
    booz_module = ModuleType("booz_xform_jax")
    booz_module.__file__ = str(booz_file)
    monkeypatch.setitem(sys.modules, "vmex", vmec_module)
    monkeypatch.setitem(sys.modules, "booz_xform_jax", booz_module)
    monkeypatch.delenv("GKX_VMEX_PATH", raising=False)
    monkeypatch.delenv("VMEX_PATH", raising=False)
    monkeypatch.delenv("GKX_BOOZ_XFORM_JAX_PATH", raising=False)
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    transport_config._pin_current_optional_backend_paths()

    assert str(vmec_root) == transport_config.os.environ["GKX_VMEX_PATH"]
    assert str(booz_root) == transport_config.os.environ["GKX_BOOZ_XFORM_JAX_PATH"]


def test_module_search_root_handles_paths_and_missing_modules(
    monkeypatch, tmp_path
) -> None:

    namespace_root = tmp_path / "namespace_backend"
    namespace_root.mkdir()
    namespace_module = ModuleType("namespace_backend")
    namespace_module.__path__ = [str(namespace_root)]

    missing_path_module = ModuleType("missing_path_backend")
    missing_path_module.__path__ = [str(tmp_path / "does_not_exist")]

    no_path_module = ModuleType("no_path_backend")

    monkeypatch.setitem(sys.modules, "namespace_backend", namespace_module)
    monkeypatch.setitem(sys.modules, "missing_path_backend", missing_path_module)
    monkeypatch.setitem(sys.modules, "no_path_backend", no_path_module)

    assert transport_config._module_search_root(
        "namespace_backend"
    ) == namespace_root.resolve(strict=False)
    assert transport_config._module_search_root("missing_path_backend") is None
    assert transport_config._module_search_root("no_path_backend") is None
    assert (
        transport_config._module_search_root("gkx_missing_backend_for_test")
        is None
    )


def test_pin_current_optional_backend_paths_respects_explicit_environment(
    monkeypatch,
) -> None:

    def unexpected_search(module_name: str):
        raise AssertionError(f"backend search should be skipped for {module_name}")

    monkeypatch.setattr(transport_config, "_module_search_root", unexpected_search)
    monkeypatch.delenv("GKX_VMEX_PATH", raising=False)
    monkeypatch.setenv("VMEX_PATH", "/explicit/vmec-jax")
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", "/explicit/booz-xform-jax")
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    transport_config._pin_current_optional_backend_paths()

    assert "GKX_VMEX_PATH" not in transport_config.os.environ
    assert transport_config.os.environ["VMEX_PATH"] == "/explicit/vmec-jax"
    assert (
        transport_config.os.environ["GKX_BOOZ_XFORM_JAX_PATH"]
        == "/explicit/booz-xform-jax"
    )


def test_static_grid_options_maps_integer_ky_multiples_to_solver_grid() -> None:

    options = transport_tables._static_grid_options_from_ky_values(
        (0.15, 0.45), min_ny=12
    )

    assert options["ky_base"] == pytest.approx(0.15)
    assert options["ly"] == pytest.approx(2.0 * np.pi / 0.15)
    assert options["ny"] == 12
    assert options["selected_ky_indices"] == (1, 3)


@pytest.mark.parametrize(
    ("ky_values", "message"),
    (
        ((), "finite non-empty vector"),
        ((0.2, np.nan), "finite non-empty vector"),
        ((0.0,), "positive"),
        ((0.2, 0.31), "integer multiples"),
        ((0.2, 0.2), "duplicate selected ky indices"),
    ),
)
def test_static_grid_options_rejects_invalid_ky_values(
    ky_values: tuple[float, ...],
    message: str,
) -> None:

    with pytest.raises(ValueError, match=message):
        transport_tables._static_grid_options_from_ky_values(ky_values, min_ny=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"kind": "invalid"}, "unknown VMEC-JAX transport objective kind"),
        ({"ntheta": 3}, "ntheta must be >= 4"),
        ({"n_laguerre": 0}, "n_laguerre and n_hermite must be positive"),
        ({"ny": 2}, "nx must be positive and ny must be at least 3"),
        ({"nonlinear_csat": 0.0}, "nonlinear_csat must be positive"),
        ({"surface_chunk_size": -1}, "surface_chunk_size must be non-negative"),
        (
            {
                "sample_set": StellaratorITGSampleSet(reduction="max"),
                "surface_chunk_size": 1,
            },
            "surface_chunk_size currently supports only mean or weighted_mean reductions",
        ),
    ),
)
def test_vmex_transport_config_rejects_invalid_edges(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VMEXTransportObjectiveConfig(**kwargs)


def test_vmex_transport_config_objective_options_filter_none_values() -> None:
    default_options = VMEXTransportObjectiveConfig().objective_options()
    configured_options = VMEXTransportObjectiveConfig(
        reference_length=2.5,
        reference_b=0.7,
        validate_finite=False,
    ).objective_options()

    assert "reference_length" not in default_options
    assert "reference_b" not in default_options
    assert configured_options["reference_length"] == 2.5
    assert configured_options["reference_b"] == 0.7
    assert configured_options["validate_finite"] is False


def test_geometry_transport_weights_use_safe_defaults_for_minimal_geometry() -> None:

    theta = jnp.linspace(-jnp.pi, jnp.pi, 6, endpoint=False)
    kperp, heat_weight, particle_weight = transport_tables._geometry_transport_weights(
        SimpleNamespace(theta=theta),
        selected_ky_index=2,
        ly=5.0,
    )

    assert np.isfinite(float(kperp))
    assert np.isfinite(float(heat_weight))
    assert np.isfinite(float(particle_weight))
    assert float(kperp) > 0.0
    assert float(heat_weight) > 0.0
    assert float(particle_weight) == pytest.approx(0.25 * float(heat_weight))


def test_transport_feature_table_rejects_empty_sample_rows() -> None:

    config = SimpleNamespace(
        sample_set=SimpleNamespace(surfaces=(), alphas=(0.0,), ky_values=(0.2,)),
        kind="growth",
    )

    with pytest.raises(RuntimeError, match="produced no sample rows"):
        transport_tables._transport_feature_table_from_state(
            "state",
            "static",
            "indata",
            object(),
            config,
            {"selected_ky_indices": (1,), "ny": 4, "ly": 2.0 * np.pi / 0.2},
        )


def test_quasilinear_flux_uses_geometry_transport_weights(monkeypatch) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.2)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMEXTransportObjectiveConfig(kind="quasilinear_flux", sample_set=samples)

    value = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )

    assert float(value) > 0.0
