"""Targeted follow-up plans for nonlinear turbulence-gradient audits.

This module turns failed long-window central finite-difference artifacts into a
bounded run prescription.  It is deliberately conservative: extra replicas are
recommended only when the finite-difference response is resolved and local, but
the propagated gradient uncertainty is slightly too large.
"""

from __future__ import annotations

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    STATE_TO_RUN_STATE as STATE_TO_RUN_STATE,
    NonlinearGradientCandidateDesignConfig as NonlinearGradientCandidateDesignConfig,
    NonlinearGradientCompositeControlConfig as NonlinearGradientCompositeControlConfig,
    NonlinearGradientControlMeanGateConfig as NonlinearGradientControlMeanGateConfig,
    NonlinearGradientControlVariateCampaignConfig as NonlinearGradientControlVariateCampaignConfig,
    NonlinearGradientFollowupConfig as NonlinearGradientFollowupConfig,
    NonlinearGradientQLSeedScreenConfig as NonlinearGradientQLSeedScreenConfig,
    NonlinearGradientStateControlRunbookConfig as NonlinearGradientStateControlRunbookConfig,
    NonlinearGradientVarianceReductionConfig as NonlinearGradientVarianceReductionConfig,
    _artifact_passed as _artifact_passed,
    _coefficient_label_from_parameter as _coefficient_label_from_parameter,
    _control_variate_candidate as _control_variate_candidate,
    _ensemble_state_variance_report as _ensemble_state_variance_report,
    _ensemble_stats_value as _ensemble_stats_value,
    _finite_float as _finite_float,
    _finite_int as _finite_int,
    _json_number as _json_number,
    _label_from_row as _label_from_row,
    _mean_and_sem as _mean_and_sem,
    _metric as _metric,
    _nested_metric as _nested_metric,
    _replicate_count as _replicate_count,
    _sample_covariance as _sample_covariance,
    _seed_numbers as _seed_numbers,
    _state_control_family as _state_control_family,
    _state_means_by_label as _state_means_by_label,
)

from spectraxgk.validation.nonlinear_gradient.followup_candidate import (
    _design_row as _design_row,
    _required_replicates_for_scaled_bracket as _required_replicates_for_scaled_bracket,
    nonlinear_gradient_candidate_design_report as nonlinear_gradient_candidate_design_report,
)
from spectraxgk.validation.nonlinear_gradient.followup_composite import (
    _composite_control_row as _composite_control_row,
    nonlinear_gradient_composite_control_report as nonlinear_gradient_composite_control_report,
)
from spectraxgk.validation.nonlinear_gradient.followup_plan import (
    _planned_matched_runs as _planned_matched_runs,
    _required_replicates as _required_replicates,
    nonlinear_gradient_followup_plan as nonlinear_gradient_followup_plan,
)
from spectraxgk.validation.nonlinear_gradient.followup_ql_seed import (
    _ql_seed_rows as _ql_seed_rows,
    _sign_consistency as _sign_consistency,
    nonlinear_gradient_ql_seed_screen_report as nonlinear_gradient_ql_seed_screen_report,
)
from spectraxgk.validation.nonlinear_gradient.followup_state_runbook import (
    _mapping_control_rows as _mapping_control_rows,
    nonlinear_gradient_state_control_runbook_report as nonlinear_gradient_state_control_runbook_report,
)

from spectraxgk.validation.nonlinear_gradient.followup_variance import (
    nonlinear_gradient_control_mean_gate as nonlinear_gradient_control_mean_gate,
    nonlinear_gradient_control_variate_campaign_plan as nonlinear_gradient_control_variate_campaign_plan,
    nonlinear_gradient_variance_reduction_plan as nonlinear_gradient_variance_reduction_plan,
)
