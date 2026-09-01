# GKX 1.8.2 public API and VMEX-use inventory

Status: Phase 0 compatibility snapshot; documentation only. This file records
the 1.8.2 facade before Phase 1 reduces it to at most 30 promoted top-level
names. Presence here does not promise that an internal or undocumented name
will survive the GKX 3 migration layer.

## Frozen contract

Baseline: GKX `4104bf4a2d7463fcd56e9c38434d88510377d2b4` and VMEX
`f7bd9469a059d2c54b6d85a125205c8c245c0a10`.

The top-level facade contains 347 names: `__version__` plus the 346 ordered
entries in `gkx.api._EXPORT_TARGETS`. The ordered name/target digest is
`d74c9ddcb00b724c2749dfeda683c8ea207b89109eb609e3d1389bbb4a929e27`.
Every registry entry appears exactly once in the table below.

`Documentation references` is an exact-token count over tracked `README.md`
and the 35 Markdown/reStructuredText files under `docs/`; it is discovery
evidence, not a quality score. `VMEX executable references` is an AST count
from `vmex/core/turbulence.py`; comments, docstrings, and tests do not count.

## Downstream compatibility findings

VMEX is the only cloned maintained downstream repository with executable GKX
imports at the frozen revisions. Its optional `turbulence` integration uses
five public facade names:

- `flux_tube_geometry_from_mapping`;
- `solver_objective_vector_from_geometry`;
- `solver_linear_operator_matrix_from_geometry`;
- `solver_scalar_objective_from_vector`;
- `VMEXTransportObjectiveConfig`.

The same module also imports two private implementation names:
`gkx.objectives.core._default_gradient_linear_params` and
`gkx.objectives.vmec_transport._solver_table_to_nonlinear_window_proxy`.
Those are explicit compatibility debt. Phase 1 must replace them with bounded
public parameter/default and nonlinear-proxy contracts or change VMEX in the
same coordinated PR sequence before the GKX internals move.

No executable GKX import was found under the frozen SOLVAX, booz_xform_jax,
GX, stella, or GS2 source trees. That does not prove there are no external
users; it bounds the local downstream evidence requested by Phase 0.

## Migration use

Phase 1 should classify—not mechanically retain—these names. A name is a
candidate for the <=30 facade only when it belongs to one obvious user workflow,
has a stable input/output contract, is documented, and has a wheel-level test.
Low-level kernels, reports, campaign builders, VMEC/Boozer implementation
helpers, and solver-specific internals should remain in deliberate subpackages
or be removed. The five VMEX public uses form a concrete compatibility set; the
two private uses require coordinated repair rather than deprecation promises.

## Complete ordered registry

| Name | Lazy target | Documentation references | VMEX executable references |
|---|---|---:|---:|
| `CycloneBaseCase` | `gkx.config:CycloneBaseCase` | 2 | 0 |
| `GridConfig` | `gkx.config:GridConfig` | 4 | 0 |
| `build_spectral_grid` | `gkx.core_grid:build_spectral_grid` | 5 | 0 |
| `KBMBaseCase` | `gkx.config:KBMBaseCase` | 0 | 0 |
| `TimeConfig` | `gkx.config:TimeConfig` | 16 | 0 |
| `SAlphaGeometry` | `gkx.geometry:SAlphaGeometry` | 5 | 0 |
| `build_flux_tube_geometry` | `gkx.geometry:build_flux_tube_geometry` | 1 | 0 |
| `booz_xform_flux_tube_mapping_from_inputs` | `gkx.geometry.differentiable:booz_xform_flux_tube_mapping_from_inputs` | 0 | 0 |
| `booz_xform_flux_tube_sensitivity_report` | `gkx.geometry.differentiable:booz_xform_flux_tube_sensitivity_report` | 0 | 0 |
| `booz_xform_spectral_sensitivity_report` | `gkx.geometry.differentiable:booz_xform_spectral_sensitivity_report` | 0 | 0 |
| `discover_differentiable_geometry_backends` | `gkx.geometry.differentiable:discover_differentiable_geometry_backends` | 1 | 0 |
| `evaluate_boozer_bmag_on_field_line` | `gkx.geometry.differentiable:evaluate_boozer_bmag_on_field_line` | 0 | 0 |
| `finite_difference_jacobian` | `gkx.geometry.differentiable:finite_difference_jacobian` | 0 | 0 |
| `flux_tube_geometry_from_mapping` | `gkx.geometry.differentiable:flux_tube_geometry_from_mapping` | 3 | 1 |
| `flux_tube_geometry_from_vmec_boozer_state` | `gkx.geometry.differentiable:flux_tube_geometry_from_vmec_boozer_state` | 2 | 0 |
| `flux_tube_geometry_observables` | `gkx.geometry.differentiable:flux_tube_geometry_observables` | 0 | 0 |
| `geometry_inverse_design_report` | `gkx.geometry.differentiable:geometry_inverse_design_report` | 1 | 0 |
| `geometry_observable_names` | `gkx.geometry.differentiable:geometry_observable_names` | 0 | 0 |
| `geometry_sensitivity_report` | `gkx.geometry.differentiable:geometry_sensitivity_report` | 1 | 0 |
| `vmex_boozer_equal_arc_core_profiles_from_state` | `gkx.geometry.differentiable:vmex_boozer_equal_arc_core_profiles_from_state` | 0 | 0 |
| `vmex_boozer_flux_tube_sensitivity_report` | `gkx.geometry.differentiable:vmex_boozer_flux_tube_sensitivity_report` | 0 | 0 |
| `vmex_field_line_tensor_sensitivity_report` | `gkx.geometry.differentiable:vmex_field_line_tensor_sensitivity_report` | 0 | 0 |
| `vmex_flux_tube_array_parity_report` | `gkx.geometry.differentiable:vmex_flux_tube_array_parity_report` | 0 | 0 |
| `vmex_flux_tube_mapping_from_state` | `gkx.geometry.differentiable:vmex_flux_tube_mapping_from_state` | 0 | 0 |
| `vmex_flux_tube_sensitivity_report` | `gkx.geometry.differentiable:vmex_flux_tube_sensitivity_report` | 0 | 0 |
| `vmex_metric_tensor_sensitivity_report` | `gkx.geometry.differentiable:vmex_metric_tensor_sensitivity_report` | 0 | 0 |
| `vmec_boundary_aspect_sensitivity_report` | `gkx.geometry.differentiable:vmec_boundary_aspect_sensitivity_report` | 0 | 0 |
| `vmec_field_line_tensor_observable_names` | `gkx.geometry.differentiable:vmec_field_line_tensor_observable_names` | 0 | 0 |
| `vmec_metric_tensor_observable_names` | `gkx.geometry.differentiable:vmec_metric_tensor_observable_names` | 0 | 0 |
| `J_l_all` | `gkx.core_velocity:J_l_all` | 1 | 0 |
| `gamma0` | `gkx.core_velocity:gamma0` | 0 | 0 |
| `load_runtime_from_toml` | `gkx.workflows.runtime.toml:load_runtime_from_toml` | 3 | 0 |
| `DiagnosticNorm` | `gkx.diagnostics.normalization:DiagnosticNorm` | 0 | 0 |
| `NormalizationContract` | `gkx.diagnostics.normalization:NormalizationContract` | 1 | 0 |
| `get_normalization_contract` | `gkx.diagnostics.normalization:get_normalization_contract` | 3 | 0 |
| `apply_diagnostic_normalization` | `gkx.diagnostics.normalization:apply_diagnostic_normalization` | 2 | 0 |
| `RuntimeConfig` | `gkx.workflows.runtime.config:RuntimeConfig` | 2 | 0 |
| `RuntimeSpeciesConfig` | `gkx.workflows.runtime.config:RuntimeSpeciesConfig` | 1 | 0 |
| `RuntimePhysicsConfig` | `gkx.workflows.runtime.config:RuntimePhysicsConfig` | 3 | 0 |
| `RuntimeCollisionConfig` | `gkx.workflows.runtime.config:RuntimeCollisionConfig` | 18 | 0 |
| `RuntimeNormalizationConfig` | `gkx.workflows.runtime.config:RuntimeNormalizationConfig` | 1 | 0 |
| `RuntimeOutputConfig` | `gkx.workflows.runtime.config:RuntimeOutputConfig` | 0 | 0 |
| `RuntimeParallelConfig` | `gkx.workflows.runtime.config:RuntimeParallelConfig` | 8 | 0 |
| `RuntimeQuasilinearConfig` | `gkx.workflows.runtime.config:RuntimeQuasilinearConfig` | 0 | 0 |
| `RuntimeTermsConfig` | `gkx.workflows.runtime.config:RuntimeTermsConfig` | 10 | 0 |
| `QuasilinearTransportResult` | `gkx.diagnostics.quasilinear_transport:QuasilinearTransportResult` | 0 | 0 |
| `compute_quasilinear_from_linear_state` | `gkx.diagnostics.quasilinear_transport:compute_quasilinear_from_linear_state` | 0 | 0 |
| `effective_kperp2` | `gkx.diagnostics.quasilinear_transport:effective_kperp2` | 0 | 0 |
| `mixing_length_amplitude2_jax` | `gkx.diagnostics.quasilinear_transport:mixing_length_amplitude2_jax` | 0 | 0 |
| `phi_norm2` | `gkx.diagnostics.quasilinear_transport:phi_norm2` | 0 | 0 |
| `quasilinear_feature_objective` | `gkx.diagnostics.quasilinear_transport:quasilinear_feature_objective` | 1 | 0 |
| `saturation_amplitude2` | `gkx.diagnostics.quasilinear_transport:saturation_amplitude2` | 0 | 0 |
| `saturated_flux_from_linear_weight` | `gkx.diagnostics.quasilinear_transport:saturated_flux_from_linear_weight` | 0 | 0 |
| `shape_aware_power_law_objective` | `gkx.diagnostics.quasilinear_transport:shape_aware_power_law_objective` | 0 | 0 |
| `QuasilinearCalibrationPoint` | `gkx.diagnostics.quasilinear_calibration:QuasilinearCalibrationPoint` | 0 | 0 |
| `apply_heat_flux_scale` | `gkx.diagnostics.quasilinear_calibration:apply_heat_flux_scale` | 0 | 0 |
| `calibration_point_from_nonlinear_window_summary` | `gkx.diagnostics.quasilinear_calibration:calibration_point_from_nonlinear_window_summary` | 1 | 0 |
| `calibration_point_from_spectrum_and_nonlinear_window` | `gkx.diagnostics.quasilinear_calibration:calibration_point_from_spectrum_and_nonlinear_window` | 0 | 0 |
| `fit_train_heat_flux_scale` | `gkx.diagnostics.quasilinear_calibration:fit_train_heat_flux_scale` | 0 | 0 |
| `integrated_quasilinear_flux_from_spectrum` | `gkx.diagnostics.quasilinear_calibration:integrated_quasilinear_flux_from_spectrum` | 0 | 0 |
| `quasilinear_calibration_report` | `gkx.diagnostics.quasilinear_calibration:quasilinear_calibration_report` | 1 | 0 |
| `write_quasilinear_calibration_report` | `gkx.diagnostics.quasilinear_calibration:write_quasilinear_calibration_report` | 0 | 0 |
| `build_quasilinear_model_selection_status` | `gkx.diagnostics.quasilinear_model_selection:build_quasilinear_model_selection_status` | 0 | 0 |
| `build_quasilinear_model_selection_status_from_paths` | `gkx.diagnostics.quasilinear_model_selection:build_quasilinear_model_selection_status_from_paths` | 0 | 0 |
| `NonlinearWindowConvergenceConfig` | `gkx.diagnostics.transport_windows:NonlinearWindowConvergenceConfig` | 0 | 0 |
| `NonlinearWindowEnsembleConfig` | `gkx.diagnostics.transport_windows:NonlinearWindowEnsembleConfig` | 0 | 0 |
| `nonlinear_window_convergence_from_csv` | `gkx.diagnostics.transport_windows:nonlinear_window_convergence_from_csv` | 0 | 0 |
| `nonlinear_window_convergence_from_summary` | `gkx.diagnostics.transport_windows:nonlinear_window_convergence_from_summary` | 0 | 0 |
| `nonlinear_window_convergence_report` | `gkx.diagnostics.transport_windows:nonlinear_window_convergence_report` | 0 | 0 |
| `nonlinear_window_ensemble_report` | `gkx.diagnostics.transport_windows:nonlinear_window_ensemble_report` | 2 | 0 |
| `nonlinear_window_stats_promotion_ready` | `gkx.diagnostics.transport_windows:nonlinear_window_stats_promotion_ready` | 0 | 0 |
| `matched_nonlinear_transport_report` | `gkx.diagnostics.validation_gates:matched_nonlinear_transport_report` | 1 | 0 |
| `ProductionNonlinearOptimizationGuardConfig` | `gkx.diagnostics.nonlinear_transport_optimization:ProductionNonlinearOptimizationGuardConfig` | 0 | 0 |
| `matched_optimized_transport_report` | `gkx.diagnostics.nonlinear_transport_optimization:matched_optimized_transport_report` | 0 | 0 |
| `optimized_equilibrium_transport_report` | `gkx.diagnostics.nonlinear_transport_optimization:optimized_equilibrium_transport_report` | 0 | 0 |
| `production_nonlinear_optimization_guard_report` | `gkx.diagnostics.nonlinear_transport_optimization:production_nonlinear_optimization_guard_report` | 0 | 0 |
| `reduced_artifact_scope_report` | `gkx.diagnostics.nonlinear_transport_optimization:reduced_artifact_scope_report` | 0 | 0 |
| `replicated_transport_ensemble_report` | `gkx.diagnostics.nonlinear_transport_optimization:replicated_transport_ensemble_report` | 0 | 0 |
| `RuntimeLinearResult` | `gkx.runtime:RuntimeLinearResult` | 1 | 0 |
| `RuntimeLinearScanResult` | `gkx.runtime:RuntimeLinearScanResult` | 0 | 0 |
| `RuntimeParameterScanResult` | `gkx.workflows.runtime.results:RuntimeParameterScanResult` | 0 | 0 |
| `build_runtime_linear_params` | `gkx.runtime:build_runtime_linear_params` | 0 | 0 |
| `build_runtime_linear_terms` | `gkx.runtime:build_runtime_linear_terms` | 0 | 0 |
| `build_runtime_term_config` | `gkx.runtime:build_runtime_term_config` | 0 | 0 |
| `run_linear_case` | `gkx.workflows.runtime.commands:run_linear_case` | 1 | 0 |
| `run_nonlinear_case` | `gkx.workflows.runtime.commands:run_nonlinear_case` | 1 | 0 |
| `run_runtime_linear` | `gkx.runtime:run_runtime_linear` | 4 | 0 |
| `run_runtime_nonlinear` | `gkx.runtime:run_runtime_nonlinear` | 0 | 0 |
| `run_runtime_parameter_scan` | `gkx.workflows.runtime.orchestration_scan:run_runtime_parameter_scan` | 2 | 0 |
| `refit_runtime_linear_trajectory` | `gkx.workflows.runtime.diagnostics:refit_runtime_linear_trajectory` | 1 | 0 |
| `run_runtime_scan` | `gkx.runtime:run_runtime_scan` | 3 | 0 |
| `hermite_streaming` | `gkx.operators:hermite_streaming` | 0 | 0 |
| `LinearParams` | `gkx.operators.linear.params:LinearParams` | 13 | 0 |
| `LinearTerms` | `gkx.operators.linear.params:LinearTerms` | 5 | 0 |
| `TermConfig` | `gkx.terms.config:TermConfig` | 4 | 0 |
| `LinearCache` | `gkx.operators.linear.cache_model:LinearCache` | 2 | 0 |
| `build_linear_cache` | `gkx.operators.linear.cache_builder:build_linear_cache` | 5 | 0 |
| `linear_terms_to_term_config` | `gkx.operators.linear.params:linear_terms_to_term_config` | 2 | 0 |
| `term_config_to_linear_terms` | `gkx.operators.linear.params:term_config_to_linear_terms` | 0 | 0 |
| `linear_rhs` | `gkx.operators.linear.rhs:linear_rhs` | 3 | 0 |
| `linear_rhs_cached` | `gkx.operators.linear.rhs:linear_rhs_cached` | 13 | 0 |
| `linear_rhs_electrostatic_slices_velocity_sharded` | `gkx.solvers_linear_parallel:linear_rhs_electrostatic_slices_velocity_sharded` | 0 | 0 |
| `linear_rhs_streaming_electrostatic_velocity_sharded` | `gkx.solvers_linear_parallel:linear_rhs_streaming_electrostatic_velocity_sharded` | 0 | 0 |
| `linear_rhs_parallel_cached` | `gkx.solvers_linear_parallel:linear_rhs_parallel_cached` | 3 | 0 |
| `linear_rhs_streaming_velocity_sharded` | `gkx.solvers_linear_parallel:linear_rhs_streaming_velocity_sharded` | 0 | 0 |
| `integrate_linear` | `gkx.solvers_linear_integrators:integrate_linear` | 8 | 0 |
| `KrylovConfig` | `gkx.solvers_linear_krylov:KrylovConfig` | 7 | 0 |
| `adaptive_propagator_eigenpair` | `gkx.solvers_linear_krylov:adaptive_propagator_eigenpair` | 1 | 0 |
| `dominant_eigenpair` | `gkx.solvers_linear_krylov:dominant_eigenpair` | 3 | 0 |
| `dominant_eigenvalue` | `gkx.solvers_linear_krylov:dominant_eigenvalue` | 0 | 0 |
| `integrate_linear_diffrax` | `gkx.solvers_time.diffrax_linear:integrate_linear_diffrax` | 4 | 0 |
| `integrate_linear_diffrax_streaming` | `gkx.solvers_time.diffrax_streaming:integrate_linear_diffrax_streaming` | 2 | 0 |
| `integrate_linear_sharded` | `gkx.parallel.integrators:integrate_linear_sharded` | 0 | 0 |
| `integrate_nonlinear_sharded` | `gkx.parallel.integrators:integrate_nonlinear_sharded` | 3 | 0 |
| `integrate_nonlinear` | `gkx.solvers_nonlinear_state_integration:integrate_nonlinear` | 3 | 0 |
| `integrate_nonlinear_cached` | `gkx.solvers_nonlinear_state_integration:integrate_nonlinear_cached` | 0 | 0 |
| `nonlinear_heat_flux_window` | `gkx.solvers_nonlinear_state_integration:nonlinear_heat_flux_window` | 5 | 0 |
| `DIVERGENCE_KNEE_STEPS` | `gkx.solvers_nonlinear_state_integration:DIVERGENCE_KNEE_STEPS` | 1 | 0 |
| `integrate_nonlinear_explicit_diagnostics` | `gkx.solvers_nonlinear_diagnostic_integration:integrate_nonlinear_explicit_diagnostics` | 0 | 0 |
| `integrate_nonlinear_diffrax` | `gkx.solvers_time.diffrax_nonlinear:integrate_nonlinear_diffrax` | 3 | 0 |
| `build_nonlinear_imex_operator` | `gkx.operators.nonlinear.policies:build_nonlinear_imex_operator` | 1 | 0 |
| `IMEXLinearOperator` | `gkx.operators.nonlinear.policies:IMEXLinearOperator` | 0 | 0 |
| `ExplicitTimeConfig` | `gkx.solvers_time_explicit:ExplicitTimeConfig` | 0 | 0 |
| `integrate_linear_explicit` | `gkx.solvers_time_explicit:integrate_linear_explicit` | 3 | 0 |
| `integrate_linear_explicit_diagnostics` | `gkx.solvers_time_explicit:integrate_linear_explicit_diagnostics` | 0 | 0 |
| `SimulationDiagnostics` | `gkx.diagnostics:SimulationDiagnostics` | 2 | 0 |
| `nonlinear_rhs_cached` | `gkx.solvers_nonlinear_state_integration:nonlinear_rhs_cached` | 3 | 0 |
| `integrate_linear_from_config` | `gkx.solvers_time_runners:integrate_linear_from_config` | 4 | 0 |
| `integrate_nonlinear_from_config` | `gkx.solvers_time_runners:integrate_nonlinear_from_config` | 0 | 0 |
| `Species` | `gkx.operators.linear.params:Species` | 5 | 0 |
| `build_linear_params` | `gkx.operators.linear.params:build_linear_params` | 2 | 0 |
| `fit_growth_rate` | `gkx.diagnostics.analysis:fit_growth_rate` | 1 | 0 |
| `fit_growth_rate_auto` | `gkx.diagnostics.analysis:fit_growth_rate_auto` | 2 | 0 |
| `instantaneous_growth_rate_from_phi` | `gkx.diagnostics.analysis:instantaneous_growth_rate_from_phi` | 0 | 0 |
| `growth_rate_from_phi` | `gkx.diagnostics.analysis:instantaneous_growth_rate_from_phi` | 0 | 0 |
| `select_fit_window` | `gkx.diagnostics.analysis:select_fit_window` | 1 | 0 |
| `ScanAndModeResult` | `gkx.workflows.linear:ScanAndModeResult` | 0 | 0 |
| `BranchContinuationMetrics` | `gkx.diagnostics.analysis:BranchContinuationMetrics` | 0 | 0 |
| `ScalarGateResult` | `gkx.diagnostics.validation_gates:ScalarGateResult` | 0 | 0 |
| `GateReport` | `gkx.diagnostics.validation_gates:GateReport` | 0 | 0 |
| `LateTimeLinearMetrics` | `gkx.diagnostics.analysis:LateTimeLinearMetrics` | 0 | 0 |
| `NonlinearWindowMetrics` | `gkx.diagnostics.analysis:NonlinearWindowMetrics` | 0 | 0 |
| `ZonalFlowResponseMetrics` | `gkx.diagnostics.validation_gates:ZonalFlowResponseMetrics` | 1 | 0 |
| `branch_continuity_gate_report` | `gkx.diagnostics.validation_gates:branch_continuity_gate_report` | 1 | 0 |
| `branch_continuity_metrics` | `gkx.diagnostics.analysis:branch_continuity_metrics` | 0 | 0 |
| `covariance_diagnostics` | `gkx.objectives.autodiff_validation:covariance_diagnostics` | 0 | 0 |
| `autodiff_finite_difference_report` | `gkx.objectives.autodiff_validation:autodiff_finite_difference_report` | 0 | 0 |
| `central_finite_difference_jacobian` | `gkx.objectives.autodiff_validation:central_finite_difference_jacobian` | 0 | 0 |
| `explicit_complex_operator_matrix` | `gkx.objectives.autodiff_validation:explicit_complex_operator_matrix` | 1 | 0 |
| `implicit_eigenpair_observable_sensitivity_report` | `gkx.objectives.autodiff_validation:implicit_eigenpair_observable_sensitivity_report` | 0 | 0 |
| `isolated_eigenpair_observable_sensitivity_report` | `gkx.objectives.autodiff_validation:isolated_eigenpair_observable_sensitivity_report` | 0 | 0 |
| `isolated_eigenvalue_sensitivity_report` | `gkx.objectives.autodiff_validation:isolated_eigenvalue_sensitivity_report` | 0 | 0 |
| `IndependentEnsembleProvenanceReport` | `gkx.parallel:IndependentEnsembleProvenanceReport` | 0 | 0 |
| `IndependentMapExecutionError` | `gkx.parallel:IndependentMapExecutionError` | 1 | 0 |
| `IndependentWorkerMetadata` | `gkx.parallel:IndependentWorkerMetadata` | 0 | 0 |
| `ParallelIdentityReport` | `gkx.parallel:ParallelIdentityReport` | 0 | 0 |
| `batch_map` | `gkx.parallel:batch_map` | 7 | 0 |
| `batch_map_identity_report` | `gkx.parallel:batch_map_identity_report` | 0 | 0 |
| `independent_ensemble_provenance_gate` | `gkx.parallel:independent_ensemble_provenance_gate` | 1 | 0 |
| `independent_map` | `gkx.parallel:independent_map` | 5 | 0 |
| `independent_map_identity_report` | `gkx.parallel:independent_map_identity_report` | 0 | 0 |
| `independent_worker_metadata` | `gkx.parallel:independent_worker_metadata` | 0 | 0 |
| `ky_scan_batches` | `gkx.parallel:ky_scan_batches` | 4 | 0 |
| `parallel_identity_report` | `gkx.parallel:parallel_identity_report` | 0 | 0 |
| `DecompositionContract` | `gkx.parallel.decomposition:DecompositionContract` | 0 | 0 |
| `ReconstructionIdentityReport` | `gkx.parallel.decomposition:ReconstructionIdentityReport` | 0 | 0 |
| `ShardAssignment` | `gkx.parallel.decomposition:ShardAssignment` | 0 | 0 |
| `build_diagnostic_nonlinear_domain_decomposition` | `gkx.parallel.decomposition:build_diagnostic_nonlinear_domain_decomposition` | 0 | 0 |
| `build_independent_portfolio_decomposition` | `gkx.parallel.decomposition:build_independent_portfolio_decomposition` | 0 | 0 |
| `reconstruct_serial` | `gkx.parallel.decomposition:reconstruct_serial` | 0 | 0 |
| `serial_reconstruction_identity_report` | `gkx.parallel.decomposition:serial_reconstruction_identity_report` | 0 | 0 |
| `shard_sequence` | `gkx.parallel.decomposition:shard_sequence` | 0 | 0 |
| `NonlinearDomainDecompositionPlan` | `gkx.operators.nonlinear.parallel:NonlinearDomainDecompositionPlan` | 0 | 0 |
| `NonlinearDomainIdentityReport` | `gkx.operators.nonlinear.parallel:NonlinearDomainIdentityReport` | 0 | 0 |
| `NonlinearDomainTransportWindowReport` | `gkx.operators.nonlinear.parallel:NonlinearDomainTransportWindowReport` | 0 | 0 |
| `NonlinearParallelStrategy` | `gkx.operators.nonlinear.parallel:NonlinearParallelStrategy` | 0 | 0 |
| `NonlinearSpectralCommunicationReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralCommunicationReport` | 0 | 0 |
| `NonlinearSpectralDevicePencilFFTBatchModel` | `gkx.operators.nonlinear.parallel:NonlinearSpectralDevicePencilFFTBatchModel` | 0 | 0 |
| `NonlinearSpectralDevicePencilRHSIdentityReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralDevicePencilRHSIdentityReport` | 0 | 0 |
| `NonlinearSpectralDevicePencilTransportWindowReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralDevicePencilTransportWindowReport` | 0 | 0 |
| `NonlinearSpectralDomainWorkModel` | `gkx.operators.nonlinear.parallel:NonlinearSpectralDomainWorkModel` | 0 | 0 |
| `NonlinearSpectralIntegratorIdentityReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralIntegratorIdentityReport` | 0 | 0 |
| `NonlinearSpectralPencilRHSIdentityReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralPencilRHSIdentityReport` | 0 | 0 |
| `NonlinearSpectralPencilTransportWindowReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralPencilTransportWindowReport` | 0 | 0 |
| `NonlinearSpectralPencilWorkModel` | `gkx.operators.nonlinear.parallel:NonlinearSpectralPencilWorkModel` | 0 | 0 |
| `NonlinearSpectralRHSIdentityReport` | `gkx.operators.nonlinear.parallel:NonlinearSpectralRHSIdentityReport` | 0 | 0 |
| `build_nonlinear_domain_decomposition_plan` | `gkx.operators.nonlinear.parallel:build_nonlinear_domain_decomposition_plan` | 0 | 0 |
| `classify_nonlinear_parallel_strategy` | `gkx.operators.nonlinear.parallel:classify_nonlinear_parallel_strategy` | 0 | 0 |
| `deterministic_nonlinear_domain_state` | `gkx.operators.nonlinear.parallel:deterministic_nonlinear_domain_state` | 0 | 0 |
| `deterministic_nonlinear_spectral_state` | `gkx.operators.nonlinear.parallel:deterministic_nonlinear_spectral_state` | 0 | 0 |
| `device_z_pencil_fft_batch_pressure_model` | `gkx.operators.nonlinear.parallel:device_z_pencil_fft_batch_pressure_model` | 0 | 0 |
| `device_z_pencil_nonlinear_spectral_rhs` | `gkx.operators.nonlinear.parallel:device_z_pencil_nonlinear_spectral_rhs` | 0 | 0 |
| `device_z_pencil_nonlinear_spectral_transport_window_identity_gate` | `gkx.operators.nonlinear.parallel:device_z_pencil_nonlinear_spectral_transport_window_identity_gate` | 0 | 0 |
| `integrate_logical_decomposed_nonlinear_spectral` | `gkx.operators.nonlinear.parallel:integrate_logical_decomposed_nonlinear_spectral` | 0 | 0 |
| `nonlinear_domain_identity_report` | `gkx.operators.nonlinear.parallel:nonlinear_domain_identity_report` | 0 | 0 |
| `nonlinear_domain_parallel_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_domain_parallel_identity_gate` | 4 | 0 |
| `nonlinear_domain_transport_window_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_domain_transport_window_identity_gate` | 0 | 0 |
| `nonlinear_parallel_strategies` | `gkx.operators.nonlinear.parallel:nonlinear_parallel_strategies` | 0 | 0 |
| `nonlinear_parallel_strategy` | `gkx.operators.nonlinear.parallel:nonlinear_parallel_strategy` | 0 | 0 |
| `nonlinear_spectral_communication_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_communication_identity_gate` | 3 | 0 |
| `nonlinear_spectral_communication_identity_report` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_communication_identity_report` | 0 | 0 |
| `nonlinear_spectral_domain_work_model` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_domain_work_model` | 0 | 0 |
| `logical_decomposed_nonlinear_spectral_rhs` | `gkx.operators.nonlinear.parallel:logical_decomposed_nonlinear_spectral_rhs` | 1 | 0 |
| `nonlinear_spectral_integrator_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_integrator_identity_gate` | 1 | 0 |
| `nonlinear_spectral_pencil_rhs_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_pencil_rhs_identity_gate` | 0 | 0 |
| `nonlinear_spectral_pencil_transport_window_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_pencil_transport_window_identity_gate` | 0 | 0 |
| `nonlinear_spectral_pencil_work_model` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_pencil_work_model` | 0 | 0 |
| `nonlinear_spectral_rhs_identity_gate` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_rhs_identity_gate` | 1 | 0 |
| `nonlinear_spectral_rhs_identity_report` | `gkx.operators.nonlinear.parallel:nonlinear_spectral_rhs_identity_report` | 0 | 0 |
| `pencil_decomposed_nonlinear_spectral_rhs` | `gkx.operators.nonlinear.parallel:pencil_decomposed_nonlinear_spectral_rhs` | 0 | 0 |
| `local_stencil_nonlinear_domain_decomposed_step` | `gkx.operators.nonlinear.parallel:local_stencil_nonlinear_domain_decomposed_step` | 0 | 0 |
| `local_stencil_nonlinear_domain_serial_step` | `gkx.operators.nonlinear.parallel:local_stencil_nonlinear_domain_serial_step` | 0 | 0 |
| `release_ready_nonlinear_parallel_strategies` | `gkx.operators.nonlinear.parallel:release_ready_nonlinear_parallel_strategies` | 0 | 0 |
| `VelocityShardingPlan` | `gkx.parallel.velocity:VelocityShardingPlan` | 0 | 0 |
| `build_velocity_sharding_plan` | `gkx.parallel.velocity:build_velocity_sharding_plan` | 1 | 0 |
| `curvature_gradb_drift_reference` | `gkx.parallel.velocity:curvature_gradb_drift_reference` | 0 | 0 |
| `curvature_gradb_drift_shard_map` | `gkx.parallel.velocity:curvature_gradb_drift_shard_map` | 0 | 0 |
| `diamagnetic_drive_reference` | `gkx.parallel.velocity:diamagnetic_drive_reference` | 0 | 0 |
| `diamagnetic_drive_shard_map` | `gkx.parallel.velocity:diamagnetic_drive_shard_map` | 0 | 0 |
| `electrostatic_phi_reference` | `gkx.parallel.velocity:electrostatic_phi_reference` | 0 | 0 |
| `electrostatic_phi_shard_map` | `gkx.parallel.velocity:electrostatic_phi_shard_map` | 0 | 0 |
| `hermite_neighbor_reference` | `gkx.parallel.velocity:hermite_neighbor_reference` | 0 | 0 |
| `hermite_neighbor_shard_map` | `gkx.parallel.velocity:hermite_neighbor_shard_map` | 0 | 0 |
| `hermite_shift_reference` | `gkx.parallel.velocity:hermite_shift_reference` | 0 | 0 |
| `hermite_shift_shard_map` | `gkx.parallel.velocity:hermite_shift_shard_map` | 0 | 0 |
| `hermite_streaming_ladder_reference` | `gkx.parallel.velocity:hermite_streaming_ladder_reference` | 0 | 0 |
| `hermite_streaming_ladder_shard_map` | `gkx.parallel.velocity:hermite_streaming_ladder_shard_map` | 0 | 0 |
| `mirror_drift_reference` | `gkx.parallel.velocity:mirror_drift_reference` | 0 | 0 |
| `mirror_drift_shard_map` | `gkx.parallel.velocity:mirror_drift_shard_map` | 0 | 0 |
| `periodic_streaming_reference` | `gkx.parallel.velocity:periodic_streaming_reference` | 0 | 0 |
| `periodic_streaming_shard_map` | `gkx.parallel.velocity:periodic_streaming_shard_map` | 0 | 0 |
| `velocity_field_reduce_reference` | `gkx.parallel.velocity:velocity_field_reduce_reference` | 0 | 0 |
| `velocity_field_reduce_shard_map` | `gkx.parallel.velocity:velocity_field_reduce_shard_map` | 0 | 0 |
| `STELLARATOR_ITG_OBSERVABLE_NAMES` | `gkx.objectives.stellarator:OBSERVABLE_NAMES` | 0 | 0 |
| `STELLARATOR_ITG_PARAMETER_NAMES` | `gkx.objectives.stellarator:PARAMETER_NAMES` | 0 | 0 |
| `StellaratorITGOptimizationConfig` | `gkx.objectives.stellarator:StellaratorITGOptimizationConfig` | 0 | 0 |
| `StellaratorITGOptimizationResult` | `gkx.objectives.stellarator:StellaratorITGOptimizationResult` | 0 | 0 |
| `StellaratorITGSampleSet` | `gkx.objectives.stellarator:StellaratorITGSampleSet` | 0 | 0 |
| `StellaratorObjectiveKind` | `gkx.objectives.stellarator:StellaratorObjectiveKind` | 0 | 0 |
| `compare_stellarator_itg_objectives` | `gkx.objectives.stellarator:compare_stellarator_itg_objectives` | 0 | 0 |
| `default_stellarator_initial_params` | `gkx.objectives.stellarator:default_stellarator_initial_params` | 0 | 0 |
| `nonlinear_heat_flux_trace` | `gkx.objectives.stellarator:nonlinear_heat_flux_trace` | 0 | 0 |
| `nonlinear_heat_flux_window_metrics` | `gkx.objectives.stellarator:nonlinear_heat_flux_window_metrics` | 0 | 0 |
| `optimize_stellarator_itg` | `gkx.objectives.stellarator:optimize_stellarator_itg` | 0 | 0 |
| `qa_max_mode1_observables` | `gkx.objectives.stellarator:qa_max_mode1_observables` | 0 | 0 |
| `qa_observable_vector` | `gkx.objectives.stellarator:qa_observable_vector` | 0 | 0 |
| `stellarator_itg_density_gradient_scan` | `gkx.objectives.stellarator:stellarator_itg_density_gradient_scan` | 0 | 0 |
| `stellarator_itg_portfolio_gate_payload` | `gkx.objectives.stellarator:stellarator_itg_portfolio_gate_payload` | 0 | 0 |
| `stellarator_itg_portfolio_sensitivity_report` | `gkx.objectives.stellarator:stellarator_itg_portfolio_sensitivity_report` | 0 | 0 |
| `stellarator_itg_objective` | `gkx.objectives.stellarator:stellarator_itg_objective` | 0 | 0 |
| `stellarator_itg_objective_residual_names` | `gkx.objectives.stellarator:stellarator_itg_objective_residual_names` | 0 | 0 |
| `stellarator_itg_objective_residual_vector` | `gkx.objectives.stellarator:stellarator_itg_objective_residual_vector` | 0 | 0 |
| `stellarator_itg_reduced_portfolio_objective` | `gkx.objectives.stellarator:stellarator_itg_reduced_portfolio_objective` | 0 | 0 |
| `stellarator_itg_residual_sensitivity_report` | `gkx.objectives.stellarator:stellarator_itg_residual_sensitivity_report` | 0 | 0 |
| `stellarator_itg_sample_objective_table` | `gkx.objectives.stellarator:stellarator_itg_sample_objective_table` | 0 | 0 |
| `stellarator_itg_vmec_boozer_portfolio_objective_from_state` | `gkx.objectives.stellarator:stellarator_itg_vmec_boozer_portfolio_objective_from_state` | 1 | 0 |
| `stellarator_itg_vmec_boozer_sample_objective_table_from_state` | `gkx.objectives.stellarator:stellarator_itg_vmec_boozer_sample_objective_table_from_state` | 1 | 0 |
| `StellaratorObjectivePortfolioContract` | `gkx.objectives.portfolio:StellaratorObjectivePortfolioContract` | 0 | 0 |
| `ProjectedLineSearchPolicy` | `gkx.objectives.vmec_transport_optimization:ProjectedLineSearchPolicy` | 0 | 0 |
| `VMEXGKXTransportObjective` | `gkx.objectives.vmec_transport:VMEXGKXTransportObjective` | 0 | 0 |
| `VMEXTransportObjectiveConfig` | `gkx.objectives.vmec_transport:VMEXTransportObjectiveConfig` | 0 | 1 |
| `VMEXTransportObjectiveKind` | `gkx.objectives.vmec_transport:VMEXTransportObjectiveKind` | 0 | 0 |
| `aggregate_objective_portfolio` | `gkx.objectives.portfolio:aggregate_objective_portfolio` | 0 | 0 |
| `boundary_spec_record` | `gkx.objectives.vmec_transport_optimization:boundary_spec_record` | 0 | 0 |
| `boundary_chain_accepted_parameter_indices` | `gkx.objectives.vmec_transport_optimization:boundary_chain_accepted_parameter_indices` | 0 | 0 |
| `boundary_chain_summary_from_probe` | `gkx.geometry.vmec_boundary_chain:boundary_chain_summary_from_probe` | 0 | 0 |
| `build_boundary_chain_collection_summary` | `gkx.geometry.vmec_boundary_chain:build_boundary_chain_collection_summary` | 0 | 0 |
| `build_boundary_chain_summary` | `gkx.geometry.vmec_boundary_chain:build_boundary_chain_summary` | 0 | 0 |
| `objective_portfolio_sensitivity_report` | `gkx.objectives.portfolio:objective_portfolio_sensitivity_report` | 0 | 0 |
| `portfolio_objective_weight_vector` | `gkx.objectives.portfolio:portfolio_objective_weight_vector` | 0 | 0 |
| `portfolio_sample_weight_tensor` | `gkx.objectives.portfolio:portfolio_sample_weight_tensor` | 0 | 0 |
| `projected_line_search_input_manifest` | `gkx.objectives.vmec_transport_optimization:projected_line_search_input_manifest` | 0 | 0 |
| `validate_objective_portfolio_contract` | `gkx.objectives.portfolio:validate_objective_portfolio_contract` | 0 | 0 |
| `ZONAL_FLOW_OBJECTIVE_NAMES` | `gkx.objectives.zonal:ZONAL_FLOW_OBJECTIVE_NAMES` | 0 | 0 |
| `ZonalFlowObjectiveConfig` | `gkx.objectives.zonal:ZonalFlowObjectiveConfig` | 0 | 0 |
| `zonal_flow_objective_artifact_from_records` | `gkx.objectives.zonal:zonal_flow_objective_artifact_from_records` | 0 | 0 |
| `zonal_flow_objective_rows` | `gkx.objectives.zonal:zonal_flow_objective_rows` | 0 | 0 |
| `zonal_flow_objective_sensitivity_report` | `gkx.objectives.zonal:zonal_flow_objective_sensitivity_report` | 0 | 0 |
| `zonal_flow_reduced_objective` | `gkx.objectives.zonal:zonal_flow_reduced_objective` | 0 | 0 |
| `SOLVER_GEOMETRY_PARAMETER_NAMES` | `gkx.objectives.geometry:SOLVER_GEOMETRY_PARAMETER_NAMES` | 0 | 0 |
| `AdaptiveLinearEigensolverConfig` | `gkx.objectives.core:AdaptiveLinearEigensolverConfig` | 3 | 0 |
| `LinearEigensolver` | `gkx.objectives.core:LinearEigensolver` | 0 | 0 |
| `SOLVER_OBJECTIVE_NAMES` | `gkx.objectives.core:SOLVER_OBJECTIVE_NAMES` | 0 | 0 |
| `SolverScalarObjective` | `gkx.objectives.core:SolverScalarObjective` | 0 | 0 |
| `VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES` | `gkx.objectives.vmec_boozer_gradients:VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES` | 0 | 0 |
| `VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES` | `gkx.objectives.vmec_boozer_gradients:VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES` | 0 | 0 |
| `VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES` | `gkx.objectives.vmec_boozer_gradients:VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES` | 0 | 0 |
| `VMEC_BOOZER_STATE_PARAMETER_FAMILIES` | `gkx.geometry.vmec_state_controls:VMEC_BOOZER_STATE_PARAMETER_FAMILIES` | 0 | 0 |
| `VMEC_BOOZER_STATE_PARAMETER_NAMES` | `gkx.geometry.vmec_state_controls:VMEC_BOOZER_STATE_PARAMETER_NAMES` | 0 | 0 |
| `default_solver_geometry_design_params` | `gkx.objectives.geometry:default_solver_geometry_design_params` | 0 | 0 |
| `dominant_eigenvalue_branch_locality_report` | `gkx.objectives.eigen:dominant_eigenvalue_branch_locality_report` | 0 | 0 |
| `dominant_real_eigenvalue` | `gkx.objectives.eigen:dominant_real_eigenvalue` | 0 | 0 |
| `mode21_vmec_boozer_linear_frequency_gradient_report` | `gkx.objectives.vmec_boozer_gradients:mode21_vmec_boozer_linear_frequency_gradient_report` | 0 | 0 |
| `mode21_vmec_boozer_nonlinear_window_gradient_report` | `gkx.objectives.vmec_boozer_gradients:mode21_vmec_boozer_nonlinear_window_gradient_report` | 0 | 0 |
| `mode21_vmec_boozer_quasilinear_gradient_report` | `gkx.objectives.vmec_boozer_gradients:mode21_vmec_boozer_quasilinear_gradient_report` | 0 | 0 |
| `solver_linear_operator_matrix_from_geometry` | `gkx.objectives.core:solver_linear_operator_matrix_from_geometry` | 1 | 1 |
| `solver_growth_rate_from_geometry` | `gkx.objectives.core:solver_growth_rate_from_geometry` | 1 | 0 |
| `solver_objective_vector_from_geometry` | `gkx.objectives.core:solver_objective_vector_from_geometry` | 3 | 1 |
| `solver_grid_options_from_ky_values` | `gkx.objectives.sampling:solver_grid_options_from_ky_values` | 0 | 0 |
| `solver_scalar_objective_from_vector` | `gkx.objectives.core:solver_scalar_objective_from_vector` | 0 | 1 |
| `build_boundary_transport_gradient_report` | `gkx.objectives.vmec_transport_optimization:build_boundary_transport_gradient_report` | 0 | 0 |
| `select_projected_line_search_candidate` | `gkx.objectives.vmec_transport_optimization:select_projected_line_search_candidate` | 0 | 0 |
| `sparse_descent_direction_from_gradient_report` | `gkx.objectives.vmec_transport_optimization:sparse_descent_direction_from_gradient_report` | 0 | 0 |
| `vmex_transport_growth_branch_locality_report_from_states` | `gkx.objectives.vmec_transport_branch:vmex_transport_growth_branch_locality_report_from_states` | 1 | 0 |
| `vmex_transport_objective_from_state` | `gkx.objectives.vmec_transport:vmex_transport_objective_from_state` | 0 | 0 |
| `write_boundary_transport_gradient_report` | `gkx.objectives.vmec_transport_optimization:write_boundary_transport_gradient_report` | 0 | 0 |
| `solver_ready_geometry_mapping` | `gkx.objectives.geometry:solver_ready_geometry_mapping` | 0 | 0 |
| `vmec_boozer_aggregate_line_search_holdout_report` | `gkx.objectives.solver_vmec:vmec_boozer_aggregate_line_search_holdout_report` | 0 | 0 |
| `vmec_boozer_aggregate_scalar_objective_finite_difference_report` | `gkx.objectives.solver_vmec:vmec_boozer_aggregate_scalar_objective_finite_difference_report` | 0 | 0 |
| `vmec_boozer_aggregate_scalar_objective_from_state` | `gkx.objectives.solver_vmec:vmec_boozer_aggregate_scalar_objective_from_state` | 0 | 0 |
| `vmec_boozer_aggregate_scalar_objective_line_search_report` | `gkx.objectives.solver_vmec:vmec_boozer_aggregate_scalar_objective_line_search_report` | 0 | 0 |
| `vmec_boozer_scalar_objective_finite_difference_report` | `gkx.objectives.solver_vmec:vmec_boozer_scalar_objective_finite_difference_report` | 0 | 0 |
| `vmec_boozer_scalar_objective_from_state` | `gkx.objectives.solver_vmec:vmec_boozer_scalar_objective_from_state` | 0 | 0 |
| `vmec_boozer_scalar_objective_line_search_report` | `gkx.objectives.solver_vmec:vmec_boozer_scalar_objective_line_search_report` | 0 | 0 |
| `vmec_boozer_solver_objective_table_from_state` | `gkx.objectives.solver_vmec:vmec_boozer_solver_objective_table_from_state` | 0 | 0 |
| `vmec_boozer_solver_objective_table_with_metadata_from_state` | `gkx.objectives.solver_vmec:vmec_boozer_solver_objective_table_with_metadata_from_state` | 0 | 0 |
| `vmec_boozer_solver_objective_vector_from_state` | `gkx.objectives.solver_vmec:vmec_boozer_solver_objective_vector_from_state` | 0 | 0 |
| `eigenfunction_gate_report` | `gkx.diagnostics.validation_gates:eigenfunction_gate_report` | 1 | 0 |
| `evaluate_scalar_gate` | `gkx.diagnostics.validation_gates:evaluate_scalar_gate` | 1 | 0 |
| `gate_report` | `gkx.diagnostics.validation_gates:gate_report` | 4 | 0 |
| `gate_report_to_dict` | `gkx.diagnostics.validation_gates:gate_report_to_dict` | 0 | 0 |
| `linear_metrics_gate_report` | `gkx.diagnostics.validation_gates:linear_metrics_gate_report` | 1 | 0 |
| `nonlinear_window_gate_report` | `gkx.diagnostics.validation_gates:nonlinear_window_gate_report` | 1 | 0 |
| `observed_order_gate_report` | `gkx.diagnostics.validation_gates:observed_order_gate_report` | 2 | 0 |
| `zonal_response_gate_report` | `gkx.diagnostics.validation_gates:zonal_response_gate_report` | 1 | 0 |
| `normalize_eigenfunction` | `gkx.diagnostics.modes:normalize_eigenfunction` | 0 | 0 |
| `run_linear_scan` | `gkx.workflows.linear:run_linear_scan` | 0 | 0 |
| `run_scan_and_mode` | `gkx.workflows.linear:run_scan_and_mode` | 0 | 0 |
| `ModeSelection` | `gkx.diagnostics.analysis:ModeSelection` | 0 | 0 |
| `extract_mode` | `gkx.diagnostics.analysis:extract_mode` | 0 | 0 |
| `extract_mode_time_series` | `gkx.diagnostics.analysis:extract_mode_time_series` | 0 | 0 |
| `extract_eigenfunction` | `gkx.diagnostics.analysis:extract_eigenfunction` | 0 | 0 |
| `select_ky_index` | `gkx.diagnostics.analysis:select_ky_index` | 0 | 0 |
| `cyclone_reference_figure` | `gkx.artifacts.plotting:cyclone_reference_figure` | 0 | 0 |
| `cyclone_comparison_figure` | `gkx.artifacts.plotting:cyclone_comparison_figure` | 0 | 0 |
| `etg_trend_figure` | `gkx.artifacts.plotting:etg_trend_figure` | 0 | 0 |
| `growth_rate_heatmap` | `gkx.artifacts.plotting:growth_rate_heatmap` | 0 | 0 |
| `growth_fit_figure` | `gkx.artifacts.plotting:growth_fit_figure` | 0 | 0 |
| `linear_validation_figure` | `gkx.artifacts.plotting:linear_validation_figure` | 0 | 0 |
| `LinearValidationPanel` | `gkx.artifacts.plotting:LinearValidationPanel` | 0 | 0 |
| `MultiReferenceValidationPanel` | `gkx.artifacts.plotting:MultiReferenceValidationPanel` | 0 | 0 |
| `ReferenceSeries` | `gkx.artifacts.plotting:ReferenceSeries` | 0 | 0 |
| `linear_validation_multi_reference_figure` | `gkx.artifacts.plotting:linear_validation_multi_reference_figure` | 0 | 0 |
| `scan_multi_reference_figure` | `gkx.artifacts.plotting:scan_multi_reference_figure` | 0 | 0 |
| `scan_comparison_figure` | `gkx.artifacts.plotting:scan_comparison_figure` | 0 | 0 |
| `set_plot_style` | `gkx.artifacts.plotting:set_plot_style` | 0 | 0 |

The additional facade name is `__version__`, owned by `gkx._version`. It is
not part of `_EXPORT_TARGETS` and is therefore excluded from the 346-row
digest payload.

## Reproduction

```bash
python - <<'PY2'
import hashlib
import gkx
from gkx import api
payload = "\n".join(
    f"{name}\t{api._EXPORT_TARGETS[name]}" for name in api.__all__
).encode()
assert len(gkx.__all__) == 347
assert len(api.__all__) == 346
digest = hashlib.sha256(payload).hexdigest()
assert digest == "d74c9ddcb00b724c2749dfeda683c8ea207b89109eb609e3d1389bbb4a929e27"
print(digest)
PY2
```

The table is generated mechanically from the registry, exact-token documentation
searches, and VMEX AST nodes. Re-run that inventory after every Phase 1 facade
change and state which names are promoted, moved, adapted, or removed.
