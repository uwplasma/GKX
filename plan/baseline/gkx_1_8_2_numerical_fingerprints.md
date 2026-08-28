# GKX 1.8.2 self-contained numerical fingerprints

Status: Phase 0 compatibility snapshot at GKX
`cb2219bbf835a7f96817bf766bbbfc29c992a0b5`. The selected cases use only GKX
source, tracked GKX-owned inputs, analytic formulae, and tracked compact GKX
artifacts. No external executable or raw external-code output is an oracle.

File SHA-256 values identify the exact historical evidence. Numerical values
are the compatibility fingerprints. A regenerated performance profile is not
expected to retain the same file hash or timing, but its matched configuration,
shapes, finiteness, and dtype-aware numerical values must satisfy the owning
tests.

## Fingerprint matrix

| Lane | Self-contained case | Frozen observation | Owner/reproduction | Claim boundary |
|---|---|---|---|---|
| Linear | Cyclone Dimits linear-threshold scan | critical multiplier `0.5612692502942052`; sign bracket `[0.5583333333333333, 0.565625]`; relative fit residual `0.05347421384328944` | `tools/campaigns/dimits_shift.py`; `tests/unit/linear/test_dimits_threshold_extraction.py` | linear threshold only; no nonlinear Dimits-shift onset claim |
| Nonlinear | 200-step adaptive RK3 Cyclone prepared solve, `(64,64,24,4,8)`, float32 | CPU/GPU final-state L2 `0.12650516629219055` / `0.12656144797801971`; phi L2 `0.04132739081978798` / `0.04132746532559395`; heat-flux L2 `0.00010252444917568937` / `0.00010252481297357008` | `tools/profiling/profile_runtime_kernels.py`; prepared-profile contract tests | short trajectory and profiler contract; not stationary or converged transport |
| Geometry | analytic s-alpha, `q=1.4`, `s_hat=1`, `epsilon=0`, `kx=0`, `ky=1`, `theta=[0,2]` | `k_perp^2=[1,5]` exactly within float dtype | `tests/unit/geometry/test_geometry.py::test_kperp2_matches_s_alpha` | analytic convention identity; not imported/VMEC/Boozer parity |
| Collision | finite-b Coulomb algebra, Hermite 0--3 and Laguerre 0--1 | all gates true; maximum invariant residual `5.551115123125783e-17`; maximum projection relative error `3.1316539653265556e-13`; maximum eigenvalue `9.00846363925679e-18` | `tools/artifacts/build_linear_validation_artifacts.py`; frozen-output release contract | offline operator algebra, not runtime transport accuracy |
| Restart | complex64 zonal/radial NetCDF round trip, shape `(1,3,4,8,8,6)` | maximum absolute error `0`; state and loaded SHA-256 both `14e4eb13...20f` after canonical load comparison | `tests/integration/runtime/test_restart_gate.py::test_netcdf_restart_roundtrips_zonal_radial_modes` | serialization identity; not interrupted long-run statistical identity |
| Differentiation | two-observable inverse-growth demo | AD/FD row relative errors `0.0003877815615851432`, `0.00019905040971934795`; final loss `7.467800514859846e-06` | `examples/theory_and_demos/autodiff_inverse_growth.py`; integration example test | reduced differentiable workflow and local conditioning only |
| Optimization | four-observable/two-parameter inverse demo | final `(tprim,fprim)=(2.8000000452018434,0.7999999842522637)`; loss `0`; AD/FD row relative errors `0.00040817007538862526`, `6.33443269180134e-05` | `examples/theory_and_demos/autodiff_inverse_twomode.py`; integration example test | small deterministic inverse problem; not nonlinear stellarator optimization |

## Artifact provenance

| Lane | Path | SHA-256 |
|---|---|---|
| Linear | `docs/_static/dimits_linear_threshold.json` | `5104ca0aec04a50149ab08c57a96c4f2ca300aa445735b23d5b536a327f0b5c1` |
| Nonlinear CPU | `docs/_static/prepared_nonlinear_runtime_cpu_profile.json` | `55d0b4c9f221c3e2388beff3345f7b1e1eda42fb335b939d0b98dc046706a5df` |
| Nonlinear GPU | `docs/_static/prepared_nonlinear_runtime_gpu_profile.json` | `3eb879da6258a9f725a9cccc39131a816b51112384e496c443b86f5483692628` |
| Collision | `docs/_static/collision_operator_verification.json` | `5429454bbc5b726c4ff0c175203babbf956d8d3fc115d4989b8b46793be46ac2` |
| Restart | `benchmarks/references/gkx_1_7_restart_identity.json` | `63fa8d86745eb257ab298430c19317ac2690822fdce9ef3194a5ed970eea453e` |
| Differentiation | `docs/_static/autodiff_inverse_growth_summary.json` | `6be7a1aa95bf1accfafdc88dc694ad1f3efad2dd91063daa9d726cf6a7e4b121` |
| Optimization | `docs/_static/autodiff_inverse_twomode_summary.json` | `5031656ab72bb7c4f047ef2c8ada5c309350924ad581e7012811128b8c1a9ccb` |

The restart filename retains its 1.7 origin because the 1.8.2 release still
uses and release-gates that unchanged compatibility reference. The collision
and restart hashes are also frozen by
`benchmarks/references/gkx_1_7_release_contract.json`.

The prepared profiles bind both machines to profiler-fix revision
`b150705cd568657443ad2433aa071aef06d57892`, Python 3.12.13, JAX/JAXLIB
0.10.2, NumPy 2.5.2, float32, the same shipped TOML, and the same 200-step
configuration. Their numerical L2 differences relative to the CPU values are
`4.4489634280364557e-4` for final state, `1.8028190140076333e-6` for phi, and
`3.548401221700347e-6` for heat flux. The owning test allows `1e-3` for the
accumulated state and `1e-5` for the other result summaries. Warm runtime and
memory are performance observations documented separately, not numerical
fingerprints.

## Exact reproduction

Run from the GKX checkout in the synchronized Phase 0 environment:

```console
python - <<'PY'
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from gkx.geometry import SAlphaGeometry

expected_hashes = {
    "docs/_static/dimits_linear_threshold.json":
        "5104ca0aec04a50149ab08c57a96c4f2ca300aa445735b23d5b536a327f0b5c1",
    "docs/_static/prepared_nonlinear_runtime_cpu_profile.json":
        "55d0b4c9f221c3e2388beff3345f7b1e1eda42fb335b939d0b98dc046706a5df",
    "docs/_static/prepared_nonlinear_runtime_gpu_profile.json":
        "3eb879da6258a9f725a9cccc39131a816b51112384e496c443b86f5483692628",
    "docs/_static/collision_operator_verification.json":
        "5429454bbc5b726c4ff0c175203babbf956d8d3fc115d4989b8b46793be46ac2",
    "benchmarks/references/gkx_1_7_restart_identity.json":
        "63fa8d86745eb257ab298430c19317ac2690822fdce9ef3194a5ed970eea453e",
    "docs/_static/autodiff_inverse_growth_summary.json":
        "6be7a1aa95bf1accfafdc88dc694ad1f3efad2dd91063daa9d726cf6a7e4b121",
    "docs/_static/autodiff_inverse_twomode_summary.json":
        "5031656ab72bb7c4f047ef2c8ada5c309350924ad581e7012811128b8c1a9ccb",
}

payloads = {}
for name, expected in expected_hashes.items():
    path = Path(name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    payloads[name] = json.loads(path.read_text())

linear = payloads["docs/_static/dimits_linear_threshold.json"]
assert linear["claim_level"] == "linear_threshold_only_no_nonlinear_onset_measured"
np.testing.assert_allclose(
    linear["threshold"]["critical_multiplier"], 0.5612692502942052,
    rtol=0, atol=0,
)
assert linear["threshold"]["sign_change_bracket"] == [
    0.5583333333333333, 0.565625
]

cpu = payloads["docs/_static/prepared_nonlinear_runtime_cpu_profile.json"]
gpu = payloads["docs/_static/prepared_nonlinear_runtime_gpu_profile.json"]
assert cpu["git_revision"] == gpu["git_revision"] == (
    "b150705cd568657443ad2433aa071aef06d57892"
)
for name, shape, rtol in (
    ("final_state", [1, 4, 8, 64, 64, 24], 1e-3),
    ("phi", [64, 64, 24], 1e-5),
    ("heat_flux", [21], 1e-5),
):
    left = cpu["result_summary"][name]
    right = gpu["result_summary"][name]
    assert left["shape"] == right["shape"] == shape
    assert left["finite_fraction"] == right["finite_fraction"] == 1.0
    np.testing.assert_allclose(left["l2_norm"], right["l2_norm"], rtol=rtol)

geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0)
np.testing.assert_allclose(
    geom.k_perp2(jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray([0.0, 2.0])),
    [1.0, 5.0], rtol=0, atol=0,
)

collision = payloads["docs/_static/collision_operator_verification.json"]
assert collision["gate_passed"] is True
assert collision["metrics"]["maximum_invariant_residual"] == 5.551115123125783e-17
assert collision["metrics"]["maximum_projection_relative_error"] == 3.1316539653265556e-13

restart = payloads["benchmarks/references/gkx_1_7_restart_identity.json"]
assert restart["passed"] is True
assert restart["max_abs_error"] == 0.0
assert restart["shape"] == [1, 3, 4, 8, 8, 6]
assert restart["loaded_sha256"] == "14e4eb13ed358afc99fd565c8896847a509e8d17423b6afad82272ab3e5db20f"

ad = payloads["docs/_static/autodiff_inverse_growth_summary.json"]
np.testing.assert_allclose(
    ad["jac_rel_error"],
    [0.0003877815615851432, 0.00019905040971934795], rtol=0, atol=0,
)

opt = payloads["docs/_static/autodiff_inverse_twomode_summary.json"]
assert opt["loss_final"] == 0.0
np.testing.assert_allclose(
    [opt["tprim_final"], opt["fprim_final"]],
    [2.8000000452018434, 0.7999999842522637], rtol=0, atol=0,
)
print("seven Phase 0 fingerprint lanes verified")
PY
```

Then run the owning self-contained tests:

```console
python -m pytest -q \
  tests/unit/linear/test_dimits_threshold_extraction.py \
  tests/tools/profiling/test_runtime_and_scaling_profile_contracts.py \
  tests/unit/geometry/test_geometry.py \
  tests/integration/runtime/test_restart_gate.py::test_netcdf_restart_roundtrips_zonal_radial_modes \
  tests/integration/examples/test_examples.py::test_autodiff_inverse_growth_demo_summary \
  tests/integration/examples/test_examples.py::test_autodiff_twomode_demo_summary \
  tests/release/test_release_gates.py
```

The collision release contract validates the tracked compact artifact. Its
high-precision builder is intentionally not rerun in every hosted quick test.

## Change policy

A refactor should retain these values within their stated gates. An approved
bug, normalization, model, or numerical-method correction may intentionally
change a fingerprint only when the PR records the old and new values, explains
the equation or convention, adds an independent gate, and updates this ledger.
Changing a generated file or relaxing a tolerance to make CI green is not a
rebaseline.
