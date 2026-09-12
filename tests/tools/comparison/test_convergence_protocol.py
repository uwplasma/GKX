"""A measured finer-grid disagreement must veto convergence certification."""

import pytest
from support.paths import load_tool_script


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 1.01, 1.02], 0),
        ([1.0, 1.01, 1.02, 0.75], None),  # early false plateau
        ([1.0, 1.04, 1.08, 1.12], None),  # accumulated sub-tolerance drift
        ([1.0, 0.75, 0.751, 0.752], 1),  # genuinely settled suffix
        ([1.0, 1.01], None),  # two refinements are required
        ([1.0, 1.01, 1.02, float("nan")], None),
        ([1.0, 1.01, 1.02, float("inf")], None),
        ([0.0, 0.0, 0.0], 0),
    ],
)
def test_refinement_requires_a_consistent_finite_suffix(values, expected):
    result = load_tool_script("campaigns", "convergence_protocol").refine(
        "velocity", range(len(values)), values.__getitem__, verbose=False
    )
    assert result.converged_value == expected
    assert result.to_dict()["converged"] == (expected is not None)
