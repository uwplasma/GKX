r"""Unit: does the gradient field hold what its name says it holds?

The linear operator is driven by :math:`a/L_T` and :math:`a/L_n` -- the TOML's
``tprim`` and ``fprim``, copied through ``build_linear_params`` unscaled. The
fields carrying them were named ``R_over_LTi`` / ``R_over_Ln`` / ``R_over_LTe``,
and their defaults (``6.9``, ``2.2``) were the Cyclone case written in
:math:`R/L` while :class:`gkx.config.ModelConfig` defaulted to the same case
written in :math:`a/L` (``2.49``, ``0.8``). One name, two units, and a factor
:math:`R/a = 2.78` between them.

That mattered the moment a number had to be quoted: ``tools/campaigns/dimits_shift.py``
reports the linear ITG threshold as a multiplier on the shipped drive, and
turning it into the literature :math:`R/L_T` requires knowing which of the two
the field holds.

These cases pin the chain end to end on the shipped Cyclone case. What would
falsify them: a runtime path that rescales the drive on its way into
:class:`LinearParams`, a default that drifts off the shipped TOML, or a
:math:`R/a` conversion folded in somewhere instead of stated.
"""

from __future__ import annotations

import dataclasses
import tomllib
import warnings

import numpy as np
import pytest
from support.paths import REPO_ROOT

CYCLONE_TOML = (
    REPO_ROOT / "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
)

# Cyclone base case, as the literature quotes it.
LITERATURE_R_OVER_LT = 6.9
LITERATURE_R_OVER_LN = 2.2


@pytest.fixture(scope="module")
def cyclone_toml() -> dict:
    return tomllib.loads(CYCLONE_TOML.read_text(encoding="utf-8"))


def test_the_runtime_hands_the_operator_the_toml_tprim_unscaled(cyclone_toml):
    from gkx.workflows.runtime.startup import build_runtime_linear_params
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, _raw = load_runtime_from_toml(CYCLONE_TOML)
    params = build_runtime_linear_params(cfg)

    ion = cyclone_toml["species"][0]
    assert np.asarray(params.tprim).ravel()[0] == pytest.approx(float(ion["tprim"]))
    assert np.asarray(params.fprim).ravel()[0] == pytest.approx(float(ion["fprim"]))


def test_only_R_over_a_turns_the_stored_drive_into_the_literature_gradient(
    cyclone_toml,
):
    # The cyclone normalization contract sets a = 1, so R/a is the TOML's own R0.
    assert cyclone_toml["normalization"]["contract"] == "cyclone"
    r_over_a = float(cyclone_toml["geometry"]["R0"])
    ion = cyclone_toml["species"][0]

    assert float(ion["tprim"]) * r_over_a == pytest.approx(
        LITERATURE_R_OVER_LT, abs=0.02
    )
    assert float(ion["fprim"]) * r_over_a == pytest.approx(
        LITERATURE_R_OVER_LN, abs=0.03
    )
    # Stated, not folded in: the stored value is not the literature one.
    assert float(ion["tprim"]) != pytest.approx(LITERATURE_R_OVER_LT, abs=0.5)


def test_both_gradient_defaults_are_the_shipped_case_in_the_units_the_operator_reads(
    cyclone_toml,
):
    from gkx.config import ModelConfig
    from gkx.operators.linear.params import LinearParams

    ion = cyclone_toml["species"][0]
    assert LinearParams().tprim == pytest.approx(float(ion["tprim"]))
    assert LinearParams().fprim == pytest.approx(float(ion["fprim"]))
    assert ModelConfig().tprim_i == pytest.approx(float(ion["tprim"]))
    assert ModelConfig().fprim == pytest.approx(float(ion["fprim"]))


def test_no_public_gradient_name_claims_a_normalization_that_is_never_applied():
    from gkx import api
    from gkx.config import KineticElectronModelConfig, ModelConfig
    from gkx.operators.linear.params import LinearParams, Species

    classes = (LinearParams, Species, ModelConfig, KineticElectronModelConfig)
    named = {
        field.name for cls in classes for field in dataclasses.fields(cls)
    } | set(api.__all__)
    offenders = sorted(name for name in named if name.startswith("R_over_L"))
    assert offenders == []


def test_the_retired_names_still_work_and_say_why_they_are_wrong():
    from gkx.operators.linear.params import (
        DEPRECATED_LINEAR_PARAM_ALIASES,
        LinearParams,
    )

    assert DEPRECATED_LINEAR_PARAM_ALIASES == {
        "R_over_LTi": "tprim",
        "R_over_Ln": "fprim",
        "R_over_LTe": "tprim_e",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = LinearParams(R_over_LTi=6.9, R_over_Ln=2.2, R_over_LTe=1.0)
    assert params.tprim == 6.9
    assert params.fprim == 2.2
    assert params.tprim_e == 1.0
    assert len(caught) == 3
    assert all(issubclass(w.category, DeprecationWarning) for w in caught)
    # The warning has to name the unit, not just the replacement: a caller who
    # passed 6.9 meant R/L_T and is now getting a/L_T.
    assert "a/L" in str(caught[0].message)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert params.R_over_LTi == 6.9
    assert len(caught) == 1

    # dataclasses.replace routes through __init__, so the alias survives it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replaced = dataclasses.replace(params, R_over_Ln=3.0)
    assert replaced.fprim == 3.0
    assert replaced.tprim == 6.9
    assert len(caught) == 1
