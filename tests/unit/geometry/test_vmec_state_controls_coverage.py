"""Unit contracts for the vmex state-control helpers in ``vmec_state_controls``.

These exercise the pure coefficient accessor / replacement / index-resolution
controls that drive the differentiable VMEC-Boozer sensitivity gates.  Every
case uses synthetic ``SpectralState``-like frozen dataclasses and small arrays
so no real equilibrium solve is required; numeric expectations, resolved index
tuples and error messages are hand-derived from each helper's contract.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx.geometry.vmec_state_controls as controls


# ---------------------------------------------------------------------------
# Synthetic-input factories
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _FakeSpectralState:
    """Minimal frozen vmex-``SpectralState`` stand-in for replace/perturb gates.

    ``R_sin`` is a witness family that no tested control touches; it must
    survive ``dataclasses.replace`` unchanged.
    """

    R_cos: jnp.ndarray
    Z_sin: jnp.ndarray
    R_sin: jnp.ndarray


class _ClosableDataset:
    """netCDF-like handle that records exactly one ``close()``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _solved_case_bundle(r_cos: object, z_sin: object) -> tuple[object, ...]:
    """Return an ``(inp, state, runtime, wout)`` bundle like a solved vmex case."""

    inp = SimpleNamespace(tag="inp")
    runtime = SimpleNamespace(tag="runtime")
    wout = SimpleNamespace(tag="wout")
    state = SimpleNamespace(R_cos=r_cos, Z_sin=z_sin)
    return inp, state, runtime, wout


_SQUARE_LAYOUT_MESSAGE = (
    "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
)


# ---------------------------------------------------------------------------
# _new_boozer_object_with_auto_fallback: classic-reader fallback policy
# ---------------------------------------------------------------------------
def test_auto_fallback_closes_dataset_when_classic_reader_also_fails(monkeypatch):
    """A square-layout failure tries the classic reader, then re-raises + closes."""

    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    calls = {"new_booz": 0, "import_preferred": "<unset>"}

    def _always_square(_backend: object, _path: object) -> object:
        calls["new_booz"] += 1
        raise ValueError(_SQUARE_LAYOUT_MESSAGE)

    def _fake_import(preferred: str | None = None) -> object:
        calls["import_preferred"] = preferred
        return SimpleNamespace(name="classic-booz")

    monkeypatch.setattr(controls, "_new_booz_object", _always_square)
    monkeypatch.setattr(controls, "_import_booz_backend", _fake_import)

    nc_obj = _ClosableDataset()
    with pytest.raises(ValueError, match="rmnc0 has unexpected shape"):
        controls._new_boozer_object_with_auto_fallback(
            SimpleNamespace(name="jax-booz"), Path("square.nc"), nc_obj
        )

    # Primary attempt + classic fallback attempt both ran; the classic reader
    # was requested explicitly, and the dataset handle is closed on failure.
    assert calls["new_booz"] == 2
    assert calls["import_preferred"] == "booz_xform"
    assert nc_obj.closed is True


@pytest.mark.parametrize(
    ("env_backend", "error"),
    [
        # auto mode, but a non-square read failure must not touch the classic reader
        (None, ValueError("wout file is corrupt")),
        # square failure, but an explicitly forced backend stays fail-fast
        ("jax", ValueError(_SQUARE_LAYOUT_MESSAGE)),
    ],
)
def test_auto_fallback_skips_classic_reader_and_closes(monkeypatch, env_backend, error):
    """Only auto-mode square-layout failures may fall back to booz_xform."""

    if env_backend is None:
        monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    else:
        monkeypatch.setenv("GKX_BOOZ_BACKEND", env_backend)

    def _raise(_backend: object, _path: object) -> object:
        raise error

    def _forbidden_import(preferred: str | None = None) -> object:
        raise AssertionError("classic booz_xform reader must not be imported")

    monkeypatch.setattr(controls, "_new_booz_object", _raise)
    monkeypatch.setattr(controls, "_import_booz_backend", _forbidden_import)

    nc_obj = _ClosableDataset()
    with pytest.raises(ValueError):
        controls._new_boozer_object_with_auto_fallback(
            SimpleNamespace(name="jax-booz"), Path("wout.nc"), nc_obj
        )
    assert nc_obj.closed is True


# ---------------------------------------------------------------------------
# _load_vmec_state_context: solved-case wiring + 2-D validation
# ---------------------------------------------------------------------------
def test_load_vmec_state_context_exposes_differentiable_state_arrays(monkeypatch):
    r_cos = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    z_sin = np.array(
        [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]]
    )
    bundle = _solved_case_bundle(r_cos, z_sin)
    seen: dict[str, str] = {}

    def _fake_resolve(name: str) -> Path:
        seen["resolve"] = name
        return Path("input.synthetic")

    def _fake_load(name: str) -> tuple[object, ...]:
        seen["load"] = name
        return bundle

    monkeypatch.setattr(controls, "resolve_vmex_case_input_path", _fake_resolve)
    monkeypatch.setattr(controls, "load_solved_vmex_case", _fake_load)

    ctx = controls._load_vmec_state_context("synthetic_case")

    # The case name is forwarded (as a string) to both resolver and loader.
    assert seen == {"resolve": "synthetic_case", "load": "synthetic_case"}
    assert ctx.input_path == Path("input.synthetic")
    assert ctx.wout_path == controls.VMEC_STATE_IN_MEMORY_WOUT_PATH
    assert ctx.inp is bundle[0]
    assert ctx.state is bundle[1]
    assert ctx.runtime is bundle[2]
    assert ctx.wout is bundle[3]
    assert ctx.base_Rcos.shape == (3, 4)
    assert ctx.base_Zsin.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(ctx.base_Rcos), r_cos)
    np.testing.assert_allclose(np.asarray(ctx.base_Zsin), z_sin)


def test_load_vmec_state_context_rejects_non_2d_state_arrays(monkeypatch):
    bundle = _solved_case_bundle(np.ones(4), np.ones((3, 4)))  # R_cos is 1-D

    monkeypatch.setattr(
        controls, "resolve_vmex_case_input_path", lambda name: Path("x")
    )
    monkeypatch.setattr(controls, "load_solved_vmex_case", lambda name: bundle)

    with pytest.raises(RuntimeError, match="R_cos/Z_sin arrays must be two-dimensional"):
        controls._load_vmec_state_context("synthetic_case")


# ---------------------------------------------------------------------------
# _vmec_state_family_attribute + _vmec_boozer_state_array
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("family", "attribute"),
    [
        ("Rcos", "R_cos"),
        ("Rsin", "R_sin"),
        ("Zcos", "Z_cos"),
        ("Zsin", "Z_sin"),
        ("Lcos", "L_cos"),
        ("Lsin", "L_sin"),
    ],
)
def test_vmec_state_family_attribute_maps_public_family_to_state_attr(family, attribute):
    assert controls._vmec_state_family_attribute(family) == attribute


def test_vmec_state_family_attribute_rejects_unknown_family():
    with pytest.raises(ValueError, match="parameter_family must be one of"):
        controls._vmec_state_family_attribute("Bcos")


def test_vmec_boozer_state_array_returns_validated_family_table():
    state = SimpleNamespace(
        R_cos=np.arange(12.0).reshape(3, 4),
        Z_sin=np.arange(6.0).reshape(3, 2),
    )

    r_table = controls._vmec_boozer_state_array(state, "Rcos")
    assert r_table.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(r_table), np.arange(12.0).reshape(3, 4))

    z_table = controls._vmec_boozer_state_array(state, "Zsin")
    assert z_table.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(z_table), np.arange(6.0).reshape(3, 2))


def test_vmec_boozer_state_array_reports_missing_family_attribute():
    state = SimpleNamespace(Z_sin=np.ones((3, 4)))  # no R_cos attribute

    with pytest.raises(RuntimeError, match="vmex state does not expose R_cos"):
        controls._vmec_boozer_state_array(state, "Rcos")


@pytest.mark.parametrize("bad", [np.ones(4), np.ones((3, 1))])
def test_vmec_boozer_state_array_requires_two_dim_non_axisymmetric_mode(bad):
    state = SimpleNamespace(R_cos=bad)

    with pytest.raises(
        RuntimeError,
        match="R_cos array must expose at least one non-axisymmetric mode",
    ):
        controls._vmec_boozer_state_array(state, "Rcos")


# ---------------------------------------------------------------------------
# _replace_vmec_boozer_state_coefficient (get/replace round-trip)
# ---------------------------------------------------------------------------
def test_replace_vmec_boozer_state_coefficient_round_trips_single_entry():
    base = jnp.asarray(np.arange(12.0).reshape(3, 4))
    witness = jnp.asarray(np.full((3, 4), 7.0))
    state = _FakeSpectralState(R_cos=base, Z_sin=jnp.zeros((3, 4)), R_sin=witness)

    fetched = controls._vmec_boozer_state_array(state, "Rcos")
    np.testing.assert_allclose(np.asarray(fetched), np.asarray(base))

    new_state = controls._replace_vmec_boozer_state_coefficient(
        state, "Rcos", fetched, radial_index=2, mode_index=3, delta=0.25
    )

    expected = np.arange(12.0).reshape(3, 4)
    expected[2, 3] += 0.25
    np.testing.assert_allclose(
        np.asarray(controls._vmec_boozer_state_array(new_state, "Rcos")), expected
    )
    # Untouched family, witness field, and the original frozen state are intact.
    np.testing.assert_allclose(np.asarray(new_state.Z_sin), np.zeros((3, 4)))
    assert new_state.R_sin is witness
    np.testing.assert_allclose(np.asarray(state.R_cos), np.arange(12.0).reshape(3, 4))


# ---------------------------------------------------------------------------
# _vmec_boozer_state_parameter_name (mid-surface vs off-surface naming)
# ---------------------------------------------------------------------------
def test_vmec_boozer_state_parameter_name_switches_on_mid_surface():
    assert (
        controls._vmec_boozer_state_parameter_name(
            "Rcos", 4, 2, default_mid_surface=4
        )
        == "Rcos_mid_surface_m2"
    )
    assert (
        controls._vmec_boozer_state_parameter_name(
            "Zsin", 3, 1, default_mid_surface=4
        )
        == "Zsin_r3_m1"
    )


# ---------------------------------------------------------------------------
# _resolve_vmec_state_indices (default resolution, clamps, validation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "radial_index", "mode_index", "surface_index", "surface_grid", "expected"),
    [
        # Default radial index is ns // 2; each grid has its own default surface.
        ((8, 5), None, 2, None, "half_mesh", (4, 2, 3)),
        ((8, 5), None, 2, None, "field_line", (4, 2, 4)),
        ((8, 5), None, 2, None, "metric", (4, 2, 3)),
        # Explicit, in-range indices pass straight through.
        ((8, 5), 2, 1, 5, "metric", (2, 1, 5)),
        # Low-radius clamps: half_mesh/metric floor at 0, field_line floors at 1.
        ((8, 5), 0, 0, None, "half_mesh", (0, 0, 0)),
        ((2, 3), 0, 0, None, "metric", (0, 0, 0)),
        ((2, 3), 0, 0, None, "field_line", (0, 0, 1)),
    ],
)
def test_resolve_vmec_state_indices_resolves_defaults_and_clamps(
    shape, radial_index, mode_index, surface_index, surface_grid, expected
):
    base = jnp.zeros(shape)

    resolved = controls._resolve_vmec_state_indices(
        base,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid=surface_grid,
    )

    assert resolved == expected
    assert all(isinstance(value, int) for value in resolved)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            dict(radial_index=8, mode_index=0, surface_index=None, surface_grid="metric"),
            "radial_index is outside the VMEC state radial grid",
        ),
        (
            dict(radial_index=-1, mode_index=0, surface_index=None, surface_grid="metric"),
            "radial_index is outside the VMEC state radial grid",
        ),
        (
            dict(radial_index=None, mode_index=5, surface_index=None, surface_grid="metric"),
            "mode_index is outside the VMEC state mode table",
        ),
        (
            dict(radial_index=None, mode_index=0, surface_index=None, surface_grid="bogus"),
            "unknown VMEC surface grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=7,
                surface_grid="half_mesh",
            ),
            "half-mesh Boozer surface grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=8,
                surface_grid="field_line",
            ),
            "VMEC metric radial grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=-1,
                surface_grid="metric",
            ),
            "VMEC metric radial grid",
        ),
    ],
)
def test_resolve_vmec_state_indices_rejects_out_of_range_and_unknown_grid(
    kwargs, message
):
    base = jnp.zeros((8, 5))

    with pytest.raises(ValueError, match=message):
        controls._resolve_vmec_state_indices(base, **kwargs)


# ---------------------------------------------------------------------------
# _perturb_vmec_state (two-control perturbation + immutability)
# ---------------------------------------------------------------------------
def _perturb_context() -> tuple[object, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    base_Rcos = jnp.asarray(np.arange(1.0, 13.0).reshape(3, 4))
    base_Zsin = jnp.asarray(
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]])
    )
    witness = jnp.asarray(np.full((3, 4), 3.0))
    # The state tables deliberately differ from base_* so the assertions prove
    # the perturbation is applied to ctx.base_Rcos/base_Zsin, not ctx.state.*.
    state = _FakeSpectralState(
        R_cos=jnp.zeros((3, 4)), Z_sin=jnp.zeros((3, 4)), R_sin=witness
    )
    ctx = controls._VMECStateContext(
        input_path=Path("input.synthetic"),
        wout_path=controls.VMEC_STATE_IN_MEMORY_WOUT_PATH,
        inp=object(),
        runtime=object(),
        wout=object(),
        state=state,
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )
    return ctx, base_Rcos, base_Zsin, witness


def test_perturb_vmec_state_increments_two_controls_from_base_tables():
    ctx, base_Rcos, base_Zsin, witness = _perturb_context()
    x = jnp.asarray([0.5, -0.3])

    perturbed = controls._perturb_vmec_state(ctx, x, radial_index=1, mode_index=2)

    expected_Rcos = np.asarray(base_Rcos).copy()
    expected_Rcos[1, 2] += 0.5
    expected_Zsin = np.asarray(base_Zsin).copy()
    expected_Zsin[1, 2] += -0.3

    assert isinstance(perturbed, _FakeSpectralState)
    np.testing.assert_allclose(np.asarray(perturbed.R_cos), expected_Rcos)
    np.testing.assert_allclose(np.asarray(perturbed.Z_sin), expected_Zsin)
    # Only the [1, 2] entries moved (the [0, 0] control is unchanged base data).
    assert float(np.asarray(perturbed.R_cos)[0, 0]) == pytest.approx(1.0)
    # The untouched family survives and the original frozen state is unchanged.
    assert perturbed.R_sin is witness
    np.testing.assert_allclose(np.asarray(ctx.state.R_cos), np.zeros((3, 4)))


def test_perturb_vmec_state_and_context_are_immutable():
    ctx, _base_Rcos, _base_Zsin, _witness = _perturb_context()

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.base_Rcos = jnp.zeros((3, 4))

    perturbed = controls._perturb_vmec_state(
        ctx, jnp.asarray([0.0, 0.0]), radial_index=0, mode_index=1
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        perturbed.R_cos = jnp.zeros((3, 4))


# ---------------------------------------------------------------------------
# _length_two_params (default fill + length-2 validation)
# ---------------------------------------------------------------------------
def test_length_two_params_fills_default_and_preserves_length_two_vectors():
    filled = controls._length_two_params(None, 2.5)
    assert filled.shape == (2,)
    assert np.asarray(filled).dtype == np.float64  # x64: forced float64 contract
    np.testing.assert_allclose(np.asarray(filled), [2.5, 2.5])

    passed = controls._length_two_params(jnp.asarray([0.1, -0.2]), 0.0)
    np.testing.assert_allclose(np.asarray(passed), [0.1, -0.2])


@pytest.mark.parametrize(
    "params",
    [jnp.asarray([1.0]), jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([[1.0, 2.0]])],
)
def test_length_two_params_rejects_non_length_two_vectors(params):
    with pytest.raises(ValueError, match="params must be a length-2 vector"):
        controls._length_two_params(params, 0.0)
