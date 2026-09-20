"""Contract: a declared velocity-space regularization must act, or be refused.

``nu_hyper_l`` and ``nu_hyper_lm`` reach the distribution only through the
constant-coefficient hypercollision branch -- ``_hypercollision_kz_source``
carries no Laguerre index at all.  With the shipped runtime defaults
(``hypercollisions_const = 0``, ``hypercollisions_kz = 1``) a deck that declared
a Laguerre sink therefore ran with no Laguerre sink and no diagnostic, which is
what plan §2.4 recorded as "``nu_hyper_l`` inert in that branch".

Two things are pinned here.  First, the refusal: declaring a Laguerre channel
while the branch that carries it is off is an error, not a silent no-op.
Second, ``nu_hyper_m_const``, the per-branch Hermite rate that makes plan
§0.5 (iii)'s "const-branch Hermite coefficient zeroed" expressible at all --
without it ``nu_hyper_m`` drives *both* branches, so a constant-branch Laguerre
sink could not be added beside GX's ``|k_z|`` Hermite hypercollisions without
damping Hermite twice.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import RuntimeCollisionConfig
from gkx.operators.linear.dissipation import hypercollisions_contribution
from gkx.operators.linear.params import LinearParams

NL = 6
NM = 8
SHAPE = (1, NL, NM, 2, 3, 4)
P_HYPER_L = 6.0
P_HYPER_M = 4.0


def _state() -> jnp.ndarray:
    rng = np.random.default_rng(20260920)
    return jnp.asarray(rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE))


def _ell() -> np.ndarray:
    return np.arange(NL, dtype=float).reshape(NL, 1, 1, 1, 1)


def _m() -> np.ndarray:
    return np.arange(NM, dtype=float).reshape(1, NM, 1, 1, 1)


def _common() -> dict:
    ell, m = _ell(), _m()
    return {
        "vth": jnp.asarray([1.0]),
        "nu_hyper": jnp.asarray(0.0),
        "nu_hyper_lm": jnp.asarray(0.0),
        "hyper_ratio": jnp.zeros((NL, NM, 1, 1, 1)),
        "ratio_l": jnp.asarray((ell / NL) ** P_HYPER_L),
        "ratio_m": jnp.asarray((m / NM) ** P_HYPER_M),
        "ratio_lm": jnp.asarray(((2 * ell + m) / (2 * NL + NM)) ** 6.0),
        "mask_const": jnp.asarray((ell + m) > 0),
        "mask_kz": jnp.asarray(m > 2),
        "m_pow": jnp.asarray((m / max(NM - 1, 1)) ** P_HYPER_M),
        "m_norm_kz_factor": jnp.asarray(1.0),
        "kz": jnp.asarray(np.fft.fftfreq(SHAPE[-1]) * 2.0 * np.pi),
        "kpar_scale": jnp.asarray(1.0),
        "weight": jnp.asarray(1.0),
    }


def _contribution(**overrides) -> jnp.ndarray:
    kwargs = dict(_common())
    kwargs.update(overrides)
    return hypercollisions_contribution(_state(), **kwargs)


def test_kz_branch_carries_no_laguerre_index() -> None:
    """The shipped default branch ignores ``nu_hyper_l`` entirely."""

    kz_only = {
        "nu_hyper_m": jnp.asarray(1.0),
        "hypercollisions_const": jnp.asarray(0.0),
        "hypercollisions_kz": jnp.asarray(1.0),
    }
    without = _contribution(nu_hyper_l=jnp.asarray(0.0), **kz_only)
    with_sink = _contribution(nu_hyper_l=jnp.asarray(0.5), **kz_only)
    assert jnp.array_equal(without, with_sink)


def test_declared_laguerre_sink_without_its_branch_is_refused() -> None:
    """A deck may not declare a sink the selected branch cannot apply."""

    for channel in ("nu_hyper_l", "nu_hyper_lm"):
        with pytest.raises(ValueError, match="silently inert"):
            RuntimeCollisionConfig(**{channel: 0.1})


def test_declared_laguerre_sink_with_its_branch_is_accepted() -> None:
    cfg = RuntimeCollisionConfig(
        nu_hyper_l=0.1,
        hypercollisions_const=1.0,
        hypercollisions_kz=1.0,
        nu_hyper_m_const=0.0,
    )
    assert cfg.nu_hyper_l == 0.1
    assert cfg.nu_hyper_m_const == 0.0
    # The shipped default stays a legal, sink-free configuration.
    assert RuntimeCollisionConfig().nu_hyper_l == 0.0


def test_const_branch_hermite_rate_defaults_to_the_shared_one() -> None:
    params = LinearParams()
    assert params.nu_hyper_m_const is None
    assert params.const_branch_nu_hyper_m() == params.nu_hyper_m
    assert replace(params, nu_hyper_m_const=0.0).const_branch_nu_hyper_m() == 0.0


def test_pure_laguerre_sink_is_exactly_the_declared_rate() -> None:
    """With the const-branch Hermite rate zeroed the sink is ``-Nl nu_l r_l G``."""

    sink = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(0.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(0.0),
    )
    common = _common()
    expected = (
        -(NL * 0.1 * common["ratio_l"])
        * jnp.where(common["mask_const"], 1.0, 0.0)
        * _state()
    )
    assert jnp.allclose(sink, expected, atol=0.0, rtol=0.0)


def test_sink_adds_to_the_kz_branch_without_touching_it() -> None:
    """The declared sink is additive: the ``|k_z|`` Hermite model is unchanged."""

    baseline = _contribution(
        nu_hyper_l=jnp.asarray(0.0),
        nu_hyper_m=jnp.asarray(1.0),
        hypercollisions_const=jnp.asarray(0.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    sink = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(0.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(0.0),
    )
    both = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(0.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    assert jnp.allclose(both, baseline + sink, atol=0.0, rtol=0.0)


def test_omitting_the_const_hermite_rate_reproduces_the_shared_coefficient() -> None:
    """Back-compatibility: ``None`` keeps ``nu_hyper_m`` driving both branches."""

    shared = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    explicit = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(1.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    assert jnp.allclose(shared, explicit, atol=0.0, rtol=0.0)


def test_sink_damps_the_laguerre_cutoff_hardest() -> None:
    """``r_l = (l/Nl)^p`` makes the rate monotone in ``l`` and zero at ``l=0``."""

    sink = np.asarray(
        _contribution(
            nu_hyper_l=jnp.asarray(0.5),
            nu_hyper_m=jnp.asarray(1.0),
            nu_hyper_m_const=jnp.asarray(0.0),
            hypercollisions_const=jnp.asarray(1.0),
            hypercollisions_kz=jnp.asarray(0.0),
        )
    )
    state = np.asarray(_state())
    rate = np.abs(sink / state).reshape(NL, -1).max(axis=1)
    assert rate[0] == 0.0
    assert np.all(np.diff(rate) > 0.0)
    assert rate[-1] == pytest.approx(NL * 0.5 * ((NL - 1) / NL) ** P_HYPER_L)


def test_linear_params_roundtrips_the_new_rate_through_the_pytree() -> None:
    import jax

    params = replace(LinearParams(), nu_hyper_m_const=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == params
