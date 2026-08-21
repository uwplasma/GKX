"""Focused tests for diffrax integrator helper and branch behavior."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

diffrax = pytest.importorskip("diffrax")
pytest.importorskip("equinox")

from gkx.diagnostics.analysis import ModeSelection, ModeSelectionBatch
from gkx.config import CycloneBaseCase, GridConfig
from gkx.solvers.time import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
    integrate_nonlinear_diffrax,
)
from gkx.solvers.time.diffrax_core import (
    _adjoint,
    _density_from_G_cached,
    _is_imex_solver,
    _is_implicit_solver,
    _pack_complex_state,
    _progress_meter,
    _require_diffrax,
    _solver_from_name,
    _stepsize_controller,
    _unpack_complex_state,
)
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.params import LinearParams
from gkx.operators.linear.params import Species, build_linear_params
from gkx.terms.config import TermConfig
from dataclasses import replace
import jax


def _tiny_diffrax_setup(nx: int = 2, ny: int = 3, nz: int = 8):
    grid_cfg = GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.0,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    G = jnp.ones(
        (2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    ) * (1.0e-6 + 1.0e-7j)
    return grid, geom, params, G


def test_adaptive_linear_forward_jvp_matches_centered_difference() -> None:
    grid, geom, params, _template = _tiny_diffrax_setup(nx=1, ny=2, nz=4)
    state = jnp.zeros(
        (1, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = state.at[0, 0, 1, 0, :].set(0.2 * jnp.exp(1j * z))
    probe = jnp.exp(1j * jnp.linspace(0.0, 1.0, grid.z.size))

    def objective(
        omega_star_scale,
        rtol=1.0e-3,
        *,
        derivative_mode="forward",
        checkpoint=False,
    ):
        final_state, _phi = integrate_linear_diffrax(
            state,
            grid,
            geom,
            replace(params, omega_star_scale=omega_star_scale),
            dt=0.02,
            steps=2,
            method="Tsit5",
            adaptive=True,
            rtol=rtol,
            atol=1.0e-6,
            max_steps=100,
            progress_bar=False,
            jit=False,
            derivative_mode=derivative_mode,
            checkpoint=checkpoint,
        )
        return jnp.real(jnp.vdot(probe, final_state[0, 0, 1, 0, :]))

    point = jnp.asarray(1.0, dtype=jnp.float32)
    value, tangent = jax.jvp(objective, (point,), (jnp.ones_like(point),))
    epsilon = jnp.asarray(5.0e-2, dtype=point.dtype)
    finite_difference = (objective(point + epsilon) - objective(point - epsilon)) / (
        2.0 * epsilon
    )
    tightened_value, tightened_tangent = jax.jvp(
        lambda x: objective(x, 3.0e-4),
        (point,),
        (jnp.ones_like(point),),
    )
    reverse_value, reverse_tangent = jax.value_and_grad(
        lambda x: objective(
            x,
            derivative_mode="reverse",
            checkpoint=True,
        )
    )(point)

    assert float(jnp.abs(tangent)) > 1.0e-4
    assert float(tangent) == pytest.approx(float(finite_difference), rel=1.0e-3)
    assert float(tightened_value) == pytest.approx(float(value), rel=2.0e-4)
    assert float(tightened_tangent) == pytest.approx(float(tangent), rel=2.0e-4)
    assert float(reverse_value) == pytest.approx(float(value), rel=2.0e-5)
    assert float(reverse_tangent) == pytest.approx(float(finite_difference), rel=1.0e-3)
    assert float(reverse_tangent) == pytest.approx(float(tangent), rel=2.0e-4)

    with pytest.raises(ValueError, match="derivative_mode"):
        integrate_linear_diffrax(
            state,
            grid,
            geom,
            params,
            dt=0.02,
            steps=1,
            derivative_mode="unsupported",
            progress_bar=False,
            jit=False,
        )


def test_diffrax_helper_functions() -> None:
    dfx, eqx = _require_diffrax()
    assert dfx is not None
    assert eqx is not None
    assert _is_imex_solver("imex")
    assert _is_imex_solver("KenCarp4")
    assert not _is_imex_solver("Tsit5")
    assert _is_implicit_solver("implicit")
    assert _is_implicit_solver("Kvaerno5")
    assert not _is_implicit_solver("Heun")
    assert _solver_from_name("rk2").__class__.__name__ == "Heun"
    assert _solver_from_name("semi-implicit").__class__.__name__ == "KenCarp4"
    assert (
        _stepsize_controller(False, 1.0e-5, 1.0e-7).__class__.__name__
        == "ConstantStepSize"
    )
    assert (
        _stepsize_controller(True, 1.0e-5, 1.0e-7).__class__.__name__ == "PIDController"
    )
    assert _adjoint(False).__class__.__name__ == "DirectAdjoint"
    assert _adjoint(True).__class__.__name__ == "RecursiveCheckpointAdjoint"
    assert _progress_meter(False).__class__.__name__ == "NoProgressMeter"
    assert "Tqdm" in _progress_meter(True).__class__.__name__
    with pytest.raises(ValueError):
        _solver_from_name("definitely-not-a-solver")


def test_require_diffrax_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import gkx.solvers.time.diffrax_core as di

    monkeypatch.setattr(di, "dfx", None)
    monkeypatch.setattr(di, "eqx", None)
    with pytest.raises(ImportError):
        di._require_diffrax()


def test_pack_unpack_complex_state_roundtrip() -> None:
    G = jnp.asarray([[1.0 + 2.0j, -3.0 + 0.5j]], dtype=jnp.complex64)
    packed = _pack_complex_state(G)
    unpacked = _unpack_complex_state(packed)
    assert packed.shape[-1] == 2
    assert jnp.allclose(unpacked, G)


def test_density_from_G_cached_all_branches() -> None:
    grid, geom, params, G5 = _tiny_diffrax_setup(nx=1, ny=2, nz=4)
    from gkx.operators.linear.cache_builder import build_linear_cache

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)
    out5 = _density_from_G_cached(G5, cache, density_species_index=None)
    assert out5.shape == (grid.ky.size, grid.kx.size, grid.z.size)

    G6 = jnp.stack([G5, 2.0 * G5], axis=0)
    out6_all = _density_from_G_cached(G6, cache, density_species_index=None)
    out6_one = _density_from_G_cached(G6, cache, density_species_index=1)
    assert out6_all.shape == out5.shape
    assert out6_one.shape == out5.shape

    cache_no_species = replace(cache, Jl=cache.Jl[0])
    out5_no_species = _density_from_G_cached(
        G5, cache_no_species, density_species_index=None
    )
    out6_no_species_all = _density_from_G_cached(
        G6, cache_no_species, density_species_index=None
    )
    out6_no_species_one = _density_from_G_cached(
        G6, cache_no_species, density_species_index=0
    )
    assert out5_no_species.shape == out5.shape
    assert out6_no_species_all.shape == out5.shape
    assert out6_no_species_one.shape == out5.shape


def test_integrate_linear_diffrax_mode_and_field_branches() -> None:
    grid, geom, params, G = _tiny_diffrax_setup()
    # save_field density branch + return_state False
    G_last, density_t = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Heun",
        adaptive=False,
        progress_bar=False,
        jit=False,
        return_state=False,
        save_field="density",
    )
    assert G_last is None
    assert density_t.shape[0] == 2

    # save_mode batch + mode_method max branch
    sel = ModeSelectionBatch(
        ky_indices=[0, min(1, grid.ky.size - 1)], kx_index=0, z_index=0
    )
    G_last2, mode_t = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        jit=False,
        return_state=True,
        save_field="phi",
        save_mode=sel,
        mode_method="max",
    )
    assert G_last2 is not None
    assert mode_t.shape[0] == 2

    # save_field phi+density branch
    _, pair_t = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        jit=False,
        return_state=False,
        save_field="phi+density",
    )
    assert isinstance(pair_t, tuple)
    assert len(pair_t) == 2

    # single mode extraction, z-index and max branches
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    _, mode_single_z = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        save_mode=sel,
        mode_method="z_index",
        jit=False,
        return_state=False,
    )
    _, mode_single_max = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        save_mode=sel,
        mode_method="max",
        jit=False,
        return_state=False,
    )
    assert mode_single_z.shape[0] == 1
    assert mode_single_max.shape[0] == 1

    _, mode_batch_z = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        save_mode=ModeSelectionBatch(ky_indices=[0, 1], kx_index=0, z_index=0),
        mode_method="z_index",
        jit=False,
        return_state=False,
    )
    assert mode_batch_z.shape[0] == 1

    # phi+density with return_state=True
    _, pair_state = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        jit=False,
        return_state=True,
        save_field="phi+density",
    )
    assert isinstance(pair_state, tuple)
    assert len(pair_state) == 2

    # IMEX branch in linear diffrax
    _, phi_imex = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="KenCarp4",
        adaptive=True,
        progress_bar=False,
        jit=False,
        return_state=False,
    )
    assert phi_imex.shape[0] == 1

    # jit=None path and device sharding branch
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    _, phi_jit_auto = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        jit=None,
        return_state=False,
        state_sharding=sharding,
    )
    assert phi_jit_auto.shape[0] == 1

    G6 = jnp.stack([G, 1.5 * G], axis=0)
    params_multi = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
            Species(
                charge=-1.0,
                mass=0.1,
                density=1.0,
                temperature=1.0,
                tprim=0.0,
                fprim=0.0,
            ),
        ],
        omega_d_scale=0.0,
        omega_star_scale=0.0,
    )
    _, phi_multi = integrate_linear_diffrax(
        G6,
        grid,
        geom,
        params_multi,
        dt=0.05,
        steps=1,
        method="Tsit5",
        adaptive=False,
        progress_bar=False,
        jit=False,
        return_state=False,
    )
    assert phi_multi.shape[0] == 1


def test_integrate_linear_diffrax_error_paths() -> None:
    grid, geom, params, G = _tiny_diffrax_setup()
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            jnp.zeros((2, 3, 4), dtype=jnp.complex64),
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            progress_bar=False,
            jit=False,
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            progress_bar=False,
            jit=False,
            sample_stride=0,
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=3,
            progress_bar=False,
            jit=False,
            sample_stride=2,
        )
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=1,
            progress_bar=False,
            jit=False,
            save_mode=sel,
            mode_method="bad",
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=1,
            progress_bar=False,
            jit=False,
            save_field="bad",
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=1,
            progress_bar=False,
            jit=False,
            save_field="phi+density",
            save_mode=sel,
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=1,
            progress_bar=False,
            jit=False,
            save_mode=ModeSelectionBatch(ky_indices=[0], kx_index=0, z_index=0),
            mode_method="bad",
        )


def test_integrate_linear_diffrax_streaming_phi_and_density_paths() -> None:
    grid, geom, params, G = _tiny_diffrax_setup(nx=1, ny=2, nz=8)
    _, gamma_d, omega_d = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Euler",
        fit_signal="density",
        mode_method="z_index",
        mode_ky_indices=0,
        progress_bar=False,
        jit=False,
        return_state=False,
    )
    assert gamma_d.shape[0] == 1
    assert omega_d.shape[0] == 1

    _, gamma_dmax, omega_dmax = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Euler",
        fit_signal="density",
        mode_method="max",
        mode_ky_indices=[0],
        progress_bar=False,
        jit=False,
        return_state=False,
    )
    assert gamma_dmax.shape[0] == 1
    assert omega_dmax.shape[0] == 1

    G_last, gamma_p, omega_p = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Euler",
        fit_signal="phi",
        mode_method="max",
        mode_ky_indices=[0, 1],
        progress_bar=False,
        jit=False,
        return_state=True,
    )
    assert G_last is not None
    assert gamma_p.shape[0] == 2
    assert omega_p.shape[0] == 2

    # IMEX + jit=True paths on streaming fit
    _, gamma_i, omega_i = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="KenCarp4",
        fit_signal="phi",
        mode_method="z_index",
        mode_ky_indices=[0],
        progress_bar=False,
        jit=True,
        return_state=False,
    )
    assert gamma_i.shape[0] == 1
    assert omega_i.shape[0] == 1

    # Multi-species path in streaming density extraction branches
    G6 = jnp.stack([G, 1.5 * G], axis=0)
    params_multi = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
            Species(
                charge=-1.0,
                mass=0.1,
                density=1.0,
                temperature=1.0,
                tprim=0.0,
                fprim=0.0,
            ),
        ],
        omega_d_scale=0.0,
        omega_star_scale=0.0,
    )
    _, gamma_6, omega_6 = integrate_linear_diffrax_streaming(
        G6,
        grid,
        geom,
        params_multi,
        dt=0.05,
        steps=1,
        method="Euler",
        fit_signal="density",
        mode_method="z_index",
        mode_ky_indices=[0],
        density_species_index=1,
        progress_bar=False,
        jit=False,
        return_state=False,
    )
    assert gamma_6.shape[0] == 1
    assert omega_6.shape[0] == 1


def test_integrate_linear_diffrax_streaming_error_paths() -> None:
    grid, geom, params, G = _tiny_diffrax_setup(nx=1, ny=2, nz=8)
    with pytest.raises(ValueError):
        integrate_linear_diffrax_streaming(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            method="Euler",
            mode_method="bad",
            progress_bar=False,
            jit=False,
        )
    with pytest.raises(ValueError):
        integrate_linear_diffrax_streaming(
            G,
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            method="Euler",
            fit_signal="bad",
            progress_bar=False,
            jit=False,
        )


def test_integrate_nonlinear_diffrax_explicit_and_imex() -> None:
    grid, geom, params, G = _tiny_diffrax_setup(nx=1, ny=2, nz=8)
    params_multi = build_linear_params(
        [
            Species(
                charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=0.0, fprim=0.0
            ),
            Species(
                charge=-1.0,
                mass=0.1,
                density=1.0,
                temperature=1.0,
                tprim=0.0,
                fprim=0.0,
            ),
        ],
        omega_d_scale=0.0,
        omega_star_scale=0.0,
    )
    # explicit/full branch
    G_last, fields = integrate_nonlinear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="Tsit5",
        terms=TermConfig(nonlinear=1.0),
        adaptive=False,
        progress_bar=False,
        jit=False,
    )
    assert G_last.shape == G.shape
    assert fields.phi.shape[0] == 2

    # imex branch with zero nonlinear
    G_last2, fields2 = integrate_nonlinear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="KenCarp4",
        terms=TermConfig(nonlinear=0.0),
        adaptive=True,
        progress_bar=False,
        jit=False,
        max_steps=1000,
    )
    assert G_last2.shape == G.shape
    assert fields2.phi.shape[0] == 2

    # jit=None and state-sharding branches
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    _, gamma_auto, omega_auto = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="Euler",
        fit_signal="phi",
        mode_method="z_index",
        mode_ky_indices=[0],
        progress_bar=False,
        jit=None,
        return_state=False,
        state_sharding=sharding,
    )
    assert gamma_auto.shape[0] == 1
    assert omega_auto.shape[0] == 1

    G_last3, fields3 = integrate_nonlinear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="KenCarp4",
        terms=TermConfig(nonlinear=0.0),
        adaptive=True,
        progress_bar=False,
        jit=None,
        state_sharding=sharding,
    )
    assert G_last3.shape == G.shape
    assert fields3.phi.shape[0] == 1

    G6 = jnp.stack([G, 2.0 * G], axis=0)
    G_last4, fields4 = integrate_nonlinear_diffrax(
        G6,
        grid,
        geom,
        params_multi,
        dt=0.05,
        steps=1,
        method="KenCarp4",
        terms=TermConfig(nonlinear=0.0),
        adaptive=True,
        progress_bar=False,
        jit=False,
    )
    assert G_last4.shape == G6.shape
    assert fields4.phi.shape[0] == 1

    with pytest.raises(ValueError):
        integrate_nonlinear_diffrax(
            jnp.zeros((2, 3, 4), dtype=jnp.complex64),
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            progress_bar=False,
            jit=False,
        )

    with pytest.raises(ValueError):
        integrate_linear_diffrax_streaming(
            jnp.zeros((2, 3, 4), dtype=jnp.complex64),
            grid,
            geom,
            params,
            dt=0.05,
            steps=2,
            method="Euler",
            progress_bar=False,
            jit=False,
        )


# Diffrax smoke contracts.
def test_integrate_linear_diffrax_runs():
    """Diffrax explicit integrator should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-6
    _, phi_t = integrate_linear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="Tsit5",
        progress_bar=False,
        adaptive=True,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_steps=20000,
        jit=False,
    )
    assert phi_t.shape[0] == 2


def test_integrate_linear_diffrax_jit_smoke():
    """One small JIT path smoke test to keep coverage."""
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-6
    with jax.disable_jit(False):
        _, phi_t = integrate_linear_diffrax(
            G,
            grid,
            geom,
            params,
            dt=0.1,
            steps=1,
            method="Tsit5",
            progress_bar=False,
            adaptive=False,
            rtol=1.0e-3,
            atol=1.0e-6,
            max_steps=100,
            jit=True,
        )
    assert phi_t.shape[0] == 1


def test_integrate_nonlinear_diffrax_imex_runs():
    """Diffrax IMEX integrator should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-6
    terms = TermConfig(nonlinear=1.0)
    _, fields_t = integrate_nonlinear_diffrax(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="KenCarp4",
        terms=terms,
        progress_bar=False,
        adaptive=True,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_steps=20000,
        jit=False,
    )
    assert fields_t.phi.shape[0] == 2


def test_integrate_linear_diffrax_streaming_runs():
    """Streaming fit path should run and return finite estimates."""
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-6
    _, gamma, omega = integrate_linear_diffrax_streaming(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="Euler",
        progress_bar=False,
        adaptive=False,
        rtol=1.0e-3,
        atol=1.0e-6,
        max_steps=100,
        jit=False,
        tmin=0.0,
        tmax=0.2,
        return_state=False,
    )
    assert gamma.shape[0] == grid.ky.size
    assert omega.shape[0] == grid.ky.size
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(jnp.isfinite(omega))
