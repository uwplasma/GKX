"""Shared test helpers: solve-layer monkeypatching, runtime decks and stub results, field assertions, attribute patching."""

from __future__ import annotations

from collections.abc import Mapping
from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.analysis import ModeSelection
from gkx.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from gkx.terms.config import FieldState
from types import SimpleNamespace
from typing import Any
import importlib
import numpy as np
import pytest


SOLVE_LAYER_MODULES = (
    "gkx.runtime",
    "gkx.cli",
    "gkx.workflows.linear",
    "gkx.workflows.nonlinear",
    "gkx.workflows.runtime.artifacts",
    "gkx.workflows.runtime.commands",
    "gkx.workflows.runtime.diagnostics",
    "gkx.workflows.runtime.orchestration_scan",
    "gkx.workflows.runtime.startup",
    "gkx.workflows.runtime.toml",
    "gkx.artifacts.nonlinear_netcdf",
)


def patch_runtime(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """Replace ``name`` in every solve-layer module that binds it."""

    hits = 0
    for module_name in SOLVE_LAYER_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            hits += 1
    if not hits:
        raise AttributeError(f"no solve-layer module binds {name!r}")


_CIRCULAR_GEOMETRY = "q = 1.4\ns_hat = 0.8\nepsilon = 0.18\nR0 = 2.77778"


def runtime_deck(
    *,
    t_max: float = 0.2,
    time_extra: str = "",
    geometry: str = _CIRCULAR_GEOMETRY,
    nonlinear: bool = False,
    normalization: bool = True,
    tail: str = "",
) -> str:
    """Tiny single-ion Cyclone-like runtime TOML deck used by CLI tests."""

    physics = "nonlinear = true\n\n[terms]\nnonlinear = 1.0\n" if nonlinear else ""
    norm = (
        '[normalization]\ncontract = "cyclone"\ndiagnostic_norm = "none"\n'
        if normalization
        else ""
    )
    return f"""
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = {t_max}
dt = 0.01
method = "rk2"
{time_extra}
[geometry]
{geometry}

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
{physics}
{norm}
{tail}
"""


def linear_result(z_index: int = 1, **overrides) -> RuntimeLinearResult:
    fields = dict(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=z_index),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0, 2.0]),
    )
    fields.update(overrides)
    return RuntimeLinearResult(**fields)


def scan_stub(ky, gamma, omega, **extra):
    return type(
        "Scan",
        (),
        {
            "ky": np.asarray(ky),
            "gamma": np.asarray(gamma),
            "omega": np.asarray(omega),
            **extra,
        },
    )()


def simple_diag(
    *,
    t=(0.1,),
    dt=0.1,
    Wg=(1.0,),
    Wphi=(0.5,),
    heat=(0.0,),
    particle=(0.0,),
    energy=(1.5,),
    **extra,
) -> SimulationDiagnostics:
    n = len(t)
    zeros = np.zeros(n)
    return SimulationDiagnostics(
        t=np.asarray(t),
        dt_t=np.full(n, dt),
        dt_mean=np.asarray(dt),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=np.asarray(Wg),
        Wphi_t=np.asarray(Wphi),
        Wapar_t=zeros,
        heat_flux_t=np.asarray(heat),
        particle_flux_t=np.asarray(particle),
        energy_t=np.asarray(energy),
        **extra,
    )


def nonlinear_result(diag=None, t=None) -> RuntimeNonlinearResult:
    if t is None:
        t = diag.t if diag is not None else np.asarray([0.1])
    return RuntimeNonlinearResult(
        t=np.asarray(t), diagnostics=diag, ky_selected=0.2, kx_selected=0.0
    )


def zero_field_diag(t, *, dt_t=None, dt_mean=None, Wphi=None) -> SimulationDiagnostics:
    """Diagnostics with every channel zero except optional ``Wphi``."""

    t = np.asarray(t, dtype=float)
    z = np.zeros_like(t)
    return SimulationDiagnostics(
        t=t,
        dt_t=t if dt_t is None else np.asarray(dt_t, dtype=float),
        dt_mean=float(t[-1]) if dt_mean is None else float(dt_mean),
        gamma_t=z,
        omega_t=z,
        Wg_t=z,
        Wphi_t=z if Wphi is None else np.asarray(Wphi, dtype=float),
        Wapar_t=z,
        heat_flux_t=z,
        particle_flux_t=z,
        energy_t=z,
    )


def state_and_fields(grid, *, fill=np.ones):
    """A (state, FieldState) pair shaped for ``Nl=3, Nm=4`` on ``grid``."""

    shape = (grid.ky.size, grid.kx.size, grid.z.size)
    return (
        fill((1, 3, 4, *shape), dtype=np.complex64),
        FieldState(phi=np.ones(shape, dtype=np.complex64), apar=None, bpar=None),
    )


def patch_nonlinear_setup(
    monkeypatch, patch_runtime, geom, grid, *, make_params=None
) -> None:
    """Stub the nonlinear startup chain so only the integrator call is exercised."""

    if make_params is None:

        def make_params():
            return type("P", (), {"rho_star": np.asarray(1.0)})()

    patch_runtime(monkeypatch, "build_runtime_geometry", lambda _cfg: geom)
    patch_runtime(
        monkeypatch, "build_runtime_linear_params", lambda *a, **k: make_params()
    )
    patch_runtime(monkeypatch, "build_runtime_term_config", lambda _cfg: object())
    patch_runtime(
        monkeypatch, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    patch_runtime(
        monkeypatch,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )


def dotted_get(obj: Any, dotted: str) -> Any:
    """Resolve ``a.b[0].c`` style paths across attributes, dict keys and indices."""

    current = obj
    for part in dotted.split("."):
        index = None
        if part.endswith("]") and "[" in part:
            part, raw = part[:-1].split("[", 1)
            index = int(raw)
        if part:
            if isinstance(current, Mapping):
                current = current[part]
            else:
                current = getattr(current, part)
        if index is not None:
            current = current[index]
    return current


def assert_fields(obj: Any, expected: Mapping[str, Any], *, label: object = "") -> None:
    """Assert every dotted path in ``expected``; floats compare with approx."""

    for dotted, want in expected.items():
        got = dotted_get(obj, dotted)
        if isinstance(want, float) and not isinstance(want, bool):
            assert got == pytest.approx(want), (label, dotted, got)
        elif want is None or isinstance(want, bool):
            assert got is want, (label, dotted, got)
        else:
            assert got == want, (label, dotted, got)


_TERM_NAMES = (
    "streaming",
    "mirror",
    "curvature",
    "gradb",
    "diamagnetic",
    "collisions",
    "hypercollisions",
    "hyperdiffusion",
    "end_damping",
    "apar",
    "bpar",
    "nonlinear",
)


def patch_attrs(monkeypatch, target, **attrs) -> None:
    """Monkeypatch every ``name=value`` pair on a module object or dotted prefix."""
    for name, value in attrs.items():
        if isinstance(target, str):
            monkeypatch.setattr(f"{target}.{name}", value)
        else:
            monkeypatch.setattr(target, name, value)


def only_terms(**enabled: float):
    """A ``TermConfig`` with every switch off except the ones given."""
    from gkx.terms.config import TermConfig

    values = {name: 0.0 for name in _TERM_NAMES}
    values.update(enabled)
    return TermConfig(**values)


def spectral_grid(
    nx: int,
    ny: int,
    nz: int,
    *,
    boundary: str = "periodic",
    lx: float = 2.0 * np.pi,
    ly: float = 2.0 * np.pi,
    **extra,
):
    """Build a small spectral grid from ``GridConfig`` keywords."""
    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid

    return build_spectral_grid(
        GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=lx, Ly=ly, boundary=boundary, **extra)
    )


def ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)
