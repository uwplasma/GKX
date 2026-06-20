"""Default executable demo workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spectraxgk.workflows.runtime.results import RuntimeLinearResult


DEFAULT_DEMO_SETTINGS: dict[str, float | int | str] = {
    "ky": 0.3,
    "Nl": 7,
    "Nm": 14,
    "solver": "time",
    "method": "rk4",
    "dt": 0.03,
    "steps": 500,
    "sample_stride": 5,
    "fit_signal": "phi",
    "mode_method": "z_index",
}


@dataclass(frozen=True)
class DefaultDemoDeps:
    """Patchable dependencies for the no-input executable demo."""

    load_case_from_toml: Callable[..., tuple[str, Any, dict[str, Any]]]
    run_cyclone_linear: Callable[..., Any]
    cyclone_base_case: Callable[[], Any]
    build_spectral_grid: Callable[[Any], Any]
    extract_mode_time_series: Callable[..., Any]
    extract_eigenfunction: Callable[..., Any]
    normalize_eigenfunction: Callable[..., Any]
    linear_runtime_panel_figure: Callable[..., tuple[Any, Any]]
    write_runtime_linear_artifacts: Callable[[str | Path, RuntimeLinearResult], dict[str, str]]


@dataclass(frozen=True)
class _DefaultDemoCase:
    cfg: Any
    source: str


@dataclass(frozen=True)
class _DefaultDemoRunSettings:
    ky: float
    n_laguerre: int
    n_hermite: int
    solver: str
    method: str
    dt: float
    steps: int
    sample_stride: int
    mode_method: str
    fit_signal: str


@dataclass(frozen=True)
class _DefaultDemoOutputs:
    t: np.ndarray
    signal: np.ndarray
    z: np.ndarray
    eigenfunction: np.ndarray


__all__ = [
    "DEFAULT_DEMO_SETTINGS",
    "DefaultDemoDeps",
    "default_demo_artifact_base",
    "default_demo_plot_path",
    "default_demo_toml_path",
    "default_demo_toml_text",
    "default_example_config_path",
    "run_default_linear_demo",
]


def default_example_config_path() -> Path | None:
    """Return the bundled Cyclone demo TOML when running from a checkout."""

    root = Path(__file__).resolve().parents[3]
    path = root / "examples" / "linear" / "axisymmetric" / "cyclone.toml"
    return path if path.exists() else None


def default_demo_plot_path() -> Path:
    """Default figure path for the no-input executable demo."""

    return Path("spectraxgk_default_linear.png")


def default_demo_artifact_base() -> Path:
    """Default artifact stem for the no-input executable demo."""

    return Path("spectraxgk_default_linear")


def default_demo_toml_path() -> Path:
    """Default reproducer TOML path for the no-input executable demo."""

    return Path("spectraxgk_default_linear.toml")


def default_demo_toml_text() -> str:
    """Build the reproducer TOML emitted by the no-input executable demo."""

    settings = DEFAULT_DEMO_SETTINGS
    return f"""# Reproducer for the no-input `spectraxgk` educational demo.
# Run with:
#   spectraxgk run-linear --config spectraxgk_default_linear.toml --progress

case = "cyclone"

[grid]
Nx = 1
Ny = 24
Nz = 96
Lx = 62.8
Ly = 62.8
boundary = "linked"
y0 = 20.0
ntheta = 32
nperiod = 2

[time]
t_max = {float(settings["dt"]) * int(settings["steps"]):.6g}
dt = {settings["dt"]}
method = "{settings["method"]}"
sample_stride = {settings["sample_stride"]}
progress_bar = true

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[model]
R_over_LTi = 2.49
R_over_LTe = 0.0
R_over_Ln = 0.8
nu_i = 0.0

[init]
init_field = "density"
init_amp = 1.0e-10
gaussian_init = true
gaussian_width = 0.5

[terms]
streaming = 1.0
mirror = 1.0
curvature = 1.0
gradb = 1.0
diamagnetic = 1.0
collisions = 1.0
hypercollisions = 1.0
end_damping = 1.0
apar = 1.0
bpar = 0.0

[run]
ky = {settings["ky"]}
Nl = {settings["Nl"]}
Nm = {settings["Nm"]}
solver = "{settings["solver"]}"
method = "{settings["method"]}"
dt = {settings["dt"]}
steps = {settings["steps"]}
sample_stride = {settings["sample_stride"]}

[fit]
fit_signal = "{settings["fit_signal"]}"
mode_method = "{settings["mode_method"]}"
auto_window = true
window_fraction = 0.4
start_fraction = 0.2
min_points = 25
"""


def _status_printer(prefix: str) -> Callable[[str], None]:
    def _emit(message: str) -> None:
        print(f"{prefix}: {message}", flush=True)

    return _emit


def _load_default_demo_case(
    *, deps: DefaultDemoDeps, example_path: Path | None
) -> _DefaultDemoCase:
    bundled_example = default_example_config_path() if example_path is None else example_path
    if bundled_example is not None:
        _case_name, cfg, _data = deps.load_case_from_toml(str(bundled_example), None)
        return _DefaultDemoCase(cfg=cfg, source=str(bundled_example))
    return _DefaultDemoCase(
        cfg=deps.cyclone_base_case(),
        source=(
            "built-in Cyclone defaults "
            "(equivalent to examples/linear/axisymmetric/cyclone.toml)"
        ),
    )


def _default_demo_run_settings() -> _DefaultDemoRunSettings:
    settings = DEFAULT_DEMO_SETTINGS
    return _DefaultDemoRunSettings(
        ky=float(settings["ky"]),
        n_laguerre=int(settings["Nl"]),
        n_hermite=int(settings["Nm"]),
        solver=str(settings["solver"]),
        method=str(settings["method"]),
        dt=float(settings["dt"]),
        steps=int(settings["steps"]),
        sample_stride=int(settings["sample_stride"]),
        mode_method=str(settings["mode_method"]),
        fit_signal=str(settings["fit_signal"]),
    )


def _print_default_demo_intro(
    *,
    source: str,
    settings: _DefaultDemoRunSettings,
    toml_path: Path,
) -> None:
    print(
        "No input file specified; running the default Cyclone initial-value demo.",
        flush=True,
    )
    print(f"source={source}", flush=True)
    print(
        "This first run may include JAX compilation; progress lines show elapsed "
        "time and ETA once the time loop starts.",
        flush=True,
    )
    print(
        f"demo settings: ky={settings.ky:.3f} Nl={settings.n_laguerre} "
        f"Nm={settings.n_hermite} solver={settings.solver} "
        f"method={settings.method} dt={settings.dt:g} steps={settings.steps} "
        f"sample_stride={settings.sample_stride}",
        flush=True,
    )
    print(f"wrote reproducible input: {toml_path}", flush=True)


def _run_default_demo_solver(
    *,
    deps: DefaultDemoDeps,
    case: _DefaultDemoCase,
    settings: _DefaultDemoRunSettings,
) -> Any:
    return deps.run_cyclone_linear(
        ky_target=settings.ky,
        cfg=case.cfg,
        Nl=settings.n_laguerre,
        Nm=settings.n_hermite,
        solver=settings.solver,
        method=settings.method,
        dt=settings.dt,
        steps=settings.steps,
        sample_stride=settings.sample_stride,
        show_progress=True,
        status_callback=_status_printer("demo"),
        fit_signal=settings.fit_signal,
        mode_method=settings.mode_method,
        auto_window=True,
        window_fraction=0.4,
        start_fraction=0.2,
        min_points=25,
    )


def _extract_default_demo_outputs(
    *,
    deps: DefaultDemoDeps,
    case: _DefaultDemoCase,
    result: Any,
    settings: _DefaultDemoRunSettings,
) -> _DefaultDemoOutputs:
    grid = deps.build_spectral_grid(case.cfg.grid)
    signal = deps.extract_mode_time_series(
        result.phi_t, result.selection, method=settings.mode_method
    )
    eigen = deps.normalize_eigenfunction(
        deps.extract_eigenfunction(
            result.phi_t,
            result.t,
            result.selection,
            z=np.asarray(grid.z, dtype=float),
            method="svd",
        ),
        np.asarray(grid.z, dtype=float),
    )
    return _DefaultDemoOutputs(
        t=np.asarray(result.t, dtype=float),
        signal=np.asarray(signal),
        z=np.asarray(grid.z, dtype=float),
        eigenfunction=np.asarray(eigen),
    )


def _write_default_demo_plot(
    *,
    deps: DefaultDemoDeps,
    result: Any,
    outputs: _DefaultDemoOutputs,
) -> Path:
    out_path = default_demo_plot_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _axes = deps.linear_runtime_panel_figure(
        t=outputs.t,
        signal=outputs.signal,
        z=outputs.z,
        eigenfunction=outputs.eigenfunction,
        gamma=float(result.gamma),
        omega=float(result.omega),
        title="SPECTRAX-GK default Cyclone initial-value demo",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def _write_default_demo_artifacts(
    *,
    deps: DefaultDemoDeps,
    result: Any,
    outputs: _DefaultDemoOutputs,
) -> dict[str, str]:
    return deps.write_runtime_linear_artifacts(
        default_demo_artifact_base(),
        RuntimeLinearResult(
            ky=float(result.ky),
            gamma=float(result.gamma),
            omega=float(result.omega),
            selection=result.selection,
            t=outputs.t,
            signal=outputs.signal,
            z=outputs.z,
            eigenfunction=outputs.eigenfunction,
        ),
    )


def _print_default_demo_results(
    *,
    result: Any,
    bundle_paths: dict[str, str],
    out_path: Path,
    toml_path: Path,
) -> None:
    print(
        f"gamma={float(result.gamma):.6f} omega={float(result.omega):.6f}",
        flush=True,
    )
    print(f"saved {bundle_paths['summary']}", flush=True)
    if "timeseries" in bundle_paths:
        print(f"saved {bundle_paths['timeseries']}", flush=True)
    if "eigenfunction" in bundle_paths:
        print(f"saved {bundle_paths['eigenfunction']}", flush=True)
    print(f"saved {out_path}", flush=True)
    print(
        f"rerun this numerical case with: spectraxgk run-linear --config {toml_path} --progress",
        flush=True,
    )


def run_default_linear_demo(*, deps: DefaultDemoDeps, example_path: Path | None = None) -> int:
    """Run the no-input Cyclone initial-value demo and write reusable artifacts."""

    case = _load_default_demo_case(deps=deps, example_path=example_path)
    settings = _default_demo_run_settings()
    toml_path = default_demo_toml_path()
    toml_path.write_text(default_demo_toml_text(), encoding="utf-8")
    _print_default_demo_intro(
        source=case.source,
        settings=settings,
        toml_path=toml_path,
    )
    result = _run_default_demo_solver(deps=deps, case=case, settings=settings)
    outputs = _extract_default_demo_outputs(
        deps=deps,
        case=case,
        result=result,
        settings=settings,
    )
    out_path = _write_default_demo_plot(deps=deps, result=result, outputs=outputs)
    bundle_paths = _write_default_demo_artifacts(
        deps=deps,
        result=result,
        outputs=outputs,
    )
    _print_default_demo_results(
        result=result,
        bundle_paths=bundle_paths,
        out_path=out_path,
        toml_path=toml_path,
    )
    return 0
