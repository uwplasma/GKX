"""Small, reproducible no-argument executable demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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
}


@dataclass(frozen=True)
class DefaultDemoDeps:
    """Patchable runtime and output dependencies for the default demo."""

    load_runtime_from_toml: Callable[..., tuple[Any, dict[str, Any]]]
    run_runtime_linear: Callable[..., Any]
    linear_runtime_panel_figure: Callable[..., tuple[Any, Any]]
    write_runtime_linear_artifacts: Callable[[str | Path, Any], dict[str, str]]


__all__ = [
    "DEFAULT_DEMO_SETTINGS",
    "DefaultDemoDeps",
    "default_demo_artifact_base",
    "default_demo_plot_path",
    "default_demo_toml_path",
    "default_demo_toml_text",
    "run_default_linear_demo",
]


def default_demo_plot_path() -> Path:
    """Return the figure path produced by the no-input executable."""

    return Path("spectraxgk_default_linear.png")


def default_demo_artifact_base() -> Path:
    """Return the artifact stem produced by the no-input executable."""

    return Path("spectraxgk_default_linear")


def default_demo_toml_path() -> Path:
    """Return the reproducer path produced by the no-input executable."""

    return Path("spectraxgk_default_linear.toml")


def default_demo_toml_text() -> str:
    """Build the complete runtime TOML used by the educational demo."""

    settings = DEFAULT_DEMO_SETTINGS
    return f"""# Reproducer for the no-input `spectraxgk` demo.
# Run with: spectraxgk spectraxgk_default_linear.toml --progress

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
nu = 0.0
kinetic = true

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
model = "s-alpha"
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1.0e-10
gaussian_init = true
gaussian_width = 0.5

[physics]
linear = true
nonlinear = false
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
collisions = true
hypercollisions = true

[collisions]
nu_hermite = 1.0
nu_laguerre = 2.0
nu_hyper = 0.0
p_hyper = 4.0
hypercollisions_const = 1.0
damp_ends_amp = 0.1
damp_ends_widthfrac = 0.125

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

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
nonlinear = 0.0

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
auto_window = true
window_fraction = 0.4
start_fraction = 0.2
min_points = 25
"""


def _status(message: str) -> None:
    print(f"demo: {message}", flush=True)


def _print_intro(toml_path: Path) -> None:
    settings = DEFAULT_DEMO_SETTINGS
    print("No input specified; running the default Cyclone initial-value demo.", flush=True)
    print(
        "The first run includes JAX compilation; progress reports elapsed time and ETA.",
        flush=True,
    )
    print(
        f"ky={settings['ky']} Nl={settings['Nl']} Nm={settings['Nm']} "
        f"method={settings['method']} dt={settings['dt']} steps={settings['steps']}",
        flush=True,
    )
    print(f"wrote reproducible input: {toml_path}", flush=True)


def _write_plot(deps: DefaultDemoDeps, result: Any) -> Path:
    path = default_demo_plot_path()
    fig, _axes = deps.linear_runtime_panel_figure(
        t=result.t,
        signal=result.signal,
        z=result.z,
        eigenfunction=result.eigenfunction,
        gamma=float(result.gamma),
        omega=float(result.omega),
        title="SPECTRAX-GK default Cyclone initial-value demo",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def run_default_linear_demo(*, deps: DefaultDemoDeps) -> int:
    """Run one small runtime case and write its TOML, data, and figure locally."""

    settings = DEFAULT_DEMO_SETTINGS
    toml_path = default_demo_toml_path()
    toml_path.write_text(default_demo_toml_text(), encoding="utf-8")
    _print_intro(toml_path)
    cfg, raw = deps.load_runtime_from_toml(toml_path)
    fit = dict(raw.get("fit", {}))
    result = deps.run_runtime_linear(
        cfg,
        ky_target=float(settings["ky"]),
        Nl=int(settings["Nl"]),
        Nm=int(settings["Nm"]),
        solver=str(settings["solver"]),
        method=str(settings["method"]),
        dt=float(settings["dt"]),
        steps=int(settings["steps"]),
        sample_stride=int(settings["sample_stride"]),
        fit_signal=str(fit.pop("fit_signal", settings["fit_signal"])),
        show_progress=True,
        status_callback=_status,
        **fit,
    )
    paths = deps.write_runtime_linear_artifacts(default_demo_artifact_base(), result)
    plot_path = _write_plot(deps, result)
    print(f"gamma={float(result.gamma):.6f} omega={float(result.omega):.6f}", flush=True)
    for path in paths.values():
        print(f"saved {path}", flush=True)
    print(f"saved {plot_path}", flush=True)
    print(f"rerun with: spectraxgk {toml_path} --progress", flush=True)
    return 0
