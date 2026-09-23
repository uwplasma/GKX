"""Smoke tests for the numbered example gallery under ``examples/NN_*``.

Every gallery script is executed top to bottom, the way a user runs it, in a
temporary working directory so its ``outputs/`` land there. Each run must
write a JSON summary whose numbers are all finite and a non-trivial figure.
Scripts run at their tutorial ``case.toml`` resolution; the few whose editable
constants set a longer calculation are turned down through those constants
(``SMOKE_KNOBS``), never through a hidden test-only switch. Scripts that need
an optional dependency or a multi-minute campaign are skipped with the reason
stated, so the gap shows in the run summary instead of disappearing.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import json
import math

import pytest

from support.paths import REPO_ROOT

EXAMPLES = REPO_ROOT / "examples"
GROUPS = sorted(
    p.name for p in EXAMPLES.iterdir() if p.is_dir() and p.name[:2].isdigit()
)
QHS_WOUT = EXAMPLES / "vmec" / "wout_NuhrenbergZille_1988_QHS.nc"


def _needs_vmex_or_wout() -> str | None:
    if QHS_WOUT.exists() or importlib.util.find_spec("vmex") is not None:
        return None
    return (
        "needs examples/vmec/wout_NuhrenbergZille_1988_QHS.nc or vmex to solve it "
        "(pip install vmex); CI installs neither"
    )


# relative script -> (summary JSON, figure) it must write under outputs/.
SMOKE_OUTPUTS: dict[str, tuple[str, str]] = {
    "01_linear_tokamak/run.py": ("summary.json", "linear_tokamak.png"),
    "02_linear_stellarator/run.py": ("summary.json", "linear_stellarator.png"),
    "03_nonlinear_tokamak/run.py": ("summary.json", "nonlinear_tokamak.png"),
    "04_nonlinear_stellarator/run.py": ("summary.json", "nonlinear_stellarator.png"),
    "05_kinetic_electrons/run.py": ("summary.json", "kinetic_electrons.png"),
    "06_electromagnetic/run.py": ("summary.json", "electromagnetic.png"),
    "07_collisions/run.py": ("summary.json", "collisions.png"),
    "08_quasilinear/run.py": ("summary.json", "quasilinear.png"),
    "08_quasilinear/implicit_sensitivity.py": (
        "quasilinear_implicit_sensitivity.json",
        "quasilinear_implicit_sensitivity.png",
    ),
    "09_autodiff/run.py": (
        "autodiff_inverse_twomode_summary.json",
        "autodiff_inverse_twomode.png",
    ),
    "11_parallel_scan/run.py": ("summary.json", "parallel_scan.png"),
    "12_restart_and_analysis/run.py": ("summary.json", "restart_and_analysis.png"),
}

# Editable constants turned down for the smoke run (name -> Python expression).
SMOKE_KNOBS: dict[str, dict[str, str]] = {
    "01_linear_tokamak/run.py": {"KY": "[0.2, 0.3]"},
    "02_linear_stellarator/run.py": {"KY": "[0.4, 0.6]"},
    "07_collisions/run.py": {"MODELS": '("lenard_bernstein", "coulomb_finite_kperp")'},
    "08_quasilinear/run.py": {"KY": "[0.2, 0.3]"},
    "09_autodiff/run.py": {"STEPS": "60", "GN_STEPS": "3"},
    "11_parallel_scan/run.py": {"KY": "[0.2, 0.3]"},
    "12_restart_and_analysis/run.py": {"STEPS_PER_LEG": "40"},
}

# Scripts that cannot run in a bounded CPU lane (reason, or a callable that
# returns a reason when the requirement is missing and None when it is met).
SMOKE_SKIPS: dict[str, object] = {
    "02_linear_stellarator/run.py": _needs_vmex_or_wout,
    "04_nonlinear_stellarator/run.py": _needs_vmex_or_wout,
    "09_autodiff/geometry_bridge.py": (
        "multi-minute vmex/booz_xform_jax gate campaign that rewrites the tracked "
        "docs/_static/differentiable_geometry_bridge.* artifacts; its imports are "
        "checked by test_gallery_first_party_imports_resolve"
    ),
    "10_vmex_optimization/run.py": (
        "needs vmex and its source-checkout seed deck "
        "examples/data/input.minimal_seed_nfp2; the default ladder is a GPU "
        "research calculation"
    ),
}


def _gallery_scripts() -> list[str]:
    return sorted(
        path.relative_to(EXAMPLES).as_posix()
        for group in GROUPS
        for path in (EXAMPLES / group).glob("*.py")
    )


def _skip_reason(relative: str) -> str | None:
    reason = SMOKE_SKIPS.get(relative)
    return reason() if callable(reason) else reason


def _with_knobs(source: str, knobs: dict[str, str], relative: str) -> str:
    """Return ``source`` with top-level ``NAME = ...`` assignments replaced."""

    tree = ast.parse(source)
    lines = source.splitlines()
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in knobs:
                first, last = node.lineno - 1, node.end_lineno - 1
                lines[first : last + 1] = [f"{target.id} = {knobs[target.id]}"] + [
                    "" for _ in range(last - first)
                ]
                seen.add(target.id)
    missing = set(knobs) - seen
    assert not missing, f"{relative} no longer defines knobs {sorted(missing)}"
    return "\n".join(lines) + "\n"


def _finite_numbers(payload) -> list[float]:
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return []
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, dict):
        return [x for value in payload.values() for x in _finite_numbers(value)]
    if isinstance(payload, (list, tuple)):
        return [x for value in payload for x in _finite_numbers(value)]
    return []


@pytest.fixture(autouse=True)
def _headless_matplotlib():
    import matplotlib

    previous = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        yield
    finally:
        import matplotlib.pyplot as plt

        plt.close("all")
        with contextlib.suppress(Exception):
            matplotlib.use(previous, force=True)


@pytest.mark.parametrize("relative", sorted(SMOKE_OUTPUTS))
def test_gallery_example_runs_and_writes_outputs(relative, tmp_path, monkeypatch):
    reason = _skip_reason(relative)
    if reason:
        pytest.skip(f"{relative}: {reason}")
    path = EXAMPLES / relative
    source = _with_knobs(
        path.read_text(encoding="utf-8"), SMOKE_KNOBS.get(relative, {}), relative
    )
    monkeypatch.chdir(tmp_path)
    # The script finds case.toml and ../vmec through __file__, so the edited
    # source runs under its original path.
    code = compile(source, str(path), "exec")
    exec(code, {"__name__": "__gallery_smoke__", "__file__": str(path)})

    summary_name, figure_name = SMOKE_OUTPUTS[relative]
    output = tmp_path / "outputs" / path.parent.name
    summary = json.loads((output / summary_name).read_text(encoding="utf-8"))
    numbers = _finite_numbers(summary)
    assert numbers, f"{relative} wrote a summary with no numbers"
    assert all(math.isfinite(x) for x in numbers), f"{relative} wrote non-finite values"
    figure = output / figure_name
    assert figure.is_file() and figure.stat().st_size > 10_000, (
        f"{relative} wrote no figure"
    )


def test_every_gallery_script_is_smoked_or_skipped_with_a_reason():
    scripts = set(_gallery_scripts())
    classified = set(SMOKE_OUTPUTS) | set(SMOKE_SKIPS)
    assert scripts - classified == set(), (
        "new gallery scripts need a smoke entry or a skip reason"
    )
    assert classified - scripts == set(), "registry names scripts that no longer exist"
    assert set(SMOKE_KNOBS) <= set(SMOKE_OUTPUTS)


def test_gallery_layout_matches_the_numbered_groups():
    assert GROUPS == [
        "01_linear_tokamak",
        "02_linear_stellarator",
        "03_nonlinear_tokamak",
        "04_nonlinear_stellarator",
        "05_kinetic_electrons",
        "06_electromagnetic",
        "07_collisions",
        "08_quasilinear",
        "09_autodiff",
        "10_vmex_optimization",
        "11_parallel_scan",
        "12_restart_and_analysis",
    ]
    extra = {p.name for p in EXAMPLES.iterdir() if p.is_dir()} - set(GROUPS)
    # vmec/ holds the shared VMEC input decks; outputs/ is a user's run directory.
    assert extra <= {"vmec", "outputs", "__pycache__"}
    for group in GROUPS:
        assert (EXAMPLES / group / "run.py").is_file(), group
    index = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    for group in GROUPS:
        assert f"{group}/" in index, f"examples/README.md does not list {group}"


@pytest.mark.parametrize("relative", _gallery_scripts())
def test_gallery_script_follows_the_example_style(relative):
    text = (EXAMPLES / relative).read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert ast.get_docstring(tree), (
        f"{relative} needs a module docstring with runtime and device"
    )
    assert "argparse" not in text
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert not functions & {"main", "_main"}, f"{relative} defines a main()"
    assert not any(
        isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        for node in tree.body
    ), f"{relative} hides its run behind a __main__ guard"


@pytest.mark.parametrize("relative", _gallery_scripts())
def test_gallery_first_party_imports_resolve(relative):
    """Every name a gallery script imports from gkx/tools must exist."""

    tree = ast.parse((EXAMPLES / relative).read_text(encoding="utf-8"))
    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"gkx", "tools"}:
                    importlib.import_module(alias.name)
                    checked += 1
        elif isinstance(node, ast.ImportFrom) and not node.level:
            owner = node.module or ""
            if owner.split(".")[0] not in {"gkx", "tools"}:
                continue
            module = importlib.import_module(owner)
            for alias in node.names:
                assert hasattr(module, alias.name), (
                    f"{relative}: {owner} has no {alias.name}"
                )
                checked += 1
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "gkx"
        ):
            assert hasattr(importlib.import_module("gkx"), node.attr), (
                f"{relative}: gkx.{node.attr} is gone"
            )
            checked += 1
    assert checked > 0, f"{relative} resolved no first-party name"
