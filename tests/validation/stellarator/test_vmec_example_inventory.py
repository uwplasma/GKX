from __future__ import annotations

import subprocess
import tomllib

from support.paths import REPO_ROOT


REPO = REPO_ROOT
EXAMPLES = REPO / "examples"
VMEC_INPUTS = EXAMPLES / "vmec"


def _walk_vmec_file_values(obj: object) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "vmec_file" and isinstance(value, str):
                values.append(value)
            values.extend(_walk_vmec_file_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_walk_vmec_file_values(value))
    return values


def test_example_tomls_do_not_ship_geometry_placeholders() -> None:
    forbidden = ("$HSX_VMEC_FILE", "$W7X_VMEC_FILE", "/path/to", "pth to")
    offenders: list[str] = []

    for path in EXAMPLES.rglob("*.toml"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)} contains {token!r}")

    assert offenders == []


def test_vmec_backed_examples_point_to_generated_wouts_with_input_decks() -> None:
    missing: list[str] = []

    for path in EXAMPLES.rglob("*.toml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for vmec_file in _walk_vmec_file_values(data):
            if not vmec_file.endswith(".nc"):
                missing.append(
                    f"{path.relative_to(REPO)} has non-WOUT vmec_file={vmec_file!r}"
                )
                continue
            resolved = (path.parent / vmec_file).resolve()
            if resolved.parent != VMEC_INPUTS.resolve():
                missing.append(
                    f"{path.relative_to(REPO)} points outside examples/vmec: {vmec_file}"
                )
                continue
            stem = resolved.name.removeprefix("wout_").removesuffix(".nc")
            input_deck = VMEC_INPUTS / f"input.{stem}"
            if not input_deck.exists():
                missing.append(
                    f"{path.relative_to(REPO)} expects {resolved.name}, but {input_deck.relative_to(REPO)} is missing"
                )

    assert missing == []


def test_vmec_input_decks_are_small_text_inputs() -> None:
    inputs = sorted(VMEC_INPUTS.glob("input.*"))
    script_text = (VMEC_INPUTS / "generate_wouts.sh").read_text(encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "examples/vmec"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert {path.name for path in inputs} >= {
        "input.circular_tokamak",
        "input.NuhrenbergZille_1988_QHS",
        "input.nfp3_QI_fixed_resolution_final",
    }
    assert all(path.stat().st_size < 100_000 for path in inputs)
    assert [path for path in tracked if path.endswith(".nc")] == []
    assert "${input/input./wout_}.nc" in script_text
    for path in inputs:
        assert path.name in script_text
