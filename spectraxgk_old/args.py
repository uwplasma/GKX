from __future__ import annotations
import argparse
import os

DEFAULT_TOML = "examples/linear_slab.toml"


def resolve_input_path(ref: str | None) -> str:
    """Resolve a user-provided reference into a TOML filepath.

    Accepts:
      - full path ".../file.toml"
      - basename with extension "file.toml"
      - basename without extension "file"

    Search order:
      1) exact ref (as-is)
      2) <ref>.toml in CWD (if no .toml given)
      3) examples/<ref>.toml
      4) default examples/linear_slab.toml
    """
    if not ref:
        return DEFAULT_TOML
    if os.path.isfile(ref):
        return ref
    if not ref.endswith(".toml"):
        cand = f"{ref}.toml"
        if os.path.isfile(cand):
            return cand
        ex = os.path.join("examples", cand)
        if os.path.isfile(ex):
            return ex
    # last resort; may still be a valid path relative to CWD
    return ref


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SPECTRAX-GK from a TOML config")
    p.add_argument(
        "input",
        nargs="?",
        help="TOML path or base name (e.g. 'linear_slab' or 'path/to/case.toml')",
    )
    p.add_argument(
        "--input",
        dest="input_flag",
        help="Optional: TOML path or base name (same as positional)",
    )
    p.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Print preflight info and exit without running the simulation",
    )
    p.add_argument(
        "--no-rich",
        action="store_true",
        help="Force plain text output (ignore Rich even if installed)",
    )
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args and attach a resolved `path` field."""
    p = build_parser()
    args = p.parse_args(argv)
    ref = args.input_flag or args.input
    args.path = resolve_input_path(ref)
    return args