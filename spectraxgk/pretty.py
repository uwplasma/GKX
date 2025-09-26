# spectraxgk/pretty.py
from __future__ import annotations
import os
from typing import Any, Dict

# Detect Rich availability (optional dependency)
_RICH_AVAIL = False
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    _RICH_AVAIL = True
except Exception:  # pragma: no cover
    pass

# Runtime switches (set by init_pretty)
_USE_RICH = False
_console = None  # type: ignore[misc]

def info_line(msg: str) -> None:
    """Print a single status line (respects Rich/NO_COLOR settings)."""
    if _USE_RICH:
        _console.print(msg)  # style can be added later if desired
    else:
        print(msg)


def init_pretty(prefer_rich: bool = True) -> None:
    """Initialize pretty output.
    - Disables color if NO_COLOR is set (case-insensitive).
    - Uses Rich only if available, preferred, and NO_COLOR is not set.
    - Prints a minimal tip if Rich is preferred but not installed.
    """
    global _USE_RICH, _console

    # Respect NO_COLOR (https://no-color.org/) — any case
    no_color_env = any(k.upper() == "NO_COLOR" for k in os.environ.keys())
    if no_color_env:
        _USE_RICH = False
        _console = None
        return

    if prefer_rich and _RICH_AVAIL:
        _USE_RICH = True
        _console = Console()
    else:
        _USE_RICH = False
        _console = None
        # Minimal, one-time hint if user wanted rich but it's missing
        if prefer_rich and not _RICH_AVAIL:
            print("Note: nicer terminal output available via `pip install rich`.")


def _sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def _estimate_sizes(cfg) -> dict:
    sim = cfg.sim.__dict__; grid = cfg.grid.__dict__
    Nn, Nm = grid.get("Nn", 0), grid.get("Nm", 0)
    nt = sim.get("nt", 0)
    prec = sim.get("precision", "x64").lower()
    bytes_per_float = 8 if prec == "x64" else 4
    bytes_per_complex = 2 * bytes_per_float
    state_len_real = 2 * Nn * Nm
    out_bytes = nt * Nn * Nm * bytes_per_complex
    return dict(
        Nn=Nn, Nm=Nm, nt=nt, prec=prec,
        state_len_real=state_len_real, out_bytes=out_bytes,
    )


def print_preflight(path: str, cfg) -> None:
    sim = cfg.sim.__dict__; grid = cfg.grid.__dict__
    sizes = _estimate_sizes(cfg)
    out_npz = os.path.join(cfg.paths.outdir, cfg.paths.outfile)
    out_png = os.path.splitext(out_npz)[0] + "_summary.png"

    if _USE_RICH:
        header = Panel.fit(
            "[bold cyan]SPECTRAX-GK preflight[/bold cyan]\n"
            f"[dim]input:[/dim] {path}",
            border_style="cyan", box=ROUNDED
        )
        _console.print(header)

        t = Table(box=ROUNDED, show_lines=False)
        t.add_column("Key", style="bold dim", no_wrap=True)
        t.add_column("Value")
        t.add_row("mode/backend", f"{sim.get('mode','?')} / {sim.get('backend','?')}")
        t.add_row("precision", sizes["prec"])
        t.add_row("solver", sim.get("solver", "tsit5"))
        if sim.get("adaptive", True):
            t.add_row("stepsize", f"adaptive (rtol={sim.get('rtol')}, atol={sim.get('atol')})")
        else:
            t.add_row("stepsize", f"constant (dt={sim.get('dt')})")
        t.add_row("tmax, nt", f"{sim.get('tmax','?')}, {sizes['nt']}")
        t.add_row("Hermite/Laguerre", f"Nn={sizes['Nn']}, Nm={sizes['Nm']}")
        t.add_row("k_parallel, vth", f"{grid.get('kpar','?')}, {grid.get('vth','?')}")
        t.add_row("nu (LB)", f"{grid.get('nu','?')}")
        if grid.get("es_drive", False):
            t.add_row("ES drive", f"enabled (coef={grid.get('e_coef', 1.0)})")
        else:
            t.add_row("ES drive", "disabled")
        t.add_row("state length", f"{sizes['state_len_real']} reals (packed Re/Im)")
        t.add_row("est. result size", _sizeof_fmt(sizes["out_bytes"]))
        t.add_row("will write (npz)", out_npz)
        t.add_row("will write (png)", out_png)
        _console.print(t)
        _console.print()  # blank line
    else:
        print("\n" + "-" * 64)
        print(" SPECTRAX-GK preflight")
        print("-" * 64)
        print(f" input file       : {path}")
        print(f" mode/backend     : {sim.get('mode','?')} / {sim.get('backend','?')}")
        print(f" precision        : {sizes['prec']}")
        print(f" tmax, nt         : {sim.get('tmax','?')}, {sizes['nt']}")
        print(f" Hermite/Laguerre : Nn={sizes['Nn']}, Nm={sizes['Nm']}")
        print(f" k_parallel, vth  : {grid.get('kpar','?')}, {grid.get('vth','?')}")
        print(f" nu (LB)          : {grid.get('nu','?')}")
        print(" ES drive         : enabled (coef=%s)" % grid.get("e_coef", 1.0)
              if grid.get("es_drive", False) else " ES drive         : disabled")
        print(f" state length     : {sizes['state_len_real']} reals (packed Re/Im)")
        print(f" est. result size : {_sizeof_fmt(sizes['out_bytes'])}")
        print(f" will write (npz) : {out_npz}")
        print(f" will write (png) : {out_png}")
        print("-" * 64 + "\n")


def print_summary(cfg, info: Dict[str, Any], elapsed_s: float) -> None:
    sim = cfg.sim.__dict__; grid = cfg.grid.__dict__
    out = info.get("outfile", "<unknown>")
    png = info.get("summary", None)
    meta = info.get("meta", {})
    git = meta.get("git", None)

    if _USE_RICH:
        header = Panel.fit(
            "[bold green]SPECTRAX-GK run summary[/bold green]",
            border_style="green", box=ROUNDED
        )
        _console.print(header)

        t = Table(box=ROUNDED, show_lines=False)
        t.add_column("Key", style="bold dim", no_wrap=True)
        t.add_column("Value")
        # Keep it brief: outputs + timing (+ optional git)
        t.add_row("output (npz)", out)
        if png:
            t.add_row("summary (png)", png)
        if git:
            t.add_row("git", git)
        t.add_row("wall time", f"{elapsed_s:.3f} s")
        _console.print(t)
        _console.print()
    else:
        print("\n" + "=" * 64)
        print(" SPECTRAX-GK run summary")
        print("=" * 64)
        print(f" output (npz)     : {out}")
        if png:
            print(f" summary (png)    : {png}")
        if git:
            print(f" git              : {git}")
        print(f" wall time        : {elapsed_s:.3f} s")
        print("=" * 64 + "\n")