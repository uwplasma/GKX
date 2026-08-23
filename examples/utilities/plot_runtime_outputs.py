#!/usr/bin/env python3
"""Plot a saved GKX runtime artifact bundle.

Loads the summary written by a runtime case (the directory or file passed to
``[output].path``) and writes the standard overview figure next to it, or to
``OUT`` when set.  Plotting only -- finishes in seconds.
"""

from __future__ import annotations

from pathlib import Path

from gkx.artifacts.plotting import plot_saved_output

# Path to a saved runtime artifact bundle; point this at the [output].path of a
# completed run (e.g. tools_out/cyclone_nonlinear_runtime).
RUN_PATH = Path("tools_out/cyclone_nonlinear_runtime")
OUT = None  # optional output figure path; None picks a default next to RUN_PATH

out = plot_saved_output(RUN_PATH, out=OUT)
print(f"Wrote {out}")
