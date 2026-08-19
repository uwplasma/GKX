"""Plot the measured nonlinear-adjoint memory and AD/FD validation ladder."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_static" / "nonlinear_autodiff_validation.png"

# XLA compiler profiles at N=2048 (MB) and the saturated Cyclone RK3 ladder.
memory = np.asarray([[758.63, 12.63], [11_879.86, 168.45]])
steps = np.asarray([64, 128, 256, 512, 1024, 2048])
adjoint = np.asarray([40.43197, 68.48254, 102.58960, 153.50523, 183.25259, 3934.65999])
finite_difference = np.asarray([40.43197, 68.48254, 102.58960, 153.50523, 183.25259, 3934.75763])

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
x = np.arange(2)
axes[0].bar(x - 0.18, memory[:, 0], 0.36, label="step checkpoint")
axes[0].bar(x + 0.18, memory[:, 1], 0.36, label="block checkpoint")
axes[0].set(xticks=x, xticklabels=("CPU", "RTX A4000"), ylabel="XLA temporary memory [MB]", yscale="log")
axes[0].legend(frameon=False)

axes[1].plot(steps, adjoint, "o-", label="discrete adjoint")
axes[1].plot(steps, finite_difference, "x--", label="centered FD")
axes[1].axvspan(1024, 2048, color="0.9", label="divergence knee")
axes[1].set(xlabel="window steps", ylabel=r"$|d\langle Q\rangle/dp|$", xscale="log", yscale="log")
axes[1].legend(frameon=False)

fig.savefig(OUT, dpi=180)
print(OUT)
