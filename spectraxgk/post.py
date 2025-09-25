from __future__ import annotations
import json
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
from .types import Result


def load_result(path: str) -> Result:
    data = np.load(path, allow_pickle=True)
    meta = data["meta"].item() if isinstance(data["meta"].item, object) else dict(data["meta"]) # type: ignore
    return Result(t=data["t"], C=data["C"], meta=meta)

def plot_energy(res: Result):
    # Simple diagnostic: L2 norm of C as a proxy for energy
    E = np.sum(np.abs(res.C)**2, axis=(1, 2))
    plt.figure()
    plt.plot(res.t, E)
    plt.xlabel("t")
    plt.ylabel(r"$\\sum_{n,m} |C_{n,m}|^2$")
    plt.title("Hermite–Laguerre energy proxy")
    plt.grid(True)
    plt.show()