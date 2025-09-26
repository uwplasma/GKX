from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Result:
    t: np.ndarray
    C: np.ndarray  # (nt, Nn, Nm) complex
    meta: dict[str, Any]
