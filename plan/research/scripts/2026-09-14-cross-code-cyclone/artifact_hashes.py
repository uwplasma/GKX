#!/usr/bin/env python3
"""Print the Q20 artifact SHA-256 list (first 16 hex digits) as markdown bullets, for the log entry.

    python artifact_hashes.py   (run from this directory)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
files = sorted(p for p in here.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
for p in files:
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"- `{p.relative_to(here)}` {digest}")
