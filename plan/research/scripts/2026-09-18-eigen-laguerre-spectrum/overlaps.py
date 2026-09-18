"""Q16: cross-rung and cross-nu eigenvector overlaps.

Distinguishes a branch change from a slowly converging single branch, the
measure the 2026-09-13 review used on the GX Nl24/Nl32 pair. Two overlaps are
reported for every pair:

``phi``   |<phi_a|phi_b>| / (||phi_a|| ||phi_b||) on the electrostatic potential
          phi(z), which lives on the same z grid at every Nl, so no truncation
          or embedding is involved. This is the resolution-independent measure.
``state`` the same quantity on the distribution eigenvectors restricted to the
          common Laguerre block ell < min(Nl_a, Nl_b) and renormalized there.
          It also reports ``retained_a``/``retained_b``, the fraction of each
          vector's 2-norm inside that block, because an overlap computed on a
          block holding little of one vector says little about the other.

Inputs: the ``RESULT`` lines under ``<run>/results`` (for phi) and the
``<run>/vectors/<key>.npy`` eigenvectors ``eigen_spectrum.py --save-vector``
wrote (for state). Output: ``overlaps.json`` and a printed table.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.txt")):
        record = None
        for line in path.read_text().splitlines():
            if line.startswith("RESULT "):
                record = json.loads(line[len("RESULT ") :])
        if record is not None and record.get("error") is None:
            records[record["key"]] = record
    return records


def phi_of(record: dict) -> np.ndarray:
    return np.asarray(record["phi_z_real"], dtype=float) + 1j * np.asarray(
        record["phi_z_imag"], dtype=float
    )


def normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def state_overlap(va: np.ndarray, vb: np.ndarray) -> dict:
    """Overlap on the common Laguerre block; axis 1 of (s, Nl, Nm, ky, kx, z)."""
    nl = min(va.shape[1], vb.shape[1])
    if va.shape[0] != vb.shape[0] or va.shape[2:] != vb.shape[2:]:
        raise ValueError(f"incompatible shapes {va.shape} {vb.shape}")
    a = va[:, :nl]
    b = vb[:, :nl]
    return {
        "common_Nl": int(nl),
        "overlap": normalized_overlap(a.reshape(-1), b.reshape(-1)),
        "retained_a": float(np.linalg.norm(a) / np.linalg.norm(va)),
        "retained_b": float(np.linalg.norm(b) / np.linalg.norm(vb)),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--vectors", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

records = load_results(args.results)
vectors: dict[str, np.ndarray] = {}
for key in records:
    path = args.vectors / f"{key}.npy"
    if path.exists():
        vectors[key] = np.load(path)

pairs: list[dict] = []
by_nu: dict[float, list[str]] = {}
by_nl: dict[int, list[str]] = {}
for key, record in records.items():
    by_nu.setdefault(float(record["nu"]), []).append(key)
    by_nl.setdefault(int(record["Nl"]), []).append(key)


def add_pairs(keys: list[str], family: str) -> None:
    ordered = sorted(keys, key=lambda k: (records[k]["nu"], records[k]["Nl"]))
    for ka, kb in itertools.combinations(ordered, 2):
        entry = {
            "family": family,
            "a": ka,
            "b": kb,
            "nu_a": records[ka]["nu"],
            "nu_b": records[kb]["nu"],
            "Nl_a": records[ka]["Nl"],
            "Nl_b": records[kb]["Nl"],
            "phi_overlap": normalized_overlap(phi_of(records[ka]), phi_of(records[kb])),
            "gamma_a": records[ka]["gamma"],
            "gamma_b": records[kb]["gamma"],
            "omega_a": records[ka]["omega"],
            "omega_b": records[kb]["omega"],
        }
        if ka in vectors and kb in vectors:
            entry.update(
                {
                    f"state_{k}": v
                    for k, v in state_overlap(vectors[ka], vectors[kb]).items()
                }
            )
        pairs.append(entry)


for nu, keys in sorted(by_nu.items()):
    add_pairs(keys, f"nl-ladder nu={nu:g}")
for nl, keys in sorted(by_nl.items()):
    add_pairs(keys, f"nu-family Nl={nl}")

payload = {
    "pairs": pairs,
    "keys": sorted(records),
    "vectors_present": sorted(vectors),
}
args.out.write_text(json.dumps(payload, indent=1) + "\n")

header = (
    "| family | a | b | phi overlap | state overlap | common Nl | "
    "retained a | retained b | gamma a | gamma b |"
)
print(header)
print("|" + "---|" * 10)
for entry in pairs:

    def fmt(value, spec="?.6f"):
        return "—" if value is None else format(value, spec.lstrip("?"))

    print(
        f"| {entry['family']} | {entry['a']} | {entry['b']} | "
        f"{fmt(entry['phi_overlap'])} | {fmt(entry.get('state_overlap'))} | "
        f"{entry.get('state_common_Nl', '—')} | "
        f"{fmt(entry.get('state_retained_a'))} | {fmt(entry.get('state_retained_b'))} | "
        f"{fmt(entry['gamma_a'])} | {fmt(entry['gamma_b'])} |"
    )
