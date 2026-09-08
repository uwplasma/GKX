"""The evidence ledger is the single index of every public number.

``tools/evidence_ledger.toml`` holds one row per published claim: the artifact
it is computed from, the generator that rebuilds that artifact, the reference it
is compared against, and that reference's rank. These tests keep the ledger
honest in the three ways it can rot.

1. A row can name an artifact or generator that no longer exists.
2. A row can drift from its own schema, so the ranks and tiers stop meaning
   anything.
3. A README number can be edited away from the artifact it claims to summarise,
   or an artifact can be refreshed without updating the README.

The third check is the mechanism introduced for the parity table in #193,
generalised so that adding a published number means adding a ledger row rather
than editing a hard-coded map in this file.
"""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import pytest

RUN_TO_REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = RUN_TO_REPO_ROOT / "tools" / "evidence_ledger.toml"

VALID_STATUS = {"open", "passing", "failing", "provisional", "superseded"}
VALID_RANKS = {1, 2, 3, 4, 5}
VALID_TIERS = {0, 1, 2, 3}
REQUIRED_KEYS = (
    "id",
    "claim",
    "observable",
    "artifact",
    "generator",
    "reference",
    "reference_rank",
    "tier",
    "status",
)


def _ledger() -> dict:
    assert LEDGER_PATH.is_file(), f"{LEDGER_PATH} is missing"
    return tomllib.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _rows() -> list[dict]:
    return _ledger()["row"]


def _readme_parity_rows() -> list[dict]:
    """Ledger rows that publish a percentage in the README parity table."""

    return [row for row in _rows() if "readme_label" in row]


def test_ledger_parses_and_declares_its_schema() -> None:
    ledger = _ledger()
    assert ledger.get("schema_version") == 1
    assert ledger.get("updated"), "the ledger records when it was last touched"
    assert _rows(), "the ledger is empty"


def test_every_row_is_well_formed() -> None:
    seen: set[str] = set()
    problems: list[str] = []
    for row in _rows():
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            problems.append(f"{row.get('id', '<no id>')}: missing {missing}")
            continue
        row_id = row["id"]
        if row_id in seen:
            problems.append(f"{row_id}: duplicate id")
        seen.add(row_id)
        if row["reference_rank"] not in VALID_RANKS:
            problems.append(f"{row_id}: reference_rank {row['reference_rank']!r}")
        if row["tier"] not in VALID_TIERS:
            problems.append(f"{row_id}: tier {row['tier']!r}")
        if row["status"] not in VALID_STATUS:
            problems.append(f"{row_id}: status {row['status']!r}")
    assert not problems, "malformed ledger rows:\n" + "\n".join(problems)


def test_every_artifact_and_generator_exists() -> None:
    """A row may not point at a file that has been moved or deleted."""

    problems: list[str] = []
    for row in _rows():
        for key in ("artifact", "companion"):
            target = row.get(key)
            if target and not (RUN_TO_REPO_ROOT / target).exists():
                problems.append(f"{row['id']}: {key} {target} is missing")
        # A generator may carry arguments; only the script path is checked.
        script = row["generator"].split()[0]
        if not (RUN_TO_REPO_ROOT / script).exists():
            problems.append(f"{row['id']}: generator {script} is missing")
    assert not problems, "ledger rows point at missing files:\n" + "\n".join(problems)


def test_provisional_rows_say_why() -> None:
    """``provisional`` is a claim about evidence, so it needs a stated reason."""

    missing = [
        row["id"]
        for row in _rows()
        if row["status"] == "provisional" and not row.get("notes")
    ]
    assert not missing, (
        "provisional rows must record what is unresolved and what would close "
        f"them: {missing}"
    )


def test_linked_boundary_references_record_their_clamp_state() -> None:
    """GX linked-boundary references must declare whether the clamp binds.

    GX launches ``dampEnds_linked`` with ``dG_all.z = min(65535, Nz*Nl*Nm)`` but,
    unlike its sibling kernels, that kernel has no grid-stride loop over z. A
    reference produced above the clamp has undamped high Hermite moments, so it
    cannot be treated as a settled comparison. Any row that records a binding
    clamp must be provisional and must explain itself.
    """

    problems: list[str] = []
    for row in _rows():
        if "reference_index_count" not in row:
            continue
        count = row["reference_index_count"]
        binds = row.get("reference_clamp_binds")
        assert binds is not None, f"{row['id']}: reference_clamp_binds is unset"
        if binds != (count > 65535):
            problems.append(
                f"{row['id']}: reference_clamp_binds={binds} contradicts "
                f"reference_index_count={count} against the 65535 clamp"
            )
        if binds and row["status"] == "passing":
            problems.append(
                f"{row['id']}: reference clamp binds, so the row cannot be "
                "'passing' until the reference is regenerated from a repaired build"
            )
    assert not problems, "\n".join(problems)


def test_shipped_reference_rows_carry_a_hash() -> None:
    """Rank-3 rows cite a specific file, so they pin it by content."""

    missing = [
        row["id"]
        for row in _rows()
        if row["reference_rank"] == 3 and not row.get("reference_sha256")
    ]
    assert not missing, (
        f"rank-3 rows cite a reference-code golden and must pin its SHA-256: {missing}"
    )


def _peak_relative_percent(rows: list[dict], ref_key: str, gkx_key: str) -> float:
    ref = [float(row[ref_key]) for row in rows]
    gkx = [float(row[gkx_key]) for row in rows]
    peak = max(abs(value) for value in ref)
    worst = max(abs(a - b) for a, b in zip(ref, gkx))
    return 100.0 * worst / peak


def _published_readme_parity() -> dict[str, tuple[float, float]]:
    text = (RUN_TO_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    labels = {row["readme_label"] for row in _readme_parity_rows()}
    published: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().strip("*` ") for cell in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        label, gamma_cell, omega_cell = cells
        if label not in labels:
            continue
        try:
            published[label] = (
                float(gamma_cell.rstrip("%")),
                float(omega_cell.rstrip("%")),
            )
        except ValueError:
            continue
    return published


def test_ledger_covers_every_published_parity_row() -> None:
    """The README table and the ledger describe the same set of claims."""

    published = set(_published_readme_parity())
    ledgered = {row["readme_label"] for row in _readme_parity_rows()}
    assert published == ledgered, (
        "the README parity table and the ledger disagree about which cases are "
        f"published. README only: {sorted(published - ledgered)}; "
        f"ledger only: {sorted(ledgered - published)}"
    )


@pytest.mark.parametrize(
    "row", _readme_parity_rows(), ids=lambda row: str(row["readme_label"])
)
def test_published_parity_recomputes_from_its_artifact(row: dict) -> None:
    """Every percentage the README publishes is recomputed from its artifact."""

    published = _published_readme_parity()
    label = row["readme_label"]
    assert label in published, f"{label} is not published in the README table"

    source = RUN_TO_REPO_ROOT / row["artifact"]
    csv_rows = list(csv.DictReader(source.open(encoding="utf-8")))
    assert csv_rows, f"{label}: {row['artifact']} has no rows"

    columns = row["columns"]
    gamma_pct, omega_pct = published[label]
    for value, ref_key, gkx_key, quantity in (
        (gamma_pct, columns["gamma_ref"], columns["gamma_gkx"], "gamma"),
        (omega_pct, columns["omega_ref"], columns["omega_gkx"], "omega"),
    ):
        measured = _peak_relative_percent(csv_rows, ref_key, gkx_key)
        # The tolerance is the rounding the table itself shows, not a physics
        # tolerance: three significant figures, with an absolute floor.
        tolerance = max(0.02 * value, 0.005)
        assert abs(measured - value) <= tolerance, (
            f"README publishes {quantity} = {value}% for {label}, but "
            f"{row['artifact']} gives {measured:.3f}%. Either the artifact was "
            "refreshed without updating the README, or the README was edited "
            "away from its artifact."
        )
