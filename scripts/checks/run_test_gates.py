#!/usr/bin/env python3
"""Run bounded release test gates from one maintained entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import cast


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DIR = REPO_ROOT / "tests"
COVERAGE_DATA_RE = re.compile(r"^\.coverage\.shard-(?P<shard>[0-9]+)\.")
#: Seconds per file under coverage on a hosted ubuntu runner (pytest ``--durations``).
#: The planner packs files longest-first on these; an unlisted file costs
#: ``DEFAULT_TEST_SECONDS``. Re-measure when a shard drifts far from the rest.
WIDE_COVERAGE_SECONDS: dict[str, float] = {
    "test_parallel_linear_velocity.py": 1040,
    "test_nonlinear.py": 925,
    "test_nonlinear_helpers_extra.py": 595,
    "test_linear.py": 400,
    "test_runtime_runner.py": 200,
    "test_autodiff_solver_objectives.py": 200,
    "test_examples.py": 150,
    "test_linear_krylov_core.py": 100,
}
DEFAULT_TEST_SECONDS = 15.0

WIDE_COVERAGE_LOGICAL_CPU_DEVICES = {
    # Exercise the real species/Hermite collectives and serial-identity gates.
    # The flag must be present before JAX is imported by the pytest subprocess.
    "test_parallel_linear_velocity.py": 4,
    # Routes [parallel] into the nonlinear path and gates the sharded answer
    # against the serial one; without real devices every case skips.
    "test_parallel_nonlinear_routing.py": 4,
    # Batch-map, runner and sharding-profile routes that take a multi-device
    # branch when more than one device is visible.
    "test_parallel_core.py": 4,
    "test_runners_and_orchestration.py": 4,
    "test_nonlinear_sharding_profile_contracts.py": 4,
}

#: Files split into this many contiguous node-ID chunks, each scheduled as its
#: own unit, so no single file sets the wall time of the whole matrix. Do not
#: add a logical-CPU owner here: those files share one JAX compilation cache
#: across their tests, and separate processes re-pay the shard-map compiles.
WIDE_COVERAGE_NODE_CHUNKS: dict[str, int] = {"test_nonlinear.py": 2}

#: One schedulable unit: (test file, chunk index, chunk count).
Unit = tuple[Path, int, int]


def _resolve_test_dir(test_dir: Path) -> Path:
    """Resolve relative test directories against the repository root."""

    return (
        (REPO_ROOT / test_dir).resolve()
        if not test_dir.is_absolute()
        else test_dir.resolve()
    )


def discover_test_files(test_dir: Path = DEFAULT_TEST_DIR) -> list[Path]:
    """Return pytest files below ``test_dir`` in deterministic order."""

    root = _resolve_test_dir(test_dir)
    return sorted(path for path in root.rglob("test_*.py") if path.is_file())


def _add_fast_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "files", nargs="*", type=Path, help="Optional explicit test files."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory tree containing test_*.py files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-file pytest timeout in seconds.",
    )
    parser.add_argument(
        "--total-timeout",
        type=float,
        default=300.0,
        help="Total runner budget in seconds; use 0 to disable the whole-run cap.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to each pytest invocation; repeat as needed.",
    )


def parse_fast_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest files with bounded per-file and total budgets."
    )
    _add_fast_arguments(parser)
    return parser.parse_args(argv)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def run_tests(
    files: list[Path],
    *,
    per_file_timeout_s: float,
    total_timeout_s: float,
    pytest_args: list[str] | None = None,
) -> tuple[int, list[tuple[str, str, float]]]:
    """Run each pytest file and return ``(exit_code, results)``.

    Exit code ``124`` means at least one subprocess hit a timeout or the
    configured total budget expired before all files were attempted.
    """

    pytest_args = list(pytest_args or [])
    deadline = time.monotonic() + total_timeout_s if total_timeout_s > 0.0 else None
    results: list[tuple[str, str, float]] = []
    timed_out = False
    failed = False

    for path in files:
        label = _relative(path)
        if deadline is not None and time.monotonic() >= deadline:
            results.append((label, "not_run(total_timeout)", 0.0))
            timed_out = True
            continue

        timeout_s = float(per_file_timeout_s)
        if deadline is not None:
            timeout_s = max(1.0, min(timeout_s, deadline - time.monotonic()))

        t0 = time.monotonic()
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--maxfail=1",
            "--disable-warnings",
            *pytest_args,
            str(path),
        ]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=timeout_s)
            status = "ok"
        except subprocess.TimeoutExpired:
            status = "timeout"
            timed_out = True
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 5:
                status = "skipped(no_tests_collected)"
            else:
                status = f"fail({exc.returncode})"
                failed = True
        dt = time.monotonic() - t0
        results.append((label, status, dt))
        print(f"{label}: {status} ({dt:.1f}s)", flush=True)

    print("SUMMARY", flush=True)
    for path, status, dt in results:
        print(f"{path}\t{status}\t{dt:.1f}s", flush=True)

    if timed_out:
        return 124, results
    if failed:
        return 1, results
    return 0, results


def main_fast(argv: list[str] | None = None) -> int:
    args = parse_fast_args(argv)
    files = [path if path.is_absolute() else (REPO_ROOT / path) for path in args.files]
    if not files:
        files = discover_test_files(args.test_dir)
    if not files:
        raise SystemExit(f"no test_*.py files found under {args.test_dir}")
    code, _results = run_tests(
        files,
        per_file_timeout_s=float(args.timeout),
        total_timeout_s=float(args.total_timeout),
        pytest_args=list(args.pytest_arg),
    )
    return code


def plan_units(files: list[Path]) -> list[Unit]:
    """Expand files into schedulable units, splitting the chunked owners."""

    return [
        (path, idx, count)
        for path in files
        for count in [WIDE_COVERAGE_NODE_CHUNKS.get(path.name, 1)]
        for idx in range(count)
    ]


def unit_seconds(unit: Unit) -> float:
    """Return the measured cost of one unit."""

    path, _, count = unit
    return WIDE_COVERAGE_SECONDS.get(path.name, DEFAULT_TEST_SECONDS) / count


def split_shards(items: list[Path], nshards: int) -> list[list[Unit]]:
    """Split test files into deterministic shards balanced on measured cost.

    Longest-processing-time-first: each unit, most expensive first, goes to the
    currently lightest shard. With equal costs this reduces to round-robin.
    """

    if nshards < 1:
        raise ValueError("nshards must be >= 1")
    units = plan_units(items)
    shards: list[list[Unit]] = [[] for _ in range(nshards)]
    loads = [0.0] * nshards
    for order, unit in sorted(
        enumerate(units), key=lambda pair: (-unit_seconds(pair[1]), pair[0])
    ):
        shard_idx = min(range(nshards), key=lambda idx: (loads[idx], idx))
        shards[shard_idx].append(unit)
        loads[shard_idx] += unit_seconds(unit)
    for shard in shards:
        shard.sort(key=units.index)
    return shards


def split_contiguous(items: list[str], nchunks: int) -> list[list[str]]:
    """Split ordered items into non-empty contiguous chunks."""

    if nchunks < 1:
        raise ValueError("nchunks must be >= 1")
    if not items:
        return []
    count = min(int(nchunks), len(items))
    base, remainder = divmod(len(items), count)
    chunks: list[list[str]] = []
    start = 0
    for idx in range(count):
        stop = start + base + (1 if idx < remainder else 0)
        chunks.append(items[start:stop])
        start = stop
    return chunks


def collect_pytest_nodeids(path: Path, pytest_args: list[str]) -> list[str]:
    """Collect selected pytest node IDs for one file in deterministic order."""

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        *pytest_args,
        str(path.relative_to(REPO_ROOT)),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    nodeids = [line.strip() for line in result.stdout.splitlines() if "::" in line]
    if not nodeids:
        raise SystemExit(f"no pytest node IDs collected from {_relative(path)}")
    return nodeids


def wide_coverage_environment(shard: list[Unit]) -> dict[str, str] | None:
    """Return the pre-import environment required by a coverage shard."""

    logical_cpu_devices = max(
        (WIDE_COVERAGE_LOGICAL_CPU_DEVICES.get(path.name, 0) for path, _, _ in shard),
        default=0,
    )
    if not logical_cpu_devices:
        return None
    env = os.environ.copy()
    flag = f"--xla_force_host_platform_device_count={logical_cpu_devices}"
    current_flags = env.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in current_flags:
        env["XLA_FLAGS"] = f"{current_flags} {flag}".strip()
    env["JAX_PLATFORMS"] = "cpu"
    return env


def wide_coverage_shard_batches(
    shard: list[Unit], *, pytest_args: list[str]
) -> list[list[str]]:
    """Return the pytest targets for one coverage shard, one entry per command.

    Whole files share one command; each chunk of a split owner is its own
    command over its contiguous slice of the collected node IDs, so the chunks
    of a file, wherever they land, run every one of its tests exactly once.
    """

    whole = [_relative(path) for path, _, count in shard if count == 1]
    batches = [whole] if whole else []
    for path, idx, count in shard:
        if count > 1:
            chunks = split_contiguous(collect_pytest_nodeids(path, pytest_args), count)
            if idx < len(chunks):
                batches.append(chunks[idx])
    return batches


def xdist_args(batch: list[str], workers: int) -> list[str]:
    """Return pytest-xdist arguments for one command of a coverage shard.

    Several files are distributed file by file, so tests that share a file
    keep sharing its in-process JAX compilations; a lone file or a node-ID
    chunk of one is distributed test by test, since file grouping would
    serialize it.
    """

    if workers < 2:
        return []
    files = {target.split("::")[0] for target in batch}
    dist = "load" if len(files) == 1 else "loadfile"
    return ["-n", str(workers), "--dist", dist]


def discover_coverage_data(root: Path = REPO_ROOT) -> list[Path]:
    """Return coverage.py data files under ``root`` without descending into docs."""

    return sorted(path for path in root.glob(".coverage.*") if path.is_file())


def discover_empty_shard_markers(root: Path = REPO_ROOT) -> list[Path]:
    """Return sentinel files written when a CI shard produced no coverage data."""

    return sorted(path for path in root.glob("EMPTY_SHARD_*") if path.is_file())


def build_coverage_shard_report(root: Path, nshards: int) -> dict[str, object]:
    """Return a JSON-ready report for combine-time shard artifact validation."""

    coverage_files = discover_coverage_data(root)
    markers = discover_empty_shard_markers(root)
    labeled: dict[int, list[Path]] = {idx: [] for idx in range(1, nshards + 1)}
    unlabeled: list[Path] = []
    out_of_range: list[Path] = []
    for path in coverage_files:
        match = COVERAGE_DATA_RE.match(path.name)
        if not match:
            unlabeled.append(path)
            continue
        shard = int(match.group("shard"))
        if 1 <= shard <= nshards:
            labeled[shard].append(path)
        else:
            out_of_range.append(path)

    def _rel(paths: list[Path]) -> list[str]:
        return [str(path.relative_to(root)) for path in paths]

    return {
        "kind": "wide_coverage_shard_report",
        "root": str(root),
        "expected_shards": int(nshards),
        "coverage_data_files": _rel(coverage_files),
        "coverage_data_file_count": len(coverage_files),
        "empty_shard_markers": _rel(markers),
        "unlabeled_coverage_data_files": _rel(unlabeled),
        "out_of_range_labeled_coverage_data_files": _rel(out_of_range),
        "labeled_shards": {
            str(idx): _rel(paths) for idx, paths in labeled.items() if paths
        },
        "missing_labeled_shards": [idx for idx, paths in labeled.items() if not paths],
    }


def validate_coverage_shard_report(
    report: dict[str, object], *, require_labeled_shards: bool
) -> list[str]:
    """Return validation failures for a combine-time coverage shard report."""

    failures: list[str] = []
    if int(cast(int, report["coverage_data_file_count"])) == 0:
        failures.append("no coverage.py data files were found")
    markers = list(cast(list[str], report["empty_shard_markers"]))
    if markers:
        failures.append(f"empty shard markers found: {markers}")
    out_of_range = list(
        cast(list[str], report["out_of_range_labeled_coverage_data_files"])
    )
    if out_of_range:
        failures.append(
            f"out-of-range labeled coverage data files found: {out_of_range}"
        )
    missing = list(cast(list[int], report["missing_labeled_shards"]))
    if require_labeled_shards and missing:
        failures.append(f"missing labeled coverage data for shards: {missing}")
    return failures


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write deterministic JSON for CI artifacts and local diagnostics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run(
    cmd: list[str],
    *,
    timeout: int | None,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=cwd, timeout=timeout, check=True, env=env)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"command timed out after {timeout}s: {' '.join(cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


def _add_wide_coverage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shards", type=int, default=6, help="Number of bounded test shards."
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-shard timeout in seconds."
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=95.0,
        help="Combined package coverage threshold.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("coverage-wide.xml"),
        help="Combined XML report path.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory tree containing test_*.py files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print shard membership without running tests.",
    )
    parser.add_argument(
        "--only-shard",
        type=int,
        default=None,
        help="Run only one 1-based shard. Useful for local bounded shard execution.",
    )
    parser.add_argument(
        "--keep-existing-coverage",
        action="store_true",
        help="Do not erase existing .coverage data before running selected shard(s).",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        help="Run selected shard(s) without combining/reporting coverage data.",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip pytest execution and only combine/report existing shard coverage data.",
    )
    parser.add_argument(
        "--require-shard-data",
        action="store_true",
        help=(
            "Before combining, require one or more labeled .coverage.shard-N.* "
            "files for every configured shard and reject EMPTY_SHARD_N markers."
        ),
    )
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=None,
        help="Optional JSON report path for combine-time coverage shard data.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed to each pytest shard; repeat as needed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="pytest-xdist workers per shard command; 0 or 1 runs in-process.",
    )


def parse_wide_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run package-wide coverage in bounded shards."
    )
    _add_wide_coverage_arguments(parser)
    return parser.parse_args(argv)


def _combine(args: argparse.Namespace) -> int:
    """Combine shard coverage data, write the XML and enforce the floor."""

    coverage = [sys.executable, "-m", "coverage"]
    _run([*coverage, "combine"], timeout=120, cwd=REPO_ROOT)
    _run([*coverage, "xml", "-o", str(args.xml)], timeout=120, cwd=REPO_ROOT)
    floor = f"--fail-under={float(args.fail_under):.6g}"
    _run([*coverage, "report", floor], timeout=120, cwd=REPO_ROOT)
    return 0


def main_wide(argv: list[str] | None = None) -> int:
    args = parse_wide_args(argv)
    test_dir = _resolve_test_dir(args.test_dir)
    tests = discover_test_files(test_dir)
    if not tests:
        raise SystemExit(f"no test_*.py files found under {test_dir}")
    shards = split_shards(tests, int(args.shards))

    for idx, shard in enumerate(shards):
        load = sum(unit_seconds(unit) for unit in shard)
        print(f"shard {idx + 1}/{len(shards)}: {len(shard)} units, ~{load:.0f} s")
        for path, chunk, count in shard:
            part = f" [chunk {chunk + 1}/{count}]" if count > 1 else ""
            print(f"  {_relative(path)}{part}")

    if args.dry_run:
        return 0

    if args.combine_only:
        report = build_coverage_shard_report(REPO_ROOT, int(args.shards))
        if args.shard_manifest is not None:
            write_json(args.shard_manifest, report)
        failures = validate_coverage_shard_report(
            report, require_labeled_shards=bool(args.require_shard_data)
        )
        if failures:
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            raise SystemExit(
                "wide coverage shard validation failed: " + "; ".join(failures)
            )
        return _combine(args)

    if args.only_shard is not None and not (1 <= int(args.only_shard) <= len(shards)):
        raise SystemExit(f"--only-shard must be in [1, {len(shards)}]")
    selected = (
        [(int(args.only_shard) - 1, shards[int(args.only_shard) - 1])]
        if args.only_shard is not None
        else list(enumerate(shards))
    )

    if not args.keep_existing_coverage:
        _run([sys.executable, "-m", "coverage", "erase"], timeout=None, cwd=REPO_ROOT)
    for idx, shard in selected:
        if not shard:
            continue
        flags = "-q --maxfail=1 --disable-warnings --cov=gkx --cov-report="
        pytest_cmd = [sys.executable, "-m", "pytest", *flags.split(), *args.pytest_arg]
        shard_env = wide_coverage_environment(shard) or os.environ.copy()
        batches = wide_coverage_shard_batches(shard, pytest_args=list(args.pytest_arg))
        for batch_idx, batch in enumerate(batches, start=1):
            label = f"running coverage shard {idx + 1}/{len(shards)}"
            if len(batches) > 1:
                label += f" batch {batch_idx}/{len(batches)} ({len(batch)} targets)"
            print(label, flush=True)
            # pytest-cov, unlike ``coverage run``, also traces xdist workers
            # and combines them into this one parallel-mode data file.
            shard_env["COVERAGE_FILE"] = str(
                REPO_ROOT / f".coverage.run-{idx + 1}-{batch_idx}"
            )
            _run(
                [*pytest_cmd, *xdist_args(batch, int(args.workers)), *batch],
                timeout=int(args.timeout),
                cwd=REPO_ROOT,
                env=shard_env,
            )

    return 0 if args.skip_combine else _combine(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fast = subparsers.add_parser(
        "fast", help="Run bounded per-file pytest invocations."
    )
    _add_fast_arguments(fast)
    wide = subparsers.add_parser(
        "wide-coverage", help="Run package-wide coverage in bounded shards."
    )
    _add_wide_coverage_arguments(wide)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "fast":
        return main_fast(argv[1:] if argv is not None else sys.argv[2:])
    if args.command == "wide-coverage":
        return main_wide(argv[1:] if argv is not None else sys.argv[2:])
    raise SystemExit(f"unknown test-gate command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
