"""Check source, packaging, coverage and version contracts before release."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$")


class ReleaseVersionError(ValueError):
    """Release metadata is inconsistent."""


class ReleaseReadinessError(RuntimeError):
    """A required source or packaging contract is missing."""


def read_project_version(root: Path = REPO_ROOT) -> str:
    """Return the PEP 621 project version from ``pyproject.toml``."""

    path = root / "pyproject.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        version = payload["project"]["version"]
    except KeyError as exc:
        raise ReleaseVersionError("pyproject.toml is missing project.version") from exc
    if not isinstance(version, str) or not version.strip():
        raise ReleaseVersionError(
            "pyproject.toml project.version must be a non-empty string"
        )
    return version.strip()


def read_source_version(root: Path = REPO_ROOT) -> str:
    """Return ``gkx.__version__`` without importing the package."""

    path = root / "src" / "gkx" / "_version.py"
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VERSION_RE.match(line.strip())
        if match:
            return match.group(1)
    raise ReleaseVersionError(f"{path.relative_to(root)} does not define __version__")


def normalize_tag(tag: str | None) -> str | None:
    """Normalize GitHub tag strings such as ``refs/tags/v1.2.3``."""

    if tag is None:
        return None
    tag = tag.strip()
    if not tag:
        return None
    if tag.startswith("refs/tags/"):
        tag = tag.removeprefix("refs/tags/")
    return tag


def default_tag_from_github_env() -> str | None:
    """Return the GitHub ref name only for tag-triggered workflows."""

    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return None
    return os.environ.get("GITHUB_REF_NAME")


def fetch_pypi_versions(package: str) -> set[str]:
    """Read PyPI versions, treating a project-level HTTP 404 as unpublished."""

    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310 - fixed PyPI HTTPS endpoint
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise
    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise ReleaseVersionError(f"PyPI response for {package!r} is missing releases")
    return {str(version) for version in releases}


def validate_release_version(
    *,
    root: Path = REPO_ROOT,
    tag: str | None = None,
    require_tag: bool = False,
    package: str = "gkx",
    pypi_versions: Iterable[str] | None = None,
) -> dict[str, object]:
    """Validate package version, source version, optional tag, and PyPI uniqueness."""

    root = root.resolve()
    project_version = read_project_version(root)
    source_version = read_source_version(root)
    if source_version != project_version:
        raise ReleaseVersionError(
            f"src/gkx/_version.py has {source_version!r}, "
            f"but pyproject.toml has {project_version!r}"
        )

    normalized_tag = normalize_tag(tag)
    if require_tag and normalized_tag is None:
        raise ReleaseVersionError("release publishing requires a tag like v1.2.3")
    if normalized_tag is not None:
        expected = f"v{project_version}"
        if normalized_tag != expected:
            raise ReleaseVersionError(
                f"release tag {normalized_tag!r} does not match project version {project_version!r}; "
                f"expected {expected!r}"
            )

    duplicate_on_pypi = False
    if pypi_versions is not None:
        duplicate_on_pypi = project_version in {
            str(version) for version in pypi_versions
        }
        if duplicate_on_pypi:
            raise ReleaseVersionError(
                f"{package} {project_version} already exists on PyPI; bump the version before publishing"
            )

    return {
        "package": package,
        "project_version": project_version,
        "source_version": source_version,
        "tag": normalized_tag,
        "require_tag": require_tag,
        "checked_pypi": pypi_versions is not None,
        "duplicate_on_pypi": duplicate_on_pypi,
    }


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ReleaseReadinessError(f"missing required file: {path}") from exc


def _missing_snippets(text: str, snippets: tuple[str, ...]) -> list[str]:
    return [snippet for snippet in snippets if snippet not in text]


def _project_metadata(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    project = pyproject.get("project", {})
    scripts = project.get("scripts", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "scripts": sorted(scripts),
    }


def build_version_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate release-version consistency before publishing artifacts."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument(
        "--tag",
        default=default_tag_from_github_env(),
        help="Release tag to validate.",
    )
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="Fail unless --tag is a v-prefixed release tag.",
    )
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="Fail if this version already exists on PyPI.",
    )
    parser.add_argument(
        "--package",
        default="gkx",
        help="PyPI package name for duplicate checks.",
    )
    return parser


def main_version(argv: list[str] | None = None) -> int:
    args = build_version_parser().parse_args(argv)
    pypi_versions = fetch_pypi_versions(args.package) if args.check_pypi else None
    try:
        report = validate_release_version(
            root=args.root,
            tag=args.tag,
            require_tag=bool(args.require_tag),
            package=str(args.package),
            pypi_versions=pypi_versions,
        )
    except ReleaseVersionError as exc:
        raise SystemExit(f"release-version check failed: {exc}") from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


# Define required CI and package contracts.
REQUIRED_CI_SNIPPETS = (
    "wide-coverage-shards",
    "--require-shard-data",
    "--fail-under 95",
    "--enforce-package-coverage",
    "ci-required:",
    "python-floor:",
    "python scripts/check.py readiness",
    "rm -rf build dist",
)
REQUIRED_RELEASE_SNIPPETS = (
    "--require-tag",
    "--check-pypi",
    "gh-action-pypi-publish",
    "python scripts/check.py readiness",
    "twine check dist/*",
    "Smoke-test installed wheel",
    "rm -rf build dist",
)


def check_release_readiness(root: Path = REPO_ROOT) -> dict[str, Any]:
    root = root.resolve()
    version = validate_release_version(root=root)
    project = _project_metadata(root)
    failures = []
    if project["name"] != "gkx" or "gkx" not in project["scripts"]:
        failures.append("project must provide the gkx package and console script")
    for relative, required in (
        (".github/workflows/ci.yml", REQUIRED_CI_SNIPPETS),
        (".github/workflows/release.yml", REQUIRED_RELEASE_SNIPPETS),
        ("README.md", ("pip install gkx", "MIT")),
    ):
        missing = _missing_snippets(_read(root / relative), required)
        if missing:
            failures.append(f"{relative}: missing contracts {missing}")
    for relative in (
        "LICENSE",
        "PROVENANCE.md",
        "tools/validation_coverage_manifest.toml",
    ):
        if not (root / relative).is_file():
            failures.append(f"missing required source file: {relative}")
    report = {
        "kind": "gkx_release_readiness",
        "project": project,
        "version": version,
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise ReleaseReadinessError("; ".join(failures))
    return report


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "version":
        return main_version(tokens[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(tokens)
    payload = (
        json.dumps(check_release_readiness(args.root), indent=2, sort_keys=True) + "\n"
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
