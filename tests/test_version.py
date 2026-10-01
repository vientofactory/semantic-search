"""Version manifest guard for the semantic-search submodule.

`pyproject.toml [project].version` is the single source of truth for this
submodule's version, and releases are tagged `semantic-search-vX.Y.Z` on the
squash commit of `main` (AGENTS.md: Version bump before merge / Step 1 -> Step 4).

The checks keep that rule mechanically enforceable in the sidecar CI job, and
pin the "metadata only" contract so the dependency list cannot drift away from
`requirements.txt` / `requirements.lock`.
"""

import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = PROJECT_ROOT / 'pyproject.toml'
# `X.Y.Z`, optionally with a prerelease suffix — matches the `semantic-search-v*` tag shape.
SEMVER = re.compile(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$')


def load_project() -> dict:
    """Parse pyproject.toml and return its [project] table."""
    with PYPROJECT_PATH.open('rb') as handle:
        return tomllib.load(handle)['project']


def test_version_is_semver() -> None:
    version = load_project()['version']
    assert SEMVER.match(version), (
        f'pyproject.toml [project].version {version!r} is not X.Y.Z — bump it in the same PR '
        'that ships the change, then tag main as semantic-search-v<version>.'
    )


def test_required_metadata_present() -> None:
    project = load_project()
    for key in ('name', 'version', 'description', 'readme', 'requires-python', 'license'):
        assert project.get(key), f'pyproject.toml [project].{key} is missing'
    assert (PROJECT_ROOT / project['readme']).exists(), (
        f'pyproject.toml readme points at {project["readme"]!r}, which does not exist'
    )


def test_manifest_stays_metadata_only() -> None:
    """Dependencies are owned by requirements.txt / requirements.lock.

    A duplicate dependency list (or a build backend) here would create a second
    source of truth and silently break the lockfile rule in AGENTS.md.
    """
    with PYPROJECT_PATH.open('rb') as handle:
        data = tomllib.load(handle)
    assert 'dependencies' not in data.get('project', {}), (
        'pyproject.toml must stay metadata-only: keep dependencies in requirements.txt'
    )
    assert 'build-system' not in data, (
        'pyproject.toml must stay metadata-only: this submodule is not installed as a package'
    )
