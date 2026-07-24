"""Packaging correctness tests.

Builds the actual wheel with `uv build` and inspects its contents. This is
the only reliable way to catch a packaging regression (e.g. a hatchling
config change that silently drops the prompt `.md` files or the console
script entry point) — unit tests importing `argus` from the source tree
would never notice, since they never install from the built artifact.

Marked `packaging` (see pyproject.toml `[tool.pytest.ini_options] markers`)
so it's easy to deselect with `-m "not packaging"` if `uv build` isn't
available in a given environment, but it is NOT excluded from the default
test run — CI's `pytest` job runs it same as any other test.
"""

from __future__ import annotations

import importlib.resources
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def built_wheel_path() -> Iterator[Path]:
    """Build the wheel into a scratch directory and return its path.

    Uses a temp `--out-dir` (rather than the repo's `dist/`) so the test
    never leaves build artifacts behind or races a developer's own
    `uv build` invocation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = list(out_dir.glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
        # Copy out of the tempdir before it's cleaned up.
        persisted = Path(tempfile.mkdtemp()) / wheels[0].name
        persisted.write_bytes(wheels[0].read_bytes())
        yield persisted
        persisted.unlink(missing_ok=True)
        persisted.parent.rmdir()


@pytest.mark.packaging
def test_wheel_builds_successfully(built_wheel_path: Path) -> None:
    assert built_wheel_path.exists()
    assert built_wheel_path.suffix == ".whl"


@pytest.mark.packaging
def test_wheel_contains_all_prompt_files(built_wheel_path: Path) -> None:
    expected = {f"argus/prompts/{p.name}" for p in (REPO_ROOT / "argus" / "prompts").glob("*.md")}
    assert expected, "no source prompt .md files found — test fixture is broken"

    with zipfile.ZipFile(built_wheel_path) as z:
        packaged = {n for n in z.namelist() if n.startswith("argus/prompts/") and n.endswith(".md")}

    assert packaged == expected


@pytest.mark.packaging
def test_wheel_does_not_contain_schema_sql(built_wheel_path: Path) -> None:
    """schema/*.sql ships in the repo/sdist only, not the installable wheel.

    See docs/RELEASING.md for the rationale: it's Postgres DDL for
    self-hosters running the postgres storage backend, not a runtime
    resource `argus` imports.
    """
    with zipfile.ZipFile(built_wheel_path) as z:
        sql_entries = [n for n in z.namelist() if n.endswith(".sql")]
    assert sql_entries == []


@pytest.mark.packaging
def test_sdist_contains_schema_sql_and_changelog() -> None:
    """The sdist (unlike the wheel) does ship schema/*.sql for self-hosters,
    and CHANGELOG.md (linked from [project.urls] but outside the `argus`
    package, so the wheel never gets it)."""
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        sdists = list(out_dir.glob("*.tar.gz"))
        assert len(sdists) == 1
        with tarfile.open(sdists[0]) as t:
            sql_entries = [n for n in t.getnames() if n.endswith(".sql")]
            changelog_entries = [n for n in t.getnames() if n.endswith("CHANGELOG.md")]
        assert sql_entries, "expected schema/*.sql inside the sdist"
        assert changelog_entries, "expected CHANGELOG.md inside the sdist"


@pytest.mark.packaging
def test_wheel_has_console_script_entry_point(built_wheel_path: Path) -> None:
    with zipfile.ZipFile(built_wheel_path) as z:
        entry_points_name = next(n for n in z.namelist() if n.endswith("entry_points.txt"))
        content = z.read(entry_points_name).decode()

    assert "[console_scripts]" in content
    assert "argus = argus.cli:main" in content


@pytest.mark.packaging
def test_installed_package_prompts_enumerable_via_importlib_resources() -> None:
    """Sanity check the *installed source* (not the wheel) is import-resource
    friendly: `importlib.resources` can enumerate `argus/prompts/*.md`. This
    is the same access pattern `argus.prompts_runtime` uses at runtime, and
    it must keep working once the package is pip/uv-installed rather than
    run from a source checkout.
    """
    prompts_pkg = importlib.resources.files("argus.prompts")
    names = {p.name for p in prompts_pkg.iterdir() if p.name.endswith(".md")}
    on_disk = {p.name for p in (REPO_ROOT / "argus" / "prompts").glob("*.md")}
    assert names == on_disk
    assert len(names) >= 20


@pytest.mark.packaging
@pytest.mark.skipif(sys.version_info < (3, 12), reason="repo targets py312+")
def test_wheel_version_is_pep440_compliant(built_wheel_path: Path) -> None:
    """The dynamic (hatch-vcs derived) version must be a well-formed PEP 440
    version string, e.g. `0.1.0` (tagged release) or `0.1.dev5+gabc1234`
    (untagged dev build) — not the old hardcoded `0.0.1.dev0` literal.
    """
    with zipfile.ZipFile(built_wheel_path) as z:
        metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = z.read(metadata_name).decode()

    version_line = next(line for line in metadata.splitlines() if line.startswith("Version: "))
    version = version_line.removeprefix("Version: ").strip()

    from packaging.version import Version

    Version(version)  # raises InvalidVersion if malformed


@pytest.mark.packaging
def test_wheel_metadata_lists_changelog_url(built_wheel_path: Path) -> None:
    """Guards against the curated Changelog project URL silently disappearing
    from [project.urls] in a future pyproject.toml edit."""
    with zipfile.ZipFile(built_wheel_path) as z:
        metadata_name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = z.read(metadata_name).decode()

    assert "Project-URL: Changelog," in metadata
