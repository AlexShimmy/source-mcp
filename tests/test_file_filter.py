import pytest
from pathlib import Path
from unittest.mock import patch

from src.services.file_filter import FileFilter, INDEXABLE_EXTENSIONS, SKIP_DIRS


@pytest.fixture
def project_root(tmp_path):
    """Create a realistic project structure."""
    # Source files (should be indexed)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "utils.ts").write_text("export const x = 1;")
    (tmp_path / "README.md").write_text("# Project")
    (tmp_path / "config.yaml").write_text("key: value")

    # Should be skipped — node_modules
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = {};")

    # Should be skipped — .expo
    (tmp_path / ".expo").mkdir()
    (tmp_path / ".expo" / "xcodebuild.log").write_text("x" * 10000)

    # Should be skipped — binary/media
    (tmp_path / "photo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)
    (tmp_path / "bundle.min.js").write_text("var a=1;")

    # Should be skipped — too large (> 1MB)
    (tmp_path / "huge.py").write_text("x" * (2 * 1024 * 1024))

    # Should be skipped — lock file
    (tmp_path / "package-lock.json").write_text("{}")

    # .gitignore
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")

    # Build dir (gitignored)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "output.js").write_text("compiled")

    return tmp_path


@pytest.fixture
def file_filter(project_root):
    return FileFilter(project_root)


def test_indexes_source_files(file_filter, project_root):
    assert file_filter.should_index(project_root / "src" / "main.py") == ""
    assert file_filter.should_index(project_root / "src" / "utils.ts") == ""
    assert file_filter.should_index(project_root / "README.md") == ""
    assert file_filter.should_index(project_root / "config.yaml") == ""


def test_skips_node_modules(file_filter, project_root):
    reason = file_filter.should_index(project_root / "node_modules" / "pkg" / "index.js")
    assert "skip-dir" in reason


def test_skips_expo(file_filter, project_root):
    reason = file_filter.should_index(project_root / ".expo" / "xcodebuild.log")
    assert reason != ""  # skipped for some reason (skip-dir or skip-suffix)


def test_skips_binary(file_filter, project_root):
    reason = file_filter.should_index(project_root / "photo.png")
    assert "skip-suffix" in reason


def test_skips_minified(file_filter, project_root):
    reason = file_filter.should_index(project_root / "bundle.min.js")
    assert "skip-suffix" in reason


def test_skips_too_large(file_filter, project_root):
    reason = file_filter.should_index(project_root / "huge.py")
    assert "too-large" in reason


def test_skips_lock_file(file_filter, project_root):
    reason = file_filter.should_index(project_root / "package-lock.json")
    assert "skip-name" in reason


def test_skips_gitignored(file_filter, project_root):
    reason = file_filter.should_index(project_root / "build" / "output.js")
    # Could be skip-dir:build or gitignored — either is correct
    assert reason != ""


def test_collect_files(file_filter):
    indexable, skipped = file_filter.collect_files()
    # Should find our 4 source files
    names = {p.name for p in indexable}
    assert "main.py" in names
    assert "utils.ts" in names
    assert "README.md" in names
    # Should NOT include skipped files
    assert "xcodebuild.log" not in names
    assert "photo.png" not in names
    assert "bundle.min.js" not in names
    assert "huge.py" not in names
    # Skipped count should be > 0
    assert skipped > 0


def test_ragmcpignore(tmp_path):
    """Test .rag-mcpignore support."""
    (tmp_path / "keep.py").write_text("keep")
    (tmp_path / "secret.py").write_text("secret")
    (tmp_path / ".rag-mcpignore").write_text("secret.py\n")

    ff = FileFilter(tmp_path)
    assert ff.should_index(tmp_path / "keep.py") == ""
    assert "rag-mcpignored" in ff.should_index(tmp_path / "secret.py")
