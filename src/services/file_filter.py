"""Smart file filter — decide what gets indexed, Cursor-style."""

from pathlib import Path
from typing import Optional, Set

import pathspec

from .monitor import logger


# ── Extensions worth indexing ───────────────────────────────
# Code
_CODE_EXTS: Set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".swift",
    ".scala", ".lua", ".r", ".m", ".mm", ".pl", ".pm",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto",
}
# Config & data
_CONFIG_EXTS: Set[str] = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".properties", ".xml", ".plist",
}
# Docs
_DOC_EXTS: Set[str] = {
    ".md", ".mdx", ".rst", ".txt", ".tex", ".adoc", ".org",
    ".csv", ".tsv",
}
# Web
_WEB_EXTS: Set[str] = {
    ".html", ".htm", ".css", ".scss", ".sass", ".less", ".vue", ".svelte",
}
# Other
_OTHER_EXTS: Set[str] = {
    ".dockerfile", ".makefile", ".cmake",
    ".tf", ".hcl",  # terraform
    ".nix", ".dhall",
}

INDEXABLE_EXTENSIONS: Set[str] = _CODE_EXTS | _CONFIG_EXTS | _DOC_EXTS | _WEB_EXTS | _OTHER_EXTS
# Also index extensionless files like Makefile, Dockerfile, etc.
INDEXABLE_NAMES: Set[str] = {
    "Makefile", "Dockerfile", "Rakefile", "Gemfile", "Procfile",
    "Vagrantfile", "CMakeLists.txt", "LICENSE", "README",
    ".gitignore", ".gitattributes", ".editorconfig",
    ".rag-mcpignore", ".cursorignore", ".prettierrc", ".eslintrc",
}

# ── Always-skip directories ────────────────────────────────
SKIP_DIRS: Set[str] = {
    ".git", ".svn", ".hg",
    "node_modules", ".expo", ".next", ".nuxt",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", ".build", "out", "target",
    ".idea", ".vscode", ".cursor",
    "coverage", ".nyc_output",
    ".tox", ".nox",
    "vendor",
    "zvec_db", "just_to_test_zvec_db",
}

# ── Always-skip patterns ───────────────────────────────────
SKIP_SUFFIXES: Set[str] = {
    ".min.js", ".min.css", ".map",
    ".lock", ".log",
    ".pyc", ".pyo", ".class", ".o", ".so", ".dylib", ".dll", ".exe",
    ".wasm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".db", ".sqlite", ".sqlite3",
    ".DS_Store",
}

# Files that should always be skipped regardless of extension
SKIP_NAMES: Set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", "composer.lock",
    "Gemfile.lock", "Cargo.lock", "go.sum",
    "uv.lock",
}

MAX_FILE_SIZE: int = 1 * 1024 * 1024  # 1 MB


class FileFilter:
    """Decides which files should be indexed, Cursor-style."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        self._ragmcpignore_spec: Optional[pathspec.PathSpec] = None
        self._load_ignore_files()

    def _load_ignore_files(self):
        """Parse .gitignore and .rag-mcpignore files."""
        for name, attr in [(".gitignore", "_gitignore_spec"), (".rag-mcpignore", "_ragmcpignore_spec")]:
            ignore_path = self.root / name
            if ignore_path.is_file():
                try:
                    patterns = ignore_path.read_text(encoding="utf-8").splitlines()
                    spec = pathspec.PathSpec.from_lines("gitignore", patterns)
                    setattr(self, attr, spec)
                    logger.info(f"Loaded {name} ({len(patterns)} patterns)")
                except Exception as exc:
                    logger.warning(f"Failed to parse {name}: {exc}")

    def should_index(self, filepath: Path) -> str:
        """Returns empty string if file should be indexed, otherwise a skip reason."""
        filepath = filepath.resolve()

        # ── Directory-based skip ────────────────────────────
        for part in filepath.parts:
            if part in SKIP_DIRS:
                return f"skip-dir:{part}"

        # ── Skip-names (lock files etc.) ────────────────────
        if filepath.name in SKIP_NAMES:
            return f"skip-name:{filepath.name}"

        # ── Dotfiles (hidden) ───────────────────────────────
        if filepath.name.startswith(".") and filepath.name not in INDEXABLE_NAMES:
            return "hidden"

        # ── Extension check ─────────────────────────────────
        suffix = filepath.suffix.lower()
        name = filepath.name

        # Check skip-suffixes first (covers compound like .min.js)
        for skip_suffix in SKIP_SUFFIXES:
            if name.endswith(skip_suffix):
                return f"skip-suffix:{skip_suffix}"

        # Must have an indexable extension or be a known name
        if suffix not in INDEXABLE_EXTENSIONS and name not in INDEXABLE_NAMES:
            return f"unknown-ext:{suffix or '(none)'}"

        # ── Size check ──────────────────────────────────────
        try:
            size = filepath.stat().st_size
            if size > MAX_FILE_SIZE:
                return f"too-large:{size // 1024}KB"
            if size == 0:
                return "empty"
        except OSError:
            return "stat-error"

        # ── .gitignore ──────────────────────────────────────
        try:
            rel = filepath.relative_to(self.root)
            rel_str = str(rel)
        except ValueError:
            rel_str = filepath.name

        if self._gitignore_spec and self._gitignore_spec.match_file(rel_str):
            return "gitignored"

        if self._ragmcpignore_spec and self._ragmcpignore_spec.match_file(rel_str):
            return "rag-mcpignored"

        return ""  # OK to index

    def collect_files(self, directory: Optional[Path] = None) -> tuple[list[Path], int]:
        """Walk directory, return (indexable_files, skipped_count)."""
        root = (directory or self.root).resolve()
        indexable: list[Path] = []
        skipped = 0

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            reason = self.should_index(path)
            if reason:
                skipped += 1
            else:
                indexable.append(path)

        logger.info(
            f"File filter: {len(indexable)} indexable, {skipped} skipped "
            f"in {root}"
        )
        return indexable, skipped
