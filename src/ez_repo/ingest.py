"""Clone a public GitHub repository (shallow, read-only) and list its Python files.

Security rule: repository code is only read as text, never imported or executed.
"""

import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from ez_repo.config import Settings, get_settings

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    "site-packages",
    "build",
    "dist",
}
PYTHON_SUFFIXES = {".py"}

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")  # SHA-1 or SHA-256 object names
_NOT_FOUND_HINTS = (
    "repository not found",
    "could not read username",
    "authentication failed",
    "terminal prompts disabled",
)


class IngestionError(Exception):
    """Base error for repository ingestion."""


class InvalidRepoURL(IngestionError):
    pass


class RepositoryNotFound(IngestionError):
    pass


class CloneError(IngestionError):
    pass


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass(frozen=True)
class SourceFile:
    path: str  # relative to the repository root, POSIX separators
    text: str
    size: int


@dataclass
class IngestResult:
    repo: RepoRef | None
    root: Path
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    truncated: bool = False  # True when max_files / max_total_size stopped the scan
    commit_sha: str | None = None  # checked-out commit, None when git could not tell


def is_commit_sha(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA_RE.match(value))


def parse_github_url(url: str) -> RepoRef:
    """Validate a public GitHub repository URL and return its owner/name."""
    url = (url or "").strip()
    if not url or any(c.isspace() for c in url):
        raise InvalidRepoURL("URL is empty or contains spaces.")
    if "://" not in url:
        url = "https://" + url

    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise InvalidRepoURL("Only http(s) GitHub URLs are supported.")
    try:
        has_extras = parts.username or parts.password or parts.port
    except ValueError as exc:  # malformed port, e.g. "git@github.com:owner/repo"
        raise InvalidRepoURL("Malformed URL.") from exc
    if has_extras:
        raise InvalidRepoURL("URL must not contain credentials or a port.")
    if (parts.hostname or "").lower() not in {"github.com", "www.github.com"}:
        raise InvalidRepoURL("Only github.com repositories are supported.")

    segments = [s for s in parts.path.split("/") if s]
    if len(segments) < 2:
        raise InvalidRepoURL("Expected https://github.com/<owner>/<repo>.")
    owner, name = segments[0], segments[1]
    if name.endswith(".git"):
        name = name[:-4]
    if not _OWNER_RE.match(owner) or not _REPO_RE.match(name) or name in {".", ".."}:
        raise InvalidRepoURL(f"Invalid GitHub owner or repository name: {owner}/{name}")
    return RepoRef(owner=owner, name=name)


def clone_dir(repo: RepoRef, dest_root: Path) -> Path:
    return Path(dest_root) / f"{repo.owner}__{repo.name}"


def clone_repo(repo: RepoRef, dest_root: Path, timeout: int) -> Path:
    """Shallow-clone `repo` into `dest_root/<owner>__<name>`, replacing any previous clone."""
    dest = clone_dir(repo, dest_root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _remove_tree(dest)

    cmd = [
        "git",
        "-c", "credential.helper=",  # never prompt for credentials (public repos only)
        "-c", "core.symlinks=false",  # symlinks become plain text files
        "-c", "core.longpaths=true",
        "clone", "--depth", "1", "--single-branch", "--no-tags", "--quiet",
        repo.url + ".git", str(dest),
    ]  # fmt: skip
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
    except FileNotFoundError as exc:
        raise CloneError("git is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        _remove_tree(dest)
        raise CloneError(f"Cloning {repo.slug} timed out after {timeout}s.") from exc

    if result.returncode != 0:
        _remove_tree(dest)
        stderr = (result.stderr or "").strip()
        if any(hint in stderr.lower() for hint in _NOT_FOUND_HINTS):
            raise RepositoryNotFound(f"Repository {repo.slug} not found or not public.")
        last_line = stderr.splitlines()[-1] if stderr else f"exit code {result.returncode}"
        raise CloneError(f"Cloning {repo.slug} failed: {last_line}")
    return dest


def head_commit(root: Path, timeout: int = 10) -> str | None:
    """SHA of the commit checked out in `root`, or None. Only reads git metadata."""
    cmd = ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", "HEAD^{commit}"]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = (result.stdout or "").strip()
    return sha if result.returncode == 0 and is_commit_sha(sha) else None


def remove_clone(repo: RepoRef, dest_root: Path) -> bool:
    """Delete the local clone of `repo`. Returns True if there was one."""
    dest = clone_dir(repo, dest_root)
    existed = dest.exists()
    _remove_tree(dest)
    return existed


def collect_python_files(
    root: Path, max_file_size: int, max_files: int, max_total_size: int
) -> IngestResult:
    """Read Python files under `root` as text, skipping ignored, binary and oversized files."""
    root = Path(root)
    result = IngestResult(repo=None, root=root)
    total = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            if path.suffix not in PYTHON_SUFFIXES:
                continue
            if path.is_symlink() or not path.is_file():
                result.skipped.append((rel, "not a regular file"))
                continue
            size = path.stat().st_size
            if size > max_file_size:
                result.skipped.append((rel, "too large"))
                continue
            if len(result.files) >= max_files or total + size > max_total_size:
                result.truncated = True
                result.skipped.append((rel, "limit reached"))
                continue

            data = path.read_bytes()
            if b"\x00" in data[:8192]:
                result.skipped.append((rel, "binary"))
                continue
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                result.skipped.append((rel, "not utf-8"))
                continue

            result.files.append(SourceFile(path=rel, text=text, size=size))
            total += size
    return result


def ingest_repository(url: str, settings: Settings | None = None) -> IngestResult:
    """Validate the URL, clone the repository and collect its Python files."""
    settings = settings or get_settings()
    repo = parse_github_url(url)
    root = clone_repo(repo, settings.repos_dir, settings.clone_timeout)
    result = collect_python_files(
        root, settings.max_file_size, settings.max_files, settings.max_total_size
    )
    result.repo = repo
    result.commit_sha = head_commit(root)
    return result


def _remove_tree(path: Path) -> None:
    """Delete a directory, including read-only files git creates on Windows."""
    if not path.exists():
        return

    def make_writable_and_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=make_writable_and_retry)
