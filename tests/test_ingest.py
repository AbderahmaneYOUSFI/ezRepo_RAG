import subprocess
from pathlib import Path

import pytest

from ez_repo import ingest
from ez_repo.ingest import (
    CloneError,
    InvalidRepoURL,
    RepoRef,
    RepositoryNotFound,
    clone_repo,
    collect_python_files,
    parse_github_url,
)

# --- URL validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/pypa/sampleproject",
        "http://github.com/pypa/sampleproject",
        "https://github.com/pypa/sampleproject.git",
        "https://github.com/pypa/sampleproject/",
        "https://www.github.com/pypa/sampleproject",
        "github.com/pypa/sampleproject",
        "  https://github.com/pypa/sampleproject  ",
        "https://github.com/pypa/sampleproject/tree/main/src",
        "https://github.com/pypa/sampleproject?tab=readme#top",
    ],
)
def test_parse_valid_urls(url):
    ref = parse_github_url(url)
    assert ref == RepoRef("pypa", "sampleproject")
    assert ref.url == "https://github.com/pypa/sampleproject"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",
        "https://github.com/",
        "ftp://github.com/owner/repo",
        "file:///C:/repo",
        "git@github.com:owner/repo.git",
        "https://user:token@github.com/owner/repo",
        "https://github.com:8443/owner/repo",
        "https://github.com/-bad/repo",
        "https://github.com/owner/..",
        "https://github.com/owner/.git",
        "https://github.com/owner/re po",
        "https://github.com.evil.com/owner/repo",
    ],
)
def test_parse_invalid_urls(url):
    with pytest.raises(InvalidRepoURL):
        parse_github_url(url)


# --- file collection -----------------------------------------------------------------


def write(root: Path, rel: str, content: bytes | str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8") if isinstance(content, str) else content
    path.write_bytes(data)


@pytest.fixture
def tree(tmp_path):
    write(tmp_path, "main.py", "print('hi')\n")
    write(tmp_path, "pkg/mod.py", "x = 1\n")
    write(tmp_path, "pkg/bom.py", "\ufeffy = 2\n")
    write(tmp_path, "README.md", "# readme\n")
    for ignored in [".git", ".venv", "node_modules", "__pycache__", "venv", "demo.egg-info"]:
        write(tmp_path, f"{ignored}/hidden.py", "z = 3\n")
    write(tmp_path, "big.py", "a = 1\n" * 1000)
    write(tmp_path, "binary.py", b"\x00\x01\x02")
    write(tmp_path, "latin1.py", "s = 'é'\n".encode("latin-1"))
    return tmp_path


def test_collects_only_readable_python_files(tree):
    result = collect_python_files(tree, max_file_size=1000, max_files=100, max_total_size=10**6)
    assert [f.path for f in result.files] == ["main.py", "pkg/bom.py", "pkg/mod.py"]
    assert result.files[1].text == "y = 2\n"  # BOM stripped
    assert dict(result.skipped) == {
        "big.py": "too large",
        "binary.py": "binary",
        "latin1.py": "not utf-8",
    }
    assert not result.truncated


def test_max_files_limit(tree):
    result = collect_python_files(tree, max_file_size=1000, max_files=2, max_total_size=10**6)
    assert len(result.files) == 2
    assert result.truncated
    assert ("pkg/mod.py", "limit reached") in result.skipped


def test_max_total_size_limit(tree):
    result = collect_python_files(tree, max_file_size=1000, max_files=100, max_total_size=15)
    assert [f.path for f in result.files] == ["main.py"]
    assert result.truncated


def test_empty_directory(tmp_path):
    result = collect_python_files(tmp_path, 1000, 100, 10**6)
    assert result.files == [] and result.skipped == [] and not result.truncated


# --- cloning (git is mocked) ---------------------------------------------------------

REPO = RepoRef("owner", "repo")


def fake_run(returncode=0, stderr="", exc=None, calls=None):
    def run(cmd, **kwargs):
        if calls is not None:
            calls.append((cmd, kwargs))
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "partial.py").write_text("x = 1\n")
        if exc is not None:
            raise exc
        return subprocess.CompletedProcess(cmd, returncode, "", stderr)

    return run


def test_clone_success_uses_safe_git_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ingest.subprocess, "run", fake_run(calls=calls))
    dest = clone_repo(REPO, tmp_path, timeout=5)

    assert dest == tmp_path / "owner__repo"
    cmd, kwargs = calls[0]
    assert cmd[:1] == ["git"] and "clone" in cmd
    assert {"--depth", "1", "--single-branch"} <= set(cmd)
    assert "credential.helper=" in cmd and "core.symlinks=false" in cmd
    assert not any("submodule" in part for part in cmd)
    assert cmd[-2] == "https://github.com/owner/repo.git"
    assert kwargs["timeout"] == 5
    assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_replaces_previous_clone(monkeypatch, tmp_path):
    old = tmp_path / "owner__repo"
    write(old, "stale.py", "old = True\n")
    (old / "stale.py").chmod(0o444)  # git objects are read-only on Windows
    monkeypatch.setattr(ingest.subprocess, "run", fake_run())
    clone_repo(REPO, tmp_path, timeout=5)
    assert not (old / "stale.py").exists()


@pytest.mark.parametrize(
    "stderr",
    [
        "remote: Repository not found.\nfatal: repository 'x' not found",
        "fatal: could not read Username for 'https://github.com': terminal prompts disabled",
    ],
)
def test_clone_missing_or_private_repo(monkeypatch, tmp_path, stderr):
    monkeypatch.setattr(ingest.subprocess, "run", fake_run(returncode=128, stderr=stderr))
    with pytest.raises(RepositoryNotFound):
        clone_repo(REPO, tmp_path, timeout=5)
    assert not (tmp_path / "owner__repo").exists()


def test_clone_generic_failure(monkeypatch, tmp_path):
    stderr = "fatal: unable to access 'https://github.com/': Could not resolve host: github.com"
    monkeypatch.setattr(ingest.subprocess, "run", fake_run(returncode=128, stderr=stderr))
    with pytest.raises(CloneError, match="Could not resolve host"):
        clone_repo(REPO, tmp_path, timeout=5)
    assert not (tmp_path / "owner__repo").exists()


def test_clone_timeout_cleans_up(monkeypatch, tmp_path):
    exc = subprocess.TimeoutExpired(cmd="git", timeout=5)
    monkeypatch.setattr(ingest.subprocess, "run", fake_run(exc=exc))
    with pytest.raises(CloneError, match="timed out"):
        clone_repo(REPO, tmp_path, timeout=5)
    assert not (tmp_path / "owner__repo").exists()


def test_clone_without_git(monkeypatch, tmp_path):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ingest.subprocess, "run", missing_git)
    with pytest.raises(CloneError, match="git is not installed"):
        clone_repo(REPO, tmp_path, timeout=5)


# --- indexed commit and cleanup ------------------------------------------------------


def git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_head_commit_of_a_local_repository(tmp_path):
    write(tmp_path, "a.py", "x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")

    sha = ingest.head_commit(tmp_path)
    assert ingest.is_commit_sha(sha) and len(sha) == 40


def test_head_commit_unknown(tmp_path):
    assert ingest.head_commit(tmp_path / "not-a-repo") is None


@pytest.mark.parametrize("stdout", ["", "main\n", "ref: refs/heads/main\n", "ABC" * 14])
def test_head_commit_rejects_unexpected_output(monkeypatch, tmp_path, stdout):
    monkeypatch.setattr(
        ingest.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout, ""),
    )
    assert ingest.head_commit(tmp_path) is None


def test_ingest_records_the_commit(monkeypatch, tmp_path):
    sha = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(ingest.subprocess, "run", fake_run())
    monkeypatch.setattr(ingest, "head_commit", lambda root: sha)
    settings = ingest.Settings(_env_file=None, repos_dir=tmp_path)
    result = ingest.ingest_repository("https://github.com/owner/repo", settings)
    assert result.commit_sha == sha and [f.path for f in result.files] == ["partial.py"]


def test_remove_clone(tmp_path):
    write(tmp_path / "owner__repo", "a.py", "x = 1\n")
    (tmp_path / "owner__repo" / "a.py").chmod(0o444)
    assert ingest.remove_clone(REPO, tmp_path) is True
    assert not (tmp_path / "owner__repo").exists()
    assert ingest.remove_clone(REPO, tmp_path) is False
