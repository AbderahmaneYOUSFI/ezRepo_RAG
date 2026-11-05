"""Real clones from GitHub (network). Run with: pytest -m integration"""

import pytest

from ez_repo.config import Settings
from ez_repo.ingest import RepositoryNotFound, ingest_repository
from ez_repo.parser import parse_repository

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, repos_dir=tmp_path, clone_timeout=120)


def test_ingest_and_parse_small_public_repo(settings):
    result = ingest_repository("https://github.com/pypa/sampleproject", settings)

    assert result.repo.slug == "pypa/sampleproject"
    paths = {f.path for f in result.files}
    assert "src/sample/simple.py" in paths
    assert not any(p.startswith(".git/") for p in paths)

    graph = parse_repository(result.files)
    assert "src/sample/simple.py::add_one" in {s.id for s in graph.symbols}


def test_missing_repository(settings):
    with pytest.raises(RepositoryNotFound):
        ingest_repository("https://github.com/pypa/this-repo-does-not-exist-ez-repo", settings)
