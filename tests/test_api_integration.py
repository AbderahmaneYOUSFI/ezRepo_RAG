"""The API over real services: Neo4j, MiniLM, Ollama and GitHub. Run with: pytest -m integration

The chat tests run the 3B model on CPU (about 25-90 s each); the import test clones
pypa/sampleproject from GitHub.
"""

import re
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from ez_repo.api import create_app
from ez_repo.config import Settings
from ez_repo.graph import GraphStore, index_repository
from ez_repo.tasks import TaskManager
from fakes import demo_graph

pytestmark = pytest.mark.integration

SHA = "0123456789abcdef0123456789abcdef01234567"
SAMPLE = "pypa/sampleproject"


class BrokenEmbedder:
    model_name = "broken"
    dimensions = 384

    def embed_documents(self, texts):
        raise RuntimeError("embedding crashed")


@pytest.fixture(scope="module")
def store():
    with GraphStore() as store:
        store.verify()
        yield store


@pytest.fixture
def client(store, tmp_path):
    tasks = TaskManager()
    with TestClient(create_app(Settings(repos_dir=tmp_path), store=store, tasks=tasks)) as client:
        yield client
    tasks.shutdown()


@pytest.fixture
def repo_id(store):
    repo_id = f"test/r{uuid.uuid4().hex[:12]}"
    yield repo_id
    store.delete_repository(repo_id)


def wait_task(client, task_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(1)
    raise AssertionError(f"indexing still {task['status']} after {timeout}s ({task['stage']})")


def test_services_health(client):
    body = client.get("/health/services").json()
    assert body["neo4j"]["ok"] and body["ollama"]["ok"], body


def test_delete_through_the_api(client, store, repo_id):
    index_repository(store, repo_id, demo_graph(), url=f"https://github.com/{repo_id}")
    assert repo_id in {r["id"] for r in client.get("/repositories").json()}

    response = client.delete(f"/repositories/{repo_id}")
    assert response.status_code == 200 and response.json()["deleted_nodes"] > 10
    assert store.get_repository(repo_id) is None
    assert client.delete(f"/repositories/{repo_id}").status_code == 404


def test_failed_indexing_is_visible_and_blocks_questions(client, store, repo_id):
    with pytest.raises(RuntimeError):
        index_repository(store, repo_id, demo_graph(), embedder=BrokenEmbedder())

    repos = {r["id"]: r for r in client.get("/repositories").json()}
    assert repos[repo_id]["status"] == "failed"
    assert "embedding crashed" in repos[repo_id]["error"]
    response = client.post("/chat", json={"repo_id": repo_id, "question": "greet?"})
    assert response.status_code == 409


def test_chat_links_sources_to_the_indexed_commit(client, store, repo_id):
    url = f"https://github.com/{repo_id}"
    index_repository(store, repo_id, demo_graph(), url=url, commit_sha=SHA)

    response = client.post(
        "/chat", json={"repo_id": repo_id, "question": "Où se trouve la fonction greet ?"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"].strip() and body["commit_sha"] == SHA
    assert body["invalid_citations"] == [] and body["unknown_paths"] == []
    first = body["sources"][0]
    assert first["location"] == "pkg/core.py:12-17"
    assert first["url"] == f"{url}/blob/{SHA}/pkg/core.py#L12-L17"
    assert any(s["cited"] for s in body["sources"])


def test_import_and_question_a_github_repository(client, store):
    existed = store.get_repository(SAMPLE) is not None
    try:
        response = client.post("/repositories", json={"url": f"https://github.com/{SAMPLE}"})
        assert response.status_code == 202
        task = wait_task(client, response.json()["id"], timeout=600)
        assert task["status"] == "completed", task["error"]
        result = task["result"]
        assert result["repo_id"] == SAMPLE and result["symbols"] > 0
        assert re.fullmatch(r"[0-9a-f]{40}", result["commit_sha"])

        repo = {r["id"]: r for r in client.get("/repositories").json()}[SAMPLE]
        assert repo["status"] == "ready" and repo["commit_sha"] == result["commit_sha"]

        body = client.post(
            "/chat", json={"repo_id": SAMPLE, "question": "Where is the function add_one defined?"}
        ).json()
        sources = {s["path"]: s for s in body["sources"]}
        assert "src/sample/simple.py" in sources
        assert sources["src/sample/simple.py"]["url"].startswith(
            f"https://github.com/{SAMPLE}/blob/{result['commit_sha']}/src/sample/simple.py#L"
        )
    finally:
        if not existed:
            store.delete_repository(SAMPLE)
