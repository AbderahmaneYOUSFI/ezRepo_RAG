import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ez_repo import __version__, qa, retriever
from ez_repo.api import app, create_app, github_line_url
from ez_repo.config import Settings
from ez_repo.graph import GraphUnavailable
from ez_repo.tasks import TaskManager
from fakes import FakeEmbedder, FakeStore, demo_graph, graph_from

REPO = "acme/demo"
OTHER = "acme/other"
SHA = "0123456789abcdef0123456789abcdef01234567"
META = {"url": "https://github.com/acme/demo", "commit_sha": SHA}


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


# --- fixtures ----------------------------------------------------------------------------


def summary(repo_id, **extra):
    return {
        "repo_id": repo_id, "files": 1, "symbols": 2, "relations": 3, "external_modules": 0,
        "removed": 0, "embedded": 2, "embeddings_reused": 0, "commit_sha": SHA,
        "skipped_files": 0, "files_with_errors": 0, "truncated": False, "warnings": [], **extra,
    }  # fmt: skip


def fake_indexer(store, url, progress, settings):
    progress("cloning the repository")
    return summary(url.removeprefix("https://github.com/").lower())


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, repos_dir=tmp_path / "repos", ollama_model="qwen2.5-coder:3b")


@pytest.fixture
def store():
    other = graph_from({"other.py": 'def greet():\n    """Say hi."""\n    return "hi"\n'})
    return FakeStore({REPO: demo_graph(), OTHER: other}, meta={REPO: META})


@pytest.fixture
def make_client(settings, store):
    managers = []

    def make(indexer=fake_indexer, tasks=None, store=store):
        tasks = tasks or TaskManager(max_workers=1, max_pending=1)
        managers.append(tasks)
        return TestClient(create_app(settings, store=store, tasks=tasks, indexer=indexer))

    yield make
    for manager in managers:
        manager.shutdown()


@pytest.fixture
def client(make_client):
    return make_client()


def answer_with(text, done_reason="stop"):
    return RunnableLambda(
        lambda _prompt: AIMessage(content=text, response_metadata={"done_reason": done_reason})
    )


@pytest.fixture
def llm(monkeypatch):
    """Scripted LLM and embedder behind the real ask_repository."""
    monkeypatch.setattr(retriever, "get_embedder", lambda *a, **k: FakeEmbedder())

    def use(runnable):
        monkeypatch.setattr(qa, "get_llm", lambda settings=None: runnable)

    use(answer_with("`greet` builds the message [S1] (pkg/core.py:12-17)."))
    return use


def wait_task(client, task_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = client.get(f"/tasks/{task_id}").json()
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.02)
    raise AssertionError(f"task still {task['status']}")


# --- repositories ------------------------------------------------------------------------


def test_list_repositories(make_client, store):
    store.meta[OTHER] = {"status": "failed", "error": "Neo4j went away"}
    repos = make_client().get("/repositories").json()

    assert [r["id"] for r in repos] == [REPO, OTHER]
    demo, other = repos
    assert other["status"] == "failed" and other["error"] == "Neo4j went away"
    assert demo["status"] == "ready" and demo["commit_sha"] == SHA and demo["task_id"] is None
    assert set(demo) >= {"url", "indexed_at", "file_count", "symbol_count", "embedding_model"}


def test_repository_left_indexing_without_task_is_interrupted(make_client, store):
    store.meta[REPO] = {"status": "indexing"}
    repos = {r["id"]: r for r in make_client().get("/repositories").json()}
    assert repos[REPO]["status"] == "interrupted"


def test_import_runs_a_background_task(client):
    response = client.post("/repositories", json={"url": "github.com/Acme/NewRepo.git"})
    assert response.status_code == 202
    task = response.json()
    assert task["repo_id"] == "acme/newrepo" and task["url"] == "https://github.com/Acme/NewRepo"
    assert task["status"] in {"pending", "running", "completed"}

    done = wait_task(client, task["id"])
    assert done["status"] == "completed" and done["error"] is None
    assert done["result"]["repo_id"] == "acme/newrepo" and done["result"]["commit_sha"] == SHA
    assert done["elapsed_seconds"] is not None
    assert client.get("/tasks").json()[0]["id"] == task["id"]


@pytest.mark.parametrize(
    "url",
    ["", "https://gitlab.com/a/b", "https://github.com/onlyowner", "git@github.com:a/b.git"],
)
def test_import_rejects_invalid_urls(client, url):
    response = client.post("/repositories", json={"url": url})
    assert response.status_code == 422
    assert response.json()["detail"]
    assert client.get("/tasks").json() == []


def test_import_requires_a_url(client):
    assert client.post("/repositories", json={}).status_code == 422


def test_failed_indexing_is_reported(make_client):
    def broken(store, url, progress, settings):
        raise RuntimeError("Repository acme/gone not found or not public.")

    client = make_client(indexer=broken)
    task = client.post("/repositories", json={"url": "https://github.com/acme/gone"}).json()
    done = wait_task(client, task["id"])
    assert done["status"] == "failed" and "not found" in done["error"] and done["result"] is None


@pytest.fixture
def gate():
    gate = threading.Event()
    yield gate
    gate.set()


def gated_indexer(gate):
    def index(store, url, progress, settings):
        assert gate.wait(5)
        return fake_indexer(store, url, progress, settings)

    return index


def test_same_repository_cannot_be_imported_twice_at_once(make_client, gate):
    client = make_client(indexer=gated_indexer(gate))
    first = client.post("/repositories", json={"url": "https://github.com/acme/demo"}).json()
    second = client.post("/repositories", json={"url": "https://github.com/ACME/demo"})
    assert second.status_code == 409 and second.json()["task_id"] == first["id"]

    repos = {r["id"]: r for r in client.get("/repositories").json()}
    assert repos[REPO]["status"] == "indexing" and repos[REPO]["task_id"] == first["id"]


def test_import_queue_is_bounded(make_client, gate):
    client = make_client(indexer=gated_indexer(gate))
    first = client.post("/repositories", json={"url": "https://github.com/acme/one"}).json()
    deadline = time.monotonic() + 5
    while client.get(f"/tasks/{first['id']}").json()["status"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert client.post("/repositories", json={"url": "https://github.com/acme/two"}).is_success
    third = client.post("/repositories", json={"url": "https://github.com/acme/three"})
    assert third.status_code == 429


def test_unknown_task(client):
    response = client.get("/tasks/does-not-exist")
    assert response.status_code == 404 and "restarts" in response.json()["detail"]


def test_neo4j_down_is_a_503(make_client, store, monkeypatch):
    def down(*args, **kwargs):
        raise GraphUnavailable("Neo4j is unreachable while listing repositories.")

    monkeypatch.setattr(store, "list_repositories", down)
    response = make_client().get("/repositories")
    assert response.status_code == 503 and "unreachable" in response.json()["detail"]


# --- delete ------------------------------------------------------------------------------


def test_delete_repository(client, store, settings):
    clone = settings.repos_dir / "acme__demo"
    (clone / "pkg").mkdir(parents=True)
    (clone / "pkg" / "a.py").write_text("x = 1\n")

    response = client.delete("/repositories/Acme/Demo")
    assert response.status_code == 200
    assert response.json() == {
        "repo_id": REPO, "deleted_nodes": response.json()["deleted_nodes"], "clone_removed": True
    }  # fmt: skip
    assert store.deleted == [REPO] and not clone.exists()
    assert [r["id"] for r in client.get("/repositories").json()] == [OTHER]  # other untouched


def test_delete_unknown_repository(client, store):
    assert client.delete("/repositories/acme/missing").status_code == 404
    assert store.deleted == []


@pytest.mark.parametrize("repo_id", ["acme", "acme/demo/extra", "acme/a b", "-x/demo"])
def test_delete_rejects_invalid_ids(client, store, repo_id):
    assert client.delete(f"/repositories/{repo_id}").status_code in {404, 422}
    assert store.deleted == []


def test_delete_is_refused_while_indexing(make_client, store, gate):
    client = make_client(indexer=gated_indexer(gate))
    client.post("/repositories", json={"url": "https://github.com/acme/demo"})
    response = client.delete("/repositories/acme/demo")
    assert response.status_code == 409 and response.json()["task_id"]
    assert store.deleted == []


# --- chat --------------------------------------------------------------------------------


def test_chat_returns_answer_and_verified_sources(client, llm):
    response = client.post("/chat", json={"repo_id": REPO, "question": "Where is greet?"})
    assert response.status_code == 200
    body = response.json()

    assert body["repo_id"] == REPO and body["question"] == "Where is greet?"
    assert body["answer"].startswith("`greet` builds the message [S1]")
    assert body["language"] == "en" and body["incomplete"] is False
    assert body["commit_sha"] == SHA and body["duration_seconds"] >= 0
    assert body["invalid_citations"] == [] and body["unknown_paths"] == []
    first = body["sources"][0]
    assert first == {
        "label": "S1", "symbol_id": "pkg/core.py::greet", "kind": "function",
        "name": "pkg.core.greet", "path": "pkg/core.py", "start_line": 12, "end_line": 17,
        "location": "pkg/core.py:12-17", "cited": True,
        "url": f"https://github.com/acme/demo/blob/{SHA}/pkg/core.py#L12-L17",
    }  # fmt: skip


def test_chat_reports_incomplete_and_unverified_references(client, llm):
    llm(answer_with("`greet` [S1] [S42] calls lib/ghost.py and", done_reason="length"))
    body = client.post("/chat", json={"repo_id": REPO, "question": "Where is greet?"}).json()
    assert body["incomplete"] is True
    assert body["invalid_citations"] == ["S42"] and body["unknown_paths"] == ["lib/ghost.py"]
    assert "[S42]" not in body["answer"]


@pytest.mark.parametrize(
    "meta",
    [
        {"url": "https://github.com/acme/demo"},  # commit unknown
        {"url": "https://github.com/acme/demo", "commit_sha": "main"},  # not a SHA
        {"url": "https://github.com/evil/demo", "commit_sha": SHA},  # another repository
        {"commit_sha": SHA},  # no URL
    ],
)
def test_chat_never_links_to_an_unverified_commit(client, store, llm, meta):
    store.meta[REPO] = meta
    body = client.post("/chat", json={"repo_id": REPO, "question": "Where is greet?"}).json()
    assert body["sources"] and all(s["url"] is None for s in body["sources"])


def test_chat_is_isolated_per_repository(client, store, llm):
    store.repo_ids.clear()
    llm(answer_with("`greet` returns hi [S1] (other.py:1-3)."))
    body = client.post("/chat", json={"repo_id": OTHER, "question": "Where is greet?"}).json()
    assert {s["path"] for s in body["sources"]} == {"other.py"}
    assert set(store.repo_ids) == {OTHER}
    assert all(s["url"] is None for s in body["sources"])  # OTHER has no known commit


def test_chat_repo_id_is_case_insensitive(client, llm):
    body = client.post("/chat", json={"repo_id": " Acme/Demo ", "question": "greet?"}).json()
    assert body["repo_id"] == REPO


@pytest.mark.parametrize(
    "payload",
    [
        {"repo_id": REPO},
        {"question": "greet?"},
        {"repo_id": REPO, "question": "   "},
        {"repo_id": REPO, "question": "x" * 2001},
        {"repo_id": "not a repo", "question": "greet?"},
        {"repo_id": "acme/demo/x", "question": "greet?"},
    ],
)
def test_chat_validation(client, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_unknown_repository(client, llm):
    response = client.post("/chat", json={"repo_id": "acme/missing", "question": "greet?"})
    assert response.status_code == 404 and "not indexed" in response.json()["detail"]


@pytest.mark.parametrize(
    ("status", "message"), [("indexing", "being indexed"), ("failed", "failed")]
)
def test_chat_on_a_repository_that_is_not_ready(client, store, llm, status, message):
    store.meta[REPO] = {**META, "status": status}
    response = client.post("/chat", json={"repo_id": REPO, "question": "greet?"})
    assert response.status_code == 409 and message in response.json()["detail"]


def test_chat_is_refused_while_the_repository_is_indexed(make_client, llm, gate):
    client = make_client(indexer=gated_indexer(gate))
    client.post("/repositories", json={"url": "https://github.com/acme/demo"})
    response = client.post("/chat", json={"repo_id": REPO, "question": "greet?"})
    assert response.status_code == 409


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ConnectError("refused"), 503),
        (httpx.ReadTimeout("slow"), 504),
        (httpx.RemoteProtocolError("broken"), 502),
    ],
)
def test_chat_llm_errors(client, llm, error, code):
    def fail(_prompt):
        raise error

    llm(RunnableLambda(fail))
    response = client.post("/chat", json={"repo_id": REPO, "question": "Where is greet?"})
    assert response.status_code == code and response.json()["detail"]


def test_one_question_at_a_time(make_client, llm):
    client = make_client()
    slot = client.app.state.chat_slot
    slot.acquire()
    try:
        response = client.post("/chat", json={"repo_id": REPO, "question": "greet?"})
        assert response.status_code == 429
    finally:
        slot.release()
    assert client.post("/chat", json={"repo_id": REPO, "question": "greet?"}).status_code == 200


# --- service health ----------------------------------------------------------------------


def test_services_health(client, monkeypatch):
    class Tags:
        def json(self):
            return {"models": [{"name": "qwen2.5-coder:3b"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Tags())
    body = client.get("/health/services").json()
    assert body["neo4j"] == {"ok": True}
    assert body["ollama"] == {"ok": True, "model": "qwen2.5-coder:3b"}


def test_services_health_reports_failures(client, store, monkeypatch):
    def unreachable(*a, **k):
        raise httpx.ConnectError("refused")

    def neo4j_down():
        raise GraphUnavailable("Neo4j is unreachable")

    monkeypatch.setattr(httpx, "get", unreachable)
    monkeypatch.setattr(store, "verify", neo4j_down)
    body = client.get("/health/services").json()
    assert body["neo4j"]["ok"] is False and "unreachable" in body["neo4j"]["error"]
    assert body["ollama"]["ok"] is False and "Ollama unreachable" in body["ollama"]["error"]


# --- links -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((REPO, "https://github.com/Acme/Demo", SHA, "a.py", 5, 5),
         f"https://github.com/Acme/Demo/blob/{SHA}/a.py#L5"),
        ((REPO, "https://github.com/acme/demo", SHA, "my dir/a b.py", 1, 9),
         f"https://github.com/acme/demo/blob/{SHA}/my%20dir/a%20b.py#L1-L9"),
        ((REPO, "https://github.com/acme/demo", SHA, "../etc/passwd", 1, 2), None),
        ((REPO, "https://github.com/acme/demo", SHA, "/abs.py", 1, 2), None),
        ((REPO, "https://example.com/acme/demo", SHA, "a.py", 1, 2), None),
        ((REPO, "https://github.com/acme/demo", SHA.upper(), "a.py", 1, 2), None),
        ((REPO, "https://github.com/acme/demo", None, "a.py", 1, 2), None),
    ],
)  # fmt: skip
def test_github_line_url(args, expected):
    assert github_line_url(*args) == expected
