"""The Streamlit page against the real FastAPI app (fake store, LLM and indexer)."""

import threading
from pathlib import Path

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from streamlit.testing.v1 import AppTest

from ez_repo import qa, retriever
from ez_repo.api import create_app
from ez_repo.config import Settings
from ez_repo.tasks import TaskManager
from fakes import FakeEmbedder, FakeStore, demo_graph

UI = Path(__file__).parents[1] / "src" / "ez_repo" / "ui.py"
REPO = "acme/demo"
SHA = "0123456789abcdef0123456789abcdef01234567"


def summary(repo_id):
    return {
        "repo_id": repo_id, "files": 1, "symbols": 1, "relations": 1, "external_modules": 0,
        "removed": 0, "embedded": 1, "embeddings_reused": 0, "commit_sha": SHA,
        "skipped_files": 0, "files_with_errors": 0, "truncated": False, "warnings": [],
    }  # fmt: skip


@pytest.fixture
def gate():
    gate = threading.Event()
    gate.set()
    yield gate
    gate.set()


@pytest.fixture
def api(monkeypatch, tmp_path, gate):
    """Route the UI's httpx client to an in-process API; returns its store."""
    meta = {REPO: {"url": "https://github.com/acme/demo", "commit_sha": SHA}}
    store = FakeStore({REPO: demo_graph()}, meta=meta)

    def indexer(store, url, progress, settings):
        assert gate.wait(5)
        return summary(url.removeprefix("https://github.com/").lower())

    tasks = TaskManager()
    app = create_app(Settings(_env_file=None, repos_dir=tmp_path), store=store, tasks=tasks,
                     indexer=indexer)  # fmt: skip
    client = TestClient(app)

    class Routed:  # the UI's httpx.Client, minus the timeout TestClient does not support
        def request(self, method, path, timeout=None, **kwargs):
            return client.request(method, path, **kwargs)

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: Routed())
    monkeypatch.setattr(retriever, "get_embedder", lambda *a, **k: FakeEmbedder())
    answer = AIMessage(content="`greet` builds the message [S1] (pkg/core.py:12-17).",
                       response_metadata={"done_reason": "length"})  # fmt: skip
    monkeypatch.setattr(qa, "get_llm", lambda settings=None: RunnableLambda(lambda _: answer))
    st.cache_resource.clear()
    st.cache_data.clear()
    yield store
    tasks.shutdown()
    st.cache_resource.clear()
    st.cache_data.clear()


def run_page() -> AppTest:
    page = AppTest.from_file(str(UI), default_timeout=20)
    page.run()
    assert not page.exception, page.exception
    return page


def texts(page) -> str:
    parts = [m.value for m in page.markdown] + [c.value for c in page.caption]
    parts += [e.value for e in (*page.error, *page.warning, *page.info)]
    return "\n".join(str(p) for p in parts)


def test_lists_repositories_and_answers_with_sources(api):
    page = run_page()
    assert page.radio[0].options == [f":green[●] {REPO}"]
    assert "### acme/demo" in texts(page)
    assert "commit [`0123456`]" in texts(page)

    page.chat_input[0].set_value("Where is greet?").run()
    assert not page.exception, page.exception

    assert [m.name for m in page.chat_message] == ["user", "assistant"]
    shown = texts(page)
    assert "`greet` builds the message [S1]" in shown
    assert "incomplete" in shown  # done_reason == "length"
    link = f"https://github.com/acme/demo/blob/{SHA}/pkg/core.py#L12-L17"
    assert f"[`pkg/core.py:12-17`]({link})" in shown
    assert any(e.label.startswith("Sources · 1 cited") for e in page.expander)


def test_empty_state(api):
    api.graphs.clear()
    page = run_page()
    assert "## ez-repo" in texts(page)
    assert "No repository yet" in texts(page)
    assert any(b.label == "Analyze repository" for b in page.button)


def test_import_invalid_url_shows_the_error(api):
    page = run_page()
    page.text_input[0].input("https://gitlab.com/a/b")
    next(b for b in page.button if b.label == "Analyze repository").click().run()
    assert not page.exception
    assert any("github.com" in e.value for e in page.error)


def test_import_disables_actions_while_indexing(api, gate):
    gate.clear()
    page = run_page()
    page.text_input[0].input("https://github.com/acme/demo")
    next(b for b in page.button if b.label == "Analyze repository").click().run()
    assert not page.exception, page.exception

    analyze = next(b for b in page.button if b.label == "Analyze repository")
    delete = next(b for b in page.button if b.label == "Delete repository")
    assert analyze.disabled and delete.disabled
    assert page.chat_input[0].disabled
    assert "being indexed" in texts(page)
    assert any(s.label == "Indexing acme/demo" for s in page.status)


def test_failed_repository_disables_questions(api):
    api.meta[REPO] = {"status": "failed", "error": "Neo4j went away"}
    page = run_page()
    assert page.chat_input[0].disabled
    assert "Neo4j went away" in texts(page)


def test_api_down_is_shown_without_crashing(monkeypatch, api):
    class Down:
        def request(self, *args, **kwargs):
            raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "Client", lambda *args, **kwargs: Down())
    st.cache_resource.clear()
    page = run_page()
    assert any("Cannot reach the ez-repo API" in e.value for e in page.error)


def test_delete_asks_for_confirmation(api):
    page = run_page()
    next(b for b in page.button if b.label == "Delete repository").click().run()
    assert not page.exception, page.exception
    assert api.deleted == []  # only the dialog opened

    next(b for b in page.button if b.label == "Delete").click().run()
    assert not page.exception, page.exception
    assert api.deleted == [REPO]
    assert "No repository yet" in texts(page)
