"""Indexing into a running Neo4j. Run with: pytest -m integration"""

import hashlib
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from ez_repo.config import Settings
from ez_repo.embeddings import EMBEDDING_DIMENSIONS, get_embedder
from ez_repo.graph import GraphStore, index_repository
from ez_repo.ingest import collect_python_files
from ez_repo.parser import CodeGraph, build_relations, parse_repository

pytestmark = pytest.mark.integration

DEMO_REPO = Path(__file__).parent / "fixtures" / "demo_repo"


class FakeEmbedder:
    """Deterministic vectors, no model: keeps graph tests fast."""

    model_name = "fake"
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts.extend(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        seed = hashlib.sha256(text.encode()).digest()
        return [(seed[i % 32] - 127.5) / 128 for i in range(EMBEDDING_DIMENSIONS)]


@pytest.fixture(scope="module")
def store():
    # A tiny batch size exercises multi-transaction writes and deletes.
    with GraphStore(Settings(neo4j_batch_size=3)) as store:
        store.verify()
        yield store


@pytest.fixture
def repo_ids(store):
    ids = []

    def new_id():
        ids.append(f"test/{uuid.uuid4().hex[:12]}")
        return ids[-1]

    yield new_id
    for repo_id in ids:
        store.delete_repository(repo_id)


@pytest.fixture(scope="module")
def graph():
    return parse_repository(collect_python_files(DEMO_REPO, 10**6, 100, 10**7).files)


def stored_edges(store, repo_id):
    return {(r.type, r.source_id, r.target_id) for r in store.get_relations(repo_id)}


def expected_edges(graph, repo_id):
    edges = {(r.type, r.source_id, r.target_id) for r in graph.relations}
    return edges | {("CONTAINS", repo_id, f.path) for f in graph.files}


def test_index_demo_repository(store, repo_ids, graph):
    repo_id = repo_ids()
    stats = index_repository(store, repo_id, graph, url="https://x/y", embedder=FakeEmbedder())

    assert stats.embedded == len(graph.symbols) and stats.embeddings_reused == 0
    repo = store.get_repository(repo_id)
    assert repo["status"] == "ready" and repo["url"] == "https://x/y"
    assert repo["symbol_count"] == len(graph.symbols)

    assert [f["path"] for f in store.get_files(repo_id)] == sorted(f.path for f in graph.files)
    assert {s["id"] for s in store.get_symbols(repo_id)} == {s.id for s in graph.symbols}
    assert stored_edges(store, repo_id) == expected_edges(graph, repo_id)

    classes = store.get_symbols(repo_id, kind="class", path="pkg/core.py")
    assert [c["id"] for c in classes] == ["pkg/core.py::Greeter"]
    greet = store.get_symbol(repo_id, "pkg/core.py::greet")
    assert greet["signature"] == "def greet(name: str) -> str"
    assert greet["source"].startswith("def greet(")


def test_reindex_is_idempotent(store, repo_ids, graph):
    repo_id = repo_ids()
    index_repository(store, repo_id, graph, embedder=FakeEmbedder())
    embedder = FakeEmbedder()
    stats = index_repository(store, repo_id, graph, embedder=embedder)

    assert stats.removed == 0
    assert stats.embedded == 0 and stats.embeddings_reused == len(graph.symbols)
    assert embedder.texts == []
    assert len(store.get_symbols(repo_id)) == len(graph.symbols)
    assert len(store.get_relations(repo_id)) == len(expected_edges(graph, repo_id))


def test_reindex_replaces_old_data(store, repo_ids, graph):
    repo_id = repo_ids()
    index_repository(store, repo_id, graph, embedder=FakeEmbedder())

    # New version: utils.py deleted, greet() documented differently.
    files = []
    for f in graph.files:
        if f.path == "pkg/utils.py":
            continue
        symbols = [replace(s, docstring="Changed.") if s.id == "pkg/core.py::greet" else s
                   for s in f.symbols]  # fmt: skip
        files.append(replace(f, symbols=symbols))
    new_graph = CodeGraph(files, build_relations(files))
    embedder = FakeEmbedder()
    stats = index_repository(store, repo_id, new_graph, embedder=embedder)

    assert stats.removed > 0
    assert stats.embedded == 1 and "docstring: Changed." in embedder.texts[0]
    assert "pkg/utils.py" not in {f["path"] for f in store.get_files(repo_id)}
    assert store.get_symbol(repo_id, "pkg/utils.py::slugify") is None
    assert store.get_symbol(repo_id, "pkg/core.py::greet")["docstring"] == "Changed."
    assert stored_edges(store, repo_id) == expected_edges(new_graph, repo_id)


def test_repositories_are_isolated(store, repo_ids, graph):
    first, second = repo_ids(), repo_ids()
    index_repository(store, first, graph, embedder=FakeEmbedder())
    index_repository(store, second, graph, embedder=FakeEmbedder())

    store.delete_repository(first)

    assert store.get_repository(first) is None
    assert store.get_symbols(first) == [] and store.get_relations(first) == []
    assert len(store.get_symbols(second)) == len(graph.symbols)
    assert stored_edges(store, second) == expected_edges(graph, second)


def test_vector_search_with_real_embeddings(store, repo_ids, graph):
    repo_id, other = repo_ids(), repo_ids()
    embedder = get_embedder()
    index_repository(store, repo_id, graph, embedder=embedder)
    index_repository(store, other, graph, embedder=embedder)

    hits = store.vector_search(repo_id, embedder.embed_query("return a greeting message"), k=3)

    assert len({h["id"] for h in hits}) == 3  # no duplicate from the other repository
    assert all(h["repo_id"] == repo_id for h in hits)
    assert {"pkg/core.py::greet", "pkg/core.py::Greeter"} & {h["id"] for h in hits}
    assert hits == sorted(hits, key=lambda h: h["score"], reverse=True)
