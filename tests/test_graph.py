"""Neo4j storage without a server: pure helpers and a fake driver."""

from pathlib import Path

import pytest
from neo4j.exceptions import AuthError, ClientError, ServiceUnavailable

from ez_repo.config import Settings
from ez_repo.graph import (
    GraphAuthError,
    GraphQueryError,
    GraphStore,
    GraphUnavailable,
    IndexStats,
    index_repository,
    node_labels,
    relation_groups,
    repo_id_for,
    symbol_properties,
)
from ez_repo.ingest import RepoRef, collect_python_files
from ez_repo.parser import CodeGraph, Relation, parse_repository

DEMO_REPO = Path(__file__).parent / "fixtures" / "demo_repo"


@pytest.fixture(scope="module")
def graph():
    return parse_repository(collect_python_files(DEMO_REPO, 10**6, 100, 10**7).files)


class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeDriver:
    """Records queries; DELETE queries report `deleted` from a queue, others return nothing."""

    def __init__(self, error=None, deleted=()):
        self.calls = []
        self.error = error
        self.deleted = list(deleted)

    def execute_query(self, query, parameters_=None, routing_=None, database_=None):
        if self.error:
            raise self.error
        self.calls.append((query, parameters_))
        if "AS deleted" in query:
            return [FakeRecord(deleted=self.deleted.pop(0) if self.deleted else 0)], None, None
        return [], None, None

    def close(self):
        pass


def make_store(driver, batch_size=500):
    return GraphStore(Settings(_env_file=None, neo4j_batch_size=batch_size), driver=driver)


def test_repo_id_is_case_insensitive():
    assert repo_id_for(RepoRef("PyPA", "SampleProject")) == "pypa/sampleproject"


def test_node_labels(graph):
    labels = node_labels(graph)
    assert labels["pkg/core.py"] == "File"
    assert labels["pkg/core.py::Greeter.run"] == "Symbol"
    assert labels["module:os"] == "ExternalModule"


def test_relation_groups_cover_every_relation(graph):
    groups = relation_groups(graph)
    assert set(groups) == {
        ("DEFINES", "File", "Symbol"),
        ("DEFINES", "Symbol", "Symbol"),
        ("IMPORTS", "File", "File"),
        ("IMPORTS", "File", "ExternalModule"),
        ("CALLS", "Symbol", "Symbol"),
        ("INHERITS", "Symbol", "Symbol"),
    }
    assert sum(len(rows) for rows in groups.values()) == len(graph.relations)
    assert {"source": "pkg/core.py::Greeter", "target": "pkg/base.py::Base", "line": 24} in (
        groups[("INHERITS", "Symbol", "Symbol")]
    )


def test_relations_to_unknown_nodes_are_skipped(graph):
    bad = CodeGraph(graph.files, [Relation("CALLS", "pkg/core.py::greet", "nowhere::f")])
    assert relation_groups(bad) == {}


def test_symbol_properties_are_neo4j_values(graph):
    symbol = next(s for s in graph.symbols if s.id == "pkg/core.py::Greeter")
    props = symbol_properties(symbol, "pkg.core")
    assert props["module"] == "pkg.core"
    assert props["bases"] == ["Base"] and props["decorators"] == []
    for value in props.values():
        assert value is None or isinstance(value, str | int | bool | list)


def test_write_graph_is_batched_and_parameterized(graph):
    driver = FakeDriver()
    make_store(driver, batch_size=2).write_graph("acme/demo", graph, run="r1")

    assert all(len(params["rows"]) <= 2 for _, params in driver.calls)
    assert all(params["repo_id"] == "acme/demo" and params["run"] == "r1"
               for _, params in driver.calls)  # fmt: skip
    written = [row["id"] for q, p in driver.calls if "MERGE (f:File" in q for row in p["rows"]]
    assert sorted(written) == sorted(f.path for f in graph.files)
    # Values travel as parameters, never inside the query text.
    queries = " ".join(q for q, _ in driver.calls)
    assert "pkg/core.py" not in queries and "acme/demo" not in queries


def test_symbols_get_their_kind_label(graph):
    driver = FakeDriver()
    make_store(driver).write_graph("acme/demo", graph, run="r1")
    by_label = {
        label: {row["id"] for row in params["rows"]}
        for query, params in driver.calls
        for label in ("Class", "Function")
        if f"SET s:{label}" in query
    }
    assert "pkg/core.py::Greeter" in by_label["Class"]
    assert {"pkg/core.py::greet", "pkg/core.py::Greeter.run"} <= by_label["Function"]


def test_stale_data_is_deleted_until_exhausted():
    # File: 2 full batches then a partial one; every other query deletes nothing.
    driver = FakeDriver(deleted=[2, 2, 1])
    assert make_store(driver, batch_size=2).remove_stale("acme/demo", "r2") == 5
    assert all(params["run"] == "r2" and params["batch"] == 2 for _, params in driver.calls)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ServiceUnavailable("connection refused"), GraphUnavailable),
        (AuthError("unauthorized"), GraphAuthError),
        (ClientError("syntax error"), GraphQueryError),
    ],
)
def test_neo4j_errors_are_translated(error, expected):
    with pytest.raises(expected):
        make_store(FakeDriver(error=error)).list_repositories()


def test_invalid_uri_is_reported():
    with pytest.raises(Exception, match="Invalid Neo4j configuration"):
        GraphStore(Settings(_env_file=None, neo4j_uri="ftp://nowhere"))


class RecordingStore:
    """Just enough of GraphStore for index_repository; `fail_on` names a failing step."""

    batch_size = 500

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == self.fail_on:
                raise GraphUnavailable(f"{name} failed")
            return {} if name == "embedding_hashes" else 0

        return method


class StubEmbedder:
    model_name = "stub"
    dimensions = 3

    def embed_documents(self, texts):
        return [[0.0, 0.0, 1.0] for _ in texts]


def test_index_repository_stores_the_commit(graph):
    store = RecordingStore()
    index_repository(store, "acme/demo", graph, embedder=StubEmbedder(), commit_sha="abc")
    name, args, _ = store.calls[-1]
    assert name == "finish" and args[2:] == ("stub", "abc")


@pytest.mark.parametrize("step", ["write_graph", "save_embeddings", "finish"])
def test_failed_indexing_marks_the_repository(graph, step):
    store = RecordingStore(fail_on=step)
    with pytest.raises(GraphUnavailable):
        index_repository(store, "acme/demo", graph, embedder=StubEmbedder())
    assert store.calls[-1][0] == "mark_failed"
    assert store.calls[-1][1] == ("acme/demo", f"{step} failed")


def test_failure_before_writing_leaves_the_repository_alone(graph):
    store = RecordingStore(fail_on="ensure_schema")
    with pytest.raises(GraphUnavailable):
        index_repository(store, "acme/demo", graph, embedder=StubEmbedder())
    assert "mark_failed" not in [c[0] for c in store.calls]


def test_finish_overwrites_the_commit_and_mark_failed_truncates():
    driver = FakeDriver()
    store = make_store(driver)
    store.finish("acme/demo", IndexStats("acme/demo"), "m")
    store.mark_failed("acme/demo", "x" * 2000)
    (finish_query, finish_params), (_, failed_params) = driver.calls
    assert "r.commit_sha = $commit_sha" in finish_query and finish_params["commit_sha"] is None
    assert len(failed_params["error"]) == 500
