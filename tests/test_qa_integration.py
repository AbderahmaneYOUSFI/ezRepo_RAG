"""GraphRAG end to end: Neo4j + MiniLM + Ollama. Run with: pytest -m integration

Each question runs the 3B model on CPU (about 25-90 s). Assertions on the generated text
stay loose; sources are checked exactly since they come from Neo4j metadata.
"""

import uuid

import pytest

from ez_repo.graph import GraphStore, index_repository
from ez_repo.qa import NO_CONTEXT_ANSWER, ask_repository
from fakes import demo_graph, graph_from

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def store():
    with GraphStore() as store:
        yield store


@pytest.fixture(scope="module")
def repos(store):
    demo, other = f"test/{uuid.uuid4().hex[:12]}", f"test/{uuid.uuid4().hex[:12]}"
    index_repository(store, demo, demo_graph())
    index_repository(
        store, other, graph_from({"other.py": 'def greet():\n    """Say hi."""\n    return "hi"\n'})
    )
    yield demo, other
    store.delete_repository(demo)
    store.delete_repository(other)


def check_grounded(answer):
    assert answer.answer.strip()
    assert answer.invalid_citations == [] and answer.unknown_paths == []


def test_where_is_a_function(store, repos):
    answer = ask_repository(store, repos[0], "Où se trouve la fonction greet ?")
    check_grounded(answer)
    assert answer.language == "fr"
    assert answer.sources[0].location == "pkg/core.py:12-17"
    assert "pkg/core.py" in answer.answer
    assert "pkg/core.py::greet" in {s.symbol_id for s in answer.cited_sources}


def test_explain_a_class(store, repos):
    answer = ask_repository(store, repos[0], "Explique la classe Greeter.")
    check_grounded(answer)
    assert answer.sources[0].symbol_id == "pkg/core.py::Greeter"
    assert "Greeter" in answer.answer


def test_who_calls_a_function(store, repos):
    answer = ask_repository(store, repos[0], "Which functions call greet?")
    check_grounded(answer)
    assert answer.language == "en"
    ids = {s.symbol_id for s in answer.sources}
    assert {"app.py::App.main", "pkg/core.py::Greeter.run"} <= ids
    assert "main" in answer.answer or "run" in answer.answer


def test_how_a_module_works(store, repos):
    answer = ask_repository(store, repos[0], "Comment fonctionne le module pkg/core.py ?")
    check_grounded(answer)
    assert answer.sources[0].path == "pkg/core.py"
    assert any(s.path == "pkg/core.py" for s in answer.cited_sources)


def test_question_without_answer_in_the_repository(store, repos):
    answer = ask_repository(store, repos[0], "How does the app connect to the PostgreSQL database?")
    assert answer.answer == NO_CONTEXT_ANSWER["en"]
    assert answer.sources == []


def test_repositories_are_isolated(store, repos):
    answer = ask_repository(store, repos[1], "Où se trouve la fonction greet ?")
    check_grounded(answer)
    assert {s.path for s in answer.sources} == {"other.py"}
    assert "pkg/core.py" not in answer.answer
