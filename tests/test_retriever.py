import pytest

from ez_repo.config import Settings
from ez_repo.retriever import is_test_path, question_terms, retrieve
from fakes import FakeEmbedder, FakeStore, demo_graph, graph_from

REPO = "acme/demo"


@pytest.fixture(scope="module")
def graph():
    return demo_graph()


def run(graph, question, hits=(), **settings):
    store = FakeStore({REPO: graph}, hits={REPO: list(hits)})
    result = retrieve(
        store,
        REPO,
        question,
        embedder=FakeEmbedder(),
        settings=Settings(_env_file=None, **settings),
    )
    return result, store


def ids(result):
    return [s.id for s in result.symbols]


def test_question_terms():
    names, paths = question_terms("Où se trouve la fonction greet() dans pkg\\core.py ?")
    assert names == {"greet"} and paths == {"pkg/core.py"}
    assert question_terms("Which functions call Greeter.run?")[0] == {"Greeter.run"}
    assert question_terms("Comment fonctionne pkg.core ?")[0] == {"pkg.core"}


def test_exact_symbol_match(graph):
    result, _ = run(graph, "Où se trouve la fonction greet ?")
    greet = result.symbols[0]
    assert (greet.id, greet.label, greet.location) == (
        "pkg/core.py::greet",
        "S1",
        "pkg/core.py:12-17",
    )
    assert "name in question" in greet.reasons
    assert "[S1] function pkg.core.greet (pkg/core.py:12-17)" in result.context
    assert "  17 |     return format_message(make_slug(name))" in result.context


def test_exact_match_is_case_sensitive(graph):
    result, _ = run(graph, "how does the app start?")
    assert result.empty  # "app" is a word, not the App class nor the app.py module


def test_callers_are_found_through_the_graph(graph):
    result, _ = run(graph, "Quelles fonctions appellent greet ?")
    assert {"app.py::App.main", "pkg/core.py::Greeter.run"} <= set(ids(result))
    calls = {(r.source_id, r.target_id) for r in result.relations if r.type == "CALLS"}
    assert ("app.py::App.main", "pkg/core.py::greet") in calls
    labels = {s.id: s.label for s in result.symbols}
    callers = next(line for line in result.context.splitlines() if "is called by:" in line)
    assert callers.startswith("- pkg.core.greet [S1] is called by: ")
    assert f"app.App.main [{labels['app.py::App.main']}] (app.py, line 9)" in callers
    run_label = labels["pkg/core.py::Greeter.run"]
    assert f"pkg.core.Greeter.run [{run_label}] (pkg/core.py, line 45)" in callers
    assert "- pkg.core.greet [S1] calls: pkg.core.format_message [S" in result.context


def test_relations_are_told_from_the_best_ranked_symbol(graph):
    result, _ = run(graph, "Explique la classe Greeter.")
    lines = result.context.splitlines()
    base = next(s.label for s in result.symbols if s.id == "pkg/base.py::Base")
    assert (
        f"- pkg.core.Greeter [S1] inherits from: pkg.base.Base [{base}] (pkg/core.py, line 24)"
        in lines
    )
    assert any(
        line.startswith("- pkg.core.Greeter [S1] is the base class of: app.App [S")
        for line in lines
    )
    assert "external module os" not in result.context  # the file was not asked about
    module, _ = run(graph, "Comment fonctionne pkg/core.py ?")
    imports = next(line for line in module.context.splitlines() if " imports: " in line)
    assert imports.startswith("- pkg/core.py imports: ")
    assert "external module os (pkg/core.py, line 4)" in imports


def test_vector_hits_are_expanded_through_the_graph(graph):
    result, _ = run(graph, "turn a text into a url slug", hits=[("pkg/utils.py::slugify", 0.8)])
    assert ids(result)[0] == "pkg/utils.py::slugify"
    greet = next(s for s in result.symbols if s.id == "pkg/core.py::greet")
    assert greet.score == pytest.approx(0.76)  # direct caller of a 0.8 hit: 0.8 * 0.95
    assert greet.reasons == ["CALLS <- pkg/utils.py::slugify"]


def test_expansion_depth(graph):
    one, _ = run(graph, "slugify", retrieval_depth=1)
    two, _ = run(graph, "slugify", retrieval_depth=2)
    assert "pkg/core.py::greet" in ids(one) and "app.py::App.main" not in ids(one)
    assert "app.py::App.main" in ids(two)  # calls greet, which calls slugify


def test_weak_vector_hits_give_no_result(graph):
    result, _ = run(graph, "Which database stores the users?", hits=[("pkg/core.py::greet", 0.55)])
    assert result.empty and result.symbols == [] and result.context == ""


def test_duplicates_are_merged(graph):
    result, _ = run(graph, "greet", hits=[("pkg/core.py::greet", 0.9), ("pkg/core.py::greet", 0.7)])
    assert len(ids(result)) == len(set(ids(result)))
    greet = result.symbols[0]
    assert greet.score == 2.0
    assert greet.reasons == ["name in question", "semantic match 0.90", "semantic match 0.70"]


def test_file_question_brings_its_symbols_and_imports(graph):
    result, _ = run(graph, "Comment fonctionne le module pkg/core.py ?")
    assert result.files == ["pkg/core.py"]
    assert ids(result)[:3] == ["pkg/core.py::greet", "pkg/core.py::format_message",
                               "pkg/core.py::Greeter"]  # fmt: skip
    imports = {r.target_id for r in result.relations if r.type == "IMPORTS"}
    assert {"module:os", "module:json", "pkg/utils.py"} <= imports


def test_number_of_symbols_is_capped(graph):
    result, _ = run(graph, "Comment fonctionne pkg/core.py ?", retrieval_max_symbols=2)
    assert [s.label for s in result.symbols] == ["S1", "S2"]


def test_context_respects_the_character_limit(graph):
    result, _ = run(graph, "Explique la classe Greeter.", qa_max_context_chars=600)
    assert len(result.context) <= 600
    assert result.truncated
    # Only what was rendered is returned as a source.
    assert result.symbols and all(f"[{s.label}]" in result.context for s in result.symbols)


def test_nested_code_is_not_repeated(graph):
    result, _ = run(graph, "Explique la classe Greeter.", hits=[("pkg/core.py::Greeter.run", 0.9)])
    assert result.symbols[0].id == "pkg/core.py::Greeter"
    assert result.context.count("def run(self):") == 1
    assert "code: shown in [S1]" in result.context


def test_repositories_are_isolated(graph):
    other = graph_from({"other.py": "def greet():\n    return 'hi'\n"})
    store = FakeStore({REPO: graph, "acme/other": other})
    result = retrieve(store, "acme/other", "Où est greet ?", embedder=FakeEmbedder(),
                      settings=Settings(_env_file=None))  # fmt: skip
    assert ids(result) == ["other.py::greet"]
    assert set(store.repo_ids) == {"acme/other"}
    assert result.known_paths == {"other.py"}


def test_is_test_path():
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/test_core.py")
    assert is_test_path("src/conftest.py") and is_test_path("pkg/core_test.py")
    assert not is_test_path("src/requests/sessions.py") and not is_test_path("pkg/contest.py")


def test_test_code_ranks_below_library_code():
    graph = graph_from({
        "lib.py": "def follow(url):\n    return url\n",
        "tests/test_lib.py": "def test_follow():\n    pass\n",
    })  # fmt: skip
    hits = [("tests/test_lib.py::test_follow", 0.8), ("lib.py::follow", 0.75)]
    result, _ = run(graph, "How are redirects followed?", hits=hits)
    assert ids(result) == ["lib.py::follow", "tests/test_lib.py::test_follow"]
    about_tests, _ = run(graph, "Which test covers redirects?", hits=hits)
    assert ids(about_tests)[0] == "tests/test_lib.py::test_follow"


def test_relations_survive_a_tight_budget(graph):
    result, _ = run(graph, "Quelles fonctions appellent greet ?", qa_max_context_chars=1200)
    assert len(result.context) <= 1200
    assert "- pkg.core.greet [S1] is called by: " in result.context


def test_code_is_cut_rather_than_dropped():
    body = "".join(f"    step_{i}()\n" for i in range(200))
    graph = graph_from({"big.py": f"def big():\n{body}"})
    result, _ = run(graph, "big", qa_max_context_chars=1000)
    assert "   1 | def big():" in result.context and result.context.rstrip().endswith("...")
    assert len(result.context) <= 1000
