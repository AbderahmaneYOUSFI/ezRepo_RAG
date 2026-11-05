import httpx
import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from ollama import ResponseError

from ez_repo import qa
from ez_repo.config import Settings
from ez_repo.qa import (
    NO_CONTEXT_ANSWER,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    RepositoryNotIndexed,
    ask_repository,
    detect_language,
    get_llm,
)
from fakes import FakeEmbedder, FakeStore, demo_graph

REPO = "acme/demo"


@pytest.fixture(scope="module")
def graph():
    return demo_graph()


@pytest.fixture
def settings():
    # Explicit model: a global OLLAMA_MODEL may be set on the machine.
    return Settings(_env_file=None, ollama_model="qwen2.5-coder:3b")


def ask(graph, question, llm, settings, status="ready"):
    store = FakeStore({REPO: graph}, status=status)
    return ask_repository(store, REPO, question, llm=llm, embedder=FakeEmbedder(),
                          settings=settings)  # fmt: skip


def failing_llm(error):
    def fail(_prompt):
        raise error

    return RunnableLambda(fail)


@pytest.mark.parametrize(
    ("question", "language"),
    [
        ("Où se trouve la fonction greet ?", "fr"),
        ("Explique la classe Greeter.", "fr"),
        ("Quelles fonctions appellent slugify ?", "fr"),
        ("Where is the function greet?", "en"),
        ("Which functions call slugify?", "en"),
        ("Explain the Greeter class.", "en"),
    ],
)
def test_detect_language(question, language):
    assert detect_language(question) == language


def test_answer_has_verified_sources(graph, settings):
    llm = FakeListChatModel(responses=[
        "`greet` is defined in pkg/core.py [S1]. See also [S9] and lib/missing.py."
    ])  # fmt: skip
    answer = ask(graph, "Where is the function greet?", llm, settings)

    assert answer.language == "en"
    assert "[S9]" not in answer.answer and "[S1]" in answer.answer
    assert answer.invalid_citations == ["S9"]
    assert answer.unknown_paths == ["lib/missing.py"]
    first = answer.cited_sources[0]
    assert (first.label, first.symbol_id, first.location) == (
        "S1", "pkg/core.py::greet", "pkg/core.py:12-17"
    )  # fmt: skip
    # Every source comes from the retrieval metadata, cited or not.
    assert {s.label for s in answer.sources} >= {"S1", "S2"}


def test_citations_by_name_or_line_are_recognised(graph, settings):
    llm = FakeListChatModel(responses=[
        "It is called by App.main and by the code at pkg/core.py:45."
    ])  # fmt: skip
    answer = ask(graph, "Which functions call greet?", llm, settings)
    cited = {s.symbol_id for s in answer.cited_sources}
    assert {"app.py::App.main", "pkg/core.py::Greeter.run"} <= cited  # by name, by line 45
    assert "pkg/core.py::greet" not in cited  # never mentioned
    assert "pkg/core.py::Greeter" not in cited  # line 45 is in the class, but run is narrower


def test_prompt_contains_rules_context_and_language(graph, settings):
    seen = []

    def capture(prompt):
        seen.extend(prompt.to_messages())
        return AIMessage("ok")

    ask(graph, "Où se trouve la fonction greet ?", RunnableLambda(capture), settings)
    system, human = seen
    assert "Answer in French." in system.content
    assert "Never invent files" in system.content and REPO in system.content
    assert "not contain enough information" in system.content
    assert "[S1] function pkg.core.greet (pkg/core.py:12-17)" in human.content
    assert human.content.endswith("Question: Où se trouve la fonction greet ?")


def test_no_context_skips_the_llm(graph, settings):
    answer = ask(graph, "How are users stored in PostgreSQL?", failing_llm(AssertionError()),
                 settings)  # fmt: skip
    assert answer.answer == NO_CONTEXT_ANSWER["en"]
    assert answer.sources == []


def test_unknown_or_unready_repository(graph, settings):
    llm = FakeListChatModel(responses=["unused"])
    with pytest.raises(RepositoryNotIndexed, match="not indexed"):
        ask_repository(FakeStore({}), REPO, "greet?", llm=llm, embedder=FakeEmbedder())
    with pytest.raises(RepositoryNotIndexed, match="still being indexed"):
        ask(graph, "greet?", llm, settings, status="indexing")
    with pytest.raises(ValueError):
        ask(graph, "   ", llm, settings)


@pytest.mark.parametrize(
    ("error", "expected", "message"),
    [
        (httpx.ConnectError("refused"), LLMUnavailable, "Cannot reach Ollama"),
        (httpx.ReadTimeout("timed out"), LLMTimeout, "did not answer within 180s"),
        (ResponseError("model not found", 404), LLMUnavailable, "ollama pull qwen2.5-coder:3b"),
        (ResponseError("out of memory", 500), LLMError, "out of memory"),
    ],
)
def test_ollama_errors(graph, settings, error, expected, message):
    with pytest.raises(expected, match=message):
        ask(graph, "Where is greet?", failing_llm(error), settings)


def test_real_client_reports_unreachable_ollama(graph):
    settings = Settings(_env_file=None, ollama_base_url="http://127.0.0.1:9", ollama_timeout=5)
    qa._llm.cache_clear()
    try:
        with pytest.raises(LLMUnavailable, match="127.0.0.1:9"):
            ask(graph, "Where is greet?", None, settings)
    finally:
        qa._llm.cache_clear()


def test_single_configured_llm(settings):
    qa._llm.cache_clear()
    llm = get_llm(settings)
    assert get_llm(settings) is llm
    assert llm.model == "qwen2.5-coder:3b"
    assert llm.temperature == 0.1 and llm.num_ctx == 4096 and llm.num_predict == 512
    assert llm.repeat_penalty == 1.15 and llm.client_kwargs == {"timeout": 180.0}
    qa._llm.cache_clear()


def test_generation_loops_are_collapsed(graph, settings):
    loop = "Callers:\n" + "- `Greeter.run` (line 45)\n" * 30 + "```\nx\n```\n```\ny\n```"
    answer = ask(
        graph, "Which functions call greet?", FakeListChatModel(responses=[loop]), settings
    )
    assert answer.answer.count("Greeter.run") == 1
    assert answer.answer.count("```") == 4  # repeated but not consecutive: kept


def test_a_method_name_does_not_cite_its_class(graph, settings):
    llm = FakeListChatModel(responses=["It is called by pkg.core.Greeter.run."])
    answer = ask(graph, "Which functions call greet?", llm, settings)
    cited = {s.symbol_id for s in answer.cited_sources}
    assert "pkg/core.py::Greeter.run" in cited and "pkg/core.py::Greeter" not in cited


def test_answer_cut_by_the_token_limit_is_flagged(graph, settings):
    cut = AIMessage("`greet` is in pkg/core.py and", response_metadata={"done_reason": "length"})
    answer = ask(graph, "Where is greet?", RunnableLambda(lambda _: cut), settings)
    assert answer.incomplete
    complete = ask(graph, "Where is greet?", FakeListChatModel(responses=["ok"]), settings)
    assert not complete.incomplete


def test_short_names_and_bare_file_names_are_recognised(graph, settings):
    # Real answer shape from psf/requests: no labels, short method name, bare file name.
    llm = FakeListChatModel(
        responses=["format_message est appelée par greet dans core.py, à la ligne 17."]
    )
    answer = ask(graph, "Quelles fonctions appellent format_message ?", llm, settings)
    cited = {s.symbol_id for s in answer.cited_sources}
    assert "pkg/core.py::format_message" in cited  # short name with "_"
    assert "pkg/core.py::greet" in cited  # "core.py ... ligne 17" falls in greet (12-17)


def test_plain_words_are_not_taken_for_short_names(graph, settings):
    llm = FakeListChatModel(responses=["The label is shown when the code runs."])
    answer = ask(graph, "Explique la classe Greeter.", llm, settings)
    assert answer.cited_sources == []  # "label" / "run" without backticks: plain English
