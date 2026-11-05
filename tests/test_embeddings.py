from dataclasses import replace
from pathlib import Path

import pytest

from ez_repo import embeddings
from ez_repo.embeddings import (
    EMBEDDING_DIMENSIONS,
    MAX_CODE_CHARS,
    get_embedder,
    symbol_documents,
    symbol_text,
    text_hash,
)
from ez_repo.ingest import collect_python_files
from ez_repo.parser import parse_repository

DEMO_REPO = Path(__file__).parent / "fixtures" / "demo_repo"


@pytest.fixture(scope="module")
def graph():
    return parse_repository(collect_python_files(DEMO_REPO, 10**6, 100, 10**7).files)


@pytest.fixture(scope="module")
def symbols(graph):
    return {s.id: s for s in graph.symbols}


def test_symbol_text_contains_identity_docstring_and_code(symbols):
    text = symbol_text(symbols["pkg/core.py::greet"], "pkg.core")
    lines = text.splitlines()
    assert lines[0] == "function pkg.core.greet"
    assert "file: pkg/core.py" in lines
    assert "signature: def greet(name: str) -> str" in lines
    assert "docstring: Return a greeting." in lines
    assert "return format_message(make_slug(name))" in text


def test_symbol_text_without_docstring(symbols):
    text = symbol_text(symbols["pkg/core.py::Greeter.run"], "pkg.core")
    assert text.startswith("method pkg.core.Greeter.run\n")
    assert "docstring:" not in text


def test_symbol_text_lists_decorators(symbols):
    assert "decorators: property" in symbol_text(symbols["pkg/core.py::Greeter.label"])


def test_long_code_is_truncated(symbols):
    text = symbol_text(replace(symbols["pkg/core.py::greet"], source="x = 1\n" * 1000))
    assert len(text) < MAX_CODE_CHARS + 300
    assert text.endswith(" ...")


def test_symbol_documents_cover_every_symbol(graph):
    docs = list(symbol_documents(graph))
    assert [s.id for s, _ in docs] == [s.id for s in graph.symbols]
    assert all(text.startswith(f"{s.kind} ") for s, text in docs)


def test_text_hash_depends_on_text_and_model():
    assert text_hash("a", "m1") == text_hash("a", "m1")
    assert text_hash("a", "m1") != text_hash("b", "m1")
    assert text_hash("a", "m1") != text_hash("a", "m2")


def test_model_is_loaded_once_per_process(monkeypatch):
    loads = []

    class FakeEmbedder:
        def __init__(self, model_name, batch_size):
            loads.append((model_name, batch_size))

    monkeypatch.setattr(embeddings, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embeddings, "_embedders", {})
    first, second = get_embedder("fake/model"), get_embedder("fake/model")
    assert first is second
    assert loads == [("fake/model", 32)]


@pytest.mark.integration
def test_real_model_on_cpu():
    """Downloads the model on first run (~90 MB), then uses the local cache."""
    embedder = get_embedder()
    assert embedder is get_embedder()
    assert embedder.dimensions == EMBEDDING_DIMENSIONS
    assert embedder.model.device.type == "cpu"

    vectors = embedder.embed_documents(["def add(a, b): return a + b", "class User: pass"])
    assert len(vectors) == 2 and all(len(v) == EMBEDDING_DIMENSIONS for v in vectors)
    assert sum(x * x for x in vectors[0]) == pytest.approx(1.0, abs=1e-4)  # normalized
    # Same text alone or in a padded batch: equal up to float32 noise.
    assert embedder.embed_query("def add(a, b): return a + b") == pytest.approx(
        vectors[0], abs=1e-5
    )
