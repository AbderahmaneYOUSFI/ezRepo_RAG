"""Local embeddings with sentence-transformers/all-MiniLM-L6-v2 (CPU, 384 dimensions).

The model is downloaded from Hugging Face on first use, then loaded from the local cache
without any network call. It is loaded once per process (see `get_embedder`).
"""

import hashlib
import threading
from collections.abc import Iterator

from langchain_core.embeddings import Embeddings

from ez_repo.config import get_settings
from ez_repo.parser import CodeGraph, Symbol

EMBEDDING_DIMENSIONS = 384
# MiniLM reads at most 256 word pieces: the header comes first, the code fills what is left.
MAX_DOCSTRING_CHARS = 500
MAX_CODE_CHARS = 800


class EmbeddingError(Exception):
    """The embedding model could not be loaded or used."""


class Embedder(Embeddings):
    """A sentence-transformers model on CPU, usable directly or as a LangChain `Embeddings`."""

    def __init__(self, model_name: str, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # heavy import (torch): deferred

        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except OSError:  # not in the local cache yet: download it once
            try:
                self.model = SentenceTransformer(model_name, device="cpu")
            except OSError as exc:
                raise EmbeddingError(
                    f"Cannot load embedding model {model_name!r} (first use needs network "
                    f"access to Hugging Face): {exc}"
                ) from exc
        self.dimensions = self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


_embedders: dict[str, Embedder] = {}
_lock = threading.Lock()


def get_embedder(model_name: str | None = None) -> Embedder:
    """Return the process-wide embedder for `model_name` (default: settings), loading it once."""
    settings = get_settings()
    model_name = model_name or settings.embedding_model
    with _lock:
        if model_name not in _embedders:
            _embedders[model_name] = Embedder(model_name, settings.embedding_batch_size)
        return _embedders[model_name]


def symbol_text(symbol: Symbol, module: str = "") -> str:
    """Text indexed for a symbol: identity first, then docstring and a code excerpt."""
    qualified = f"{module}.{symbol.qualified_name}" if module else symbol.qualified_name
    lines = [
        f"{symbol.kind} {qualified}",
        f"file: {symbol.path}",
        f"signature: {symbol.signature}",
    ]
    if symbol.decorators:
        lines.append(f"decorators: {', '.join(symbol.decorators)}")
    if symbol.docstring:
        lines.append(f"docstring: {_truncate(symbol.docstring, MAX_DOCSTRING_CHARS)}")
    lines.append(f"code:\n{_truncate(symbol.source, MAX_CODE_CHARS)}")
    return "\n".join(lines)


def symbol_documents(graph: CodeGraph) -> Iterator[tuple[Symbol, str]]:
    for f in graph.files:
        for symbol in f.symbols:
            yield symbol, symbol_text(symbol, f.module)


def text_hash(text: str, model_name: str) -> str:
    """Identifies an embedding: same text and same model give the same vector."""
    return hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."
