"""Question answering: build a prompt from retrieved context and query Ollama,
returning the answer with source file references.

Sources come from the retrieval metadata (Neo4j), not from the generated text: the model
cites context labels ([S1], [S2]...), Python maps them back to verified paths and lines and
drops labels that do not exist.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from ollama import ResponseError

from ez_repo.config import Settings, get_settings
from ez_repo.embeddings import Embedder
from ez_repo.graph import GraphStore
from ez_repo.retriever import RetrievalResult, RetrievedSymbol, retrieve

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are ez-repo, an assistant that explains the source code of the repository {repo_id}.

Rules:
- Answer only from the code context provided by the user message. It is the only \
information you have about this repository.
- Never invent files, functions, classes, line numbers or dependencies that are not in \
the context.
- If the context does not contain enough information to answer, say so clearly and say \
what is missing. Do not guess.
- Explain what the relevant functions and classes do, their role, and how they relate \
(calls, inheritance, imports).
- Cite the context labels right after the statements they support, with the file path and \
line numbers given in the context. Example: "`greet` builds the message [S1] \
(pkg/core.py:12-17)."
- Be concise and precise: under about 200 words. Do not copy the code; quote at most a \
short line when it helps.
- Answer in {language}."""

HUMAN_PROMPT = """\
Code context:

{context}

Question: {question}"""

NO_CONTEXT_ANSWER = {
    "fr": "Je n'ai trouvé aucun code pertinent dans ce repository pour répondre à cette "
    "question. L'information est insuffisante.",
    "en": "I found no relevant code in this repository to answer this question. "
    "There is not enough information.",
}
LANGUAGE_NAMES = {"fr": "French", "en": "English"}

_FRENCH_WORDS = frozenset(
    """le la les un une des du de est sont que qui quoi quel quelle quels quelles où comment
    pourquoi dans pour par sur avec ce cette ces se trouve fait font explique décris fonction
    fonctions classe fichier appellent appelle utilise module ne pas il elle et ou""".split()
)
_ENGLISH_WORDS = frozenset(
    """the a an is are what which who where how why does do in for on with this that these
    find explain describe function functions class file call calls use uses module not it
    and or of to""".split()
)
_CITATION = re.compile(r"\[([^\[\]]*\bS\d+\b[^\[\]]*)\]")
_PATH = re.compile(r"[\w./-]+\.py\b")


class QAError(Exception):
    """Base error for question answering."""


class RepositoryNotIndexed(QAError):
    pass


class RepositoryNotReady(RepositoryNotIndexed):
    """The repository exists but is being indexed, or its last indexing failed."""


class LLMError(QAError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


@dataclass(frozen=True)
class Source:
    label: str
    symbol_id: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int
    cited: bool  # the answer refers to it

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass
class Answer:
    repo_id: str
    question: str
    answer: str
    language: str
    sources: list[Source] = field(default_factory=list)  # context sent to the model
    invalid_citations: list[str] = field(default_factory=list)  # labels removed from answer
    unknown_paths: list[str] = field(default_factory=list)  # .py paths not in the repository
    context_truncated: bool = False
    incomplete: bool = False  # the answer hit OLLAMA_NUM_PREDICT

    @property
    def cited_sources(self) -> list[Source]:
        return [s for s in self.sources if s.cited]


def detect_language(text: str) -> str:
    """Return "fr" or "en", from common words and accents."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    french = sum(w in _FRENCH_WORDS for w in words) + len(re.findall(r"[àâçéèêëîïôûùœ]", lowered))
    english = sum(w in _ENGLISH_WORDS for w in words)
    return "fr" if french > english else "en"


def get_llm(settings: Settings | None = None) -> ChatOllama:
    """The configured Ollama chat model, created once per process."""
    s = settings or get_settings()
    return _llm(
        s.ollama_model,
        s.ollama_base_url,
        s.ollama_temperature,
        s.ollama_num_ctx,
        s.ollama_num_predict,
        s.ollama_timeout,
    )


@lru_cache(maxsize=1)  # a single LLM: changing settings replaces it
def _llm(model, base_url, temperature, num_ctx, num_predict, timeout) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        repeat_penalty=1.15,  # small models at low temperature tend to loop on lists
        client_kwargs={"timeout": timeout},
    )


def build_chain(llm: BaseChatModel | Runnable) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm


def ask_repository(
    store: GraphStore,
    repo_id: str,
    question: str,
    *,
    llm: BaseChatModel | Runnable | None = None,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> Answer:
    """Answer `question` about the indexed repository `repo_id`, with verified sources."""
    settings = settings or get_settings()
    question = question.strip()
    if not question:
        raise ValueError("The question is empty.")
    repo = store.get_repository(repo_id)
    if repo is None:
        raise RepositoryNotIndexed(f"Repository {repo_id} is not indexed.")
    if repo["status"] == "failed":
        raise RepositoryNotReady(f"The last indexing of {repo_id} failed: import it again.")
    if repo["status"] != "ready":
        raise RepositoryNotReady(f"Repository {repo_id} is still being indexed.")

    language = detect_language(question)
    retrieval = retrieve(store, repo_id, question, embedder=embedder, settings=settings)
    if retrieval.empty:
        return Answer(repo_id, question, NO_CONTEXT_ANSWER[language], language)

    chain = build_chain(llm or get_llm(settings))
    message = _invoke(
        chain,
        {
            "repo_id": repo_id,
            "language": LANGUAGE_NAMES[language],
            "context": retrieval.context,
            "question": question,
        },
        settings,
    )
    text = collapse_repeated_lines(str(message.content))
    text, sources, invalid = verify_citations(text, retrieval)
    unknown = unknown_paths(text, retrieval.known_paths)
    if invalid or unknown:
        log.warning("Unverified references in answer: labels=%s paths=%s", invalid, unknown)
    return Answer(
        repo_id=repo_id,
        question=question,
        answer=text,
        language=language,
        sources=sources,
        invalid_citations=invalid,
        unknown_paths=unknown,
        context_truncated=retrieval.truncated,
        incomplete=message.response_metadata.get("done_reason") == "length",
    )


def verify_citations(text: str, retrieval: RetrievalResult) -> tuple[str, list[Source], list[str]]:
    """Keep citation labels that exist in the context; return (text, sources, invalid labels)."""
    known = retrieval.by_label()
    cited: set[str] = set()
    invalid: list[str] = []

    def keep_known(match: re.Match) -> str:
        labels = re.findall(r"S\d+", match.group(1))
        valid = [label for label in labels if label in known]
        cited.update(valid)
        invalid.extend(label for label in labels if label not in known and label not in invalid)
        return f"[{', '.join(valid)}]" if valid else ""

    text = _CITATION.sub(keep_known, text).strip()
    # Small models often cite by name or "path:line" instead of labels.
    cited.update(_cited_by_name(text, retrieval.symbols))
    cited.update(_cited_by_line(text, retrieval.symbols))
    sources = [
        Source(
            label=s.label,
            symbol_id=s.id,
            kind=s.kind,
            name=s.full_name,
            path=s.path,
            start_line=s.start_line,
            end_line=s.end_line,
            cited=s.label in cited,
        )
        for s in retrieval.symbols
    ]
    return text, sources, invalid


def collapse_repeated_lines(text: str) -> str:
    """Drop a line identical to the previous one (degenerate generation loops).
    Code fences are kept: a closing fence may directly precede an opening one."""
    kept: list[str] = []
    for line in text.splitlines():
        key = line.strip()
        if key and not key.startswith("```") and kept and key == kept[-1].strip():
            continue
        kept.append(line)
    return "\n".join(kept)


def _cited_by_name(text: str, symbols: list[RetrievedSymbol]) -> set[str]:
    """Labels of symbols named in `text`: by qualified name ("Greeter.run"), or by short name
    ("resolve_redirects") when no other source shares it and it looks like code."""
    short_counts = Counter(s.qualified_name.rsplit(".", 1)[-1] for s in symbols)
    labels = set()
    for s in symbols:
        names = [s.qualified_name]
        short = s.qualified_name.rsplit(".", 1)[-1]
        code_like = "_" in short or short != short.lower() or f"`{short}`" in text
        if short != s.qualified_name and short_counts[short] == 1 and code_like:
            names.append(short)
        # "Greeter" inside "Greeter.run" names the method, not the class.
        if any(re.search(rf"(?<!\w){re.escape(n)}(?!\w|\.\w)", text) for n in names):
            labels.add(s.label)
    return labels


def _cited_by_line(text: str, symbols: list[RetrievedSymbol]) -> set[str]:
    """Labels of the narrowest symbol around each "path ... line N" in `text`; a bare file
    name ("sessions.py") counts when no other source file has it."""
    labels = set()
    paths = {s.path for s in symbols}
    basenames = Counter(p.rsplit("/", 1)[-1] for p in paths)
    for path in paths:
        base = path.rsplit("/", 1)[-1]
        spelled = (
            re.escape(path) if basenames[base] > 1 else rf"(?<![\w/.-])(?:\S*/)?{re.escape(base)}"
        )
        for match in re.finditer(rf"{spelled}\D{{0,20}}?(\d+)", text):
            line = int(match.group(1))
            around = [s for s in symbols if s.path == path and s.start_line <= line <= s.end_line]
            if around:
                labels.add(min(around, key=lambda s: s.end_line - s.start_line).label)
    return labels


def unknown_paths(text: str, known_paths: set[str]) -> list[str]:
    """Python file paths mentioned in `text` that do not exist in the repository."""
    unknown = []
    for path in dict.fromkeys(_PATH.findall(text)):
        path = path.lstrip("./")
        if not any(p == path or p.endswith("/" + path) for p in known_paths):
            unknown.append(path)
    return unknown


def _invoke(chain: Runnable, inputs: dict, settings: Settings) -> BaseMessage:
    url, model = settings.ollama_base_url, settings.ollama_model
    try:
        return chain.invoke(inputs)
    except httpx.TimeoutException as exc:
        raise LLMTimeout(
            f"Ollama did not answer within {settings.ollama_timeout:.0f}s "
            "(OLLAMA_TIMEOUT). Try a shorter question or a smaller QA_MAX_CONTEXT_CHARS."
        ) from exc
    except (httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailable(f"Cannot reach Ollama at {url}. Is Ollama running?") from exc
    except ResponseError as exc:
        if exc.status_code == 404:
            raise LLMUnavailable(f"Model {model} is not installed: ollama pull {model}") from exc
        raise LLMError(f"Ollama error: {exc.error}") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
