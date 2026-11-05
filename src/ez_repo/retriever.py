"""Hybrid retrieval for one repository: exact names, vector search, then graph expansion.

1. Exact: identifiers and file names written in the question (`greet`, `Greeter.run`,
   `pkg/core.py`, `pkg.core`) are matched against the repository's symbols and files.
2. Semantic: the question's embedding is searched in the `symbol_embeddings` index.
3. Graph: CALLS / INHERITS / DEFINES neighbours of the best symbols (1 or 2 hops) and the
   IMPORTS of matched files are added.

Every store call is scoped to one repo_id. Symbols are deduplicated, ranked and capped,
then rendered as a labelled context ([S1], [S2], ...) under a character budget.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from ez_repo.config import Settings, get_settings
from ez_repo.embeddings import Embedder, get_embedder
from ez_repo.graph import GraphStore
from ez_repo.parser import Relation

EXPANSION_TYPES = ("CALLS", "INHERITS", "DEFINES", "IMPORTS")
EXACT_SCORE = 2.0  # names written in the question beat any cosine score (<= 1)
FILE_SCORE = 1.5
# Neighbour score = seed score * factor, per hop. A direct caller/callee competes with the
# next vector hits; the enclosing class or its members matter less.
NEIGHBOUR_FACTORS = {"CALLS": 0.95, "INHERITS": 0.9, "DEFINES": 0.6}
TEST_PENALTY = 0.9  # vector hits in test files, unless the question is about tests
MAX_SEEDS = 3  # symbols whose neighbours are explored
MAX_FILES = 2
MAX_SOURCE_CHARS = 1500  # per symbol
MIN_CODE_CHARS = 300  # below this, a symbol that does not fit is listed without code
MAX_DOCSTRING_CHARS = 300
MAX_RELATIONS = 25
# (active, passive) wording: small models misread the direction of "A CALLS B" less often
# when relations are sentences told from the point of view of the most relevant symbol.
_VERBS = {
    "CALLS": ("calls", "is called by"),
    "INHERITS": ("inherits from", "is the base class of"),
    "DEFINES": ("defines", "is defined in"),
    "IMPORTS": ("imports", "is imported by"),
    "CONTAINS": ("contains", "is contained in"),
}

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
# Question words never looked up as symbol names (such symbols are still reachable through
# vector search).
_STOPWORDS = frozenset(
    """
    the and are what where which who why how does did do this that these those with from for
    into about there here its explain describe show find list tell work works working use used
    uses using function functions method methods class classes file files module modules code
    call calls called caller callers repository repo project defined define defines implemented
    import imports inherit inherits return returns value values main part
    les des une est que qui quoi quel quelle quels quelles comment pourquoi dans pour par sur
    avec cette ces son ses leur fonction fonctions méthode méthodes classe classes fichier
    fichiers module modules appelle appellent appel appels explique expliquer décris trouve
    trouver fonctionne utilise utilisent définie défini définit importe hérite renvoie retourne
    """.split()
)


@dataclass
class RetrievedSymbol:
    id: str
    kind: str
    qualified_name: str
    module: str
    path: str
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    source: str
    score: float
    reasons: list[str] = field(default_factory=list)
    label: str = ""  # "S1", "S2"... once ranked

    @classmethod
    def from_record(cls, record: dict[str, Any], score: float, reason: str) -> "RetrievedSymbol":
        return cls(
            id=record["id"],
            kind=record["kind"],
            qualified_name=record["qualified_name"],
            module=record.get("module") or "",
            path=record["path"],
            start_line=record["start_line"],
            end_line=record["end_line"],
            signature=record["signature"],
            docstring=record.get("docstring"),
            source=record.get("source") or "",
            score=score,
            reasons=[reason],
        )

    @property
    def full_name(self) -> str:
        return f"{self.module}.{self.qualified_name}" if self.module else self.qualified_name

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def contains(self, other: "RetrievedSymbol") -> bool:
        return (
            self.path == other.path
            and self.start_line <= other.start_line
            and other.end_line <= self.end_line
        )


@dataclass
class RetrievalResult:
    repo_id: str
    question: str
    symbols: list[RetrievedSymbol]  # ranked, labelled; only those rendered in `context`
    relations: list[Relation]
    files: list[str]  # files named in the question
    known_paths: set[str]  # every file of the repository, to verify citations
    context: str = ""
    truncated: bool = False

    @property
    def empty(self) -> bool:
        return not self.symbols and not self.relations

    def by_label(self) -> dict[str, RetrievedSymbol]:
        return {s.label: s for s in self.symbols}


def question_terms(question: str) -> tuple[set[str], set[str]]:
    """Return (identifiers, file paths) written in the question."""
    names: set[str] = set()
    paths: set[str] = set()
    for raw in re.findall(r"[\w./\\-]+", question):
        token = raw.replace("\\", "/").strip(".-/")
        if token.lower().endswith(".py"):
            paths.add(token)
        elif _IDENTIFIER.fullmatch(token) and len(token) >= 3 and token.lower() not in _STOPWORDS:
            names.add(token)
    return names, paths


def match_files(files: list[dict[str, Any]], names: set[str], paths: set[str]) -> list[str]:
    """Files named in the question by path ("pkg/core.py", "core.py") or dotted module
    ("pkg.core"; a bare word such as "app" is too ambiguous)."""
    lowered_paths = {p.lower() for p in paths}
    modules = {n for n in names if "." in n}
    matched = []
    for f in files:
        path = f["path"].lower()
        if any(path == p or path.endswith("/" + p) for p in lowered_paths) or (
            f.get("module") in modules
        ):
            matched.append(f["path"])
    return matched


def retrieve(
    store: GraphStore,
    repo_id: str,
    question: str,
    *,
    embedder: Embedder | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    settings = settings or get_settings()
    max_symbols = settings.retrieval_max_symbols
    found: dict[str, RetrievedSymbol] = {}

    about_tests = "test" in question.lower()

    def add(record: dict[str, Any], score: float, reason: str, penalize: bool = True) -> None:
        if penalize and not about_tests and is_test_path(record["path"]):
            score *= TEST_PENALTY
        item = found.get(record["id"])
        if item is None:
            found[record["id"]] = RetrievedSymbol.from_record(record, score, reason)
            return
        item.score = max(item.score, score)
        if reason not in item.reasons:
            item.reasons.append(reason)

    # 1. Exact names and files.
    files = store.get_files(repo_id)
    names, paths = question_terms(question)
    if names:
        for record in store.find_symbols(repo_id, names, limit=max_symbols):
            add(record, EXACT_SCORE, "name in question", penalize=False)
    matched_files = match_files(files, names, paths)[:MAX_FILES]
    for path in matched_files:
        for record in store.get_symbols(repo_id, path=path, with_source=True):
            if record["parent_id"] is None:
                add(record, FILE_SCORE, f"defined in {path}", penalize=False)

    # 2. Semantic search.
    embedder = embedder or get_embedder()
    vector = embedder.embed_query(question)
    for hit in store.vector_search(repo_id, vector, k=settings.retrieval_top_k):
        if hit["score"] >= settings.retrieval_min_score:
            add(hit, hit["score"], f"semantic match {hit['score']:.2f}")

    # 3. Graph expansion around the best symbols and the matched files.
    seeds = _ranked(found.values())[:MAX_SEEDS]
    relations = _expand(store, repo_id, seeds, matched_files, found, add, settings)

    ranked = _ranked(found.values())[:max_symbols]
    for i, symbol in enumerate(ranked, start=1):
        symbol.label = f"S{i}"
    kept = {s.id for s in ranked} | set(matched_files)
    relations = [r for r in relations if _relevant(r, kept)][:MAX_RELATIONS]

    result = RetrievalResult(
        repo_id=repo_id,
        question=question,
        symbols=ranked,
        relations=relations,
        files=matched_files,
        known_paths={f["path"] for f in files},
    )
    result.context, result.symbols, result.truncated = render_context(
        ranked, relations, matched_files, settings.qa_max_context_chars
    )
    return result


def _expand(store, repo_id, seeds, matched_files, found, add, settings) -> list[Relation]:
    relations: dict[tuple[str, str, str], Relation] = {}
    scores = {s.id: s.score for s in seeds} | {path: FILE_SCORE for path in matched_files}
    frontier = list(scores)
    for _ in range(max(1, min(settings.retrieval_depth, 2))):
        if not frontier:
            break
        current = set(frontier)
        neighbours: dict[str, tuple[float, str]] = {}
        for rel in store.get_relations(repo_id, EXPANSION_TYPES, node_ids=frontier):
            relations.setdefault((rel.type, rel.source_id, rel.target_id), rel)
            for node, other in ((rel.source_id, rel.target_id), (rel.target_id, rel.source_id)):
                if node not in current or other in found or "::" not in other:
                    continue  # only symbols ("<path>::<name>") become context items
                if rel.type == "DEFINES" and "::" not in node:
                    continue  # a matched file's own definitions are already in
                score = scores[node] * NEIGHBOUR_FACTORS.get(rel.type, 0.5)
                if score > neighbours.get(other, (0.0, ""))[0]:
                    direction = "->" if node == rel.source_id else "<-"
                    neighbours[other] = (score, f"{rel.type} {direction} {node}")
        best = sorted(neighbours.items(), key=lambda kv: (-kv[1][0], kv[0]))
        best = best[: settings.retrieval_max_symbols]
        frontier = []
        for symbol_id, (score, reason) in best:
            record = store.get_symbol(repo_id, symbol_id)
            if record is not None:
                add(record, score, reason)
                scores[symbol_id] = score
                frontier.append(symbol_id)
    return list(relations.values())


def is_test_path(path: str) -> bool:
    *dirs, name = path.lower().split("/")
    return (
        bool({"test", "tests", "testing"} & set(dirs))
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _ranked(symbols) -> list[RetrievedSymbol]:
    return sorted(symbols, key=lambda s: (-s.score, s.path, s.start_line))


def _relevant(rel: Relation, kept: set[str]) -> bool:
    if rel.source_id not in kept:
        return False
    if rel.type == "DEFINES" and "::" not in rel.source_id:
        return False  # "file DEFINES symbol" repeats the symbol's location
    return rel.target_id in kept or rel.type == "IMPORTS"


def render_context(
    symbols: list[RetrievedSymbol],
    relations: list[Relation],
    files: list[str],
    max_chars: int,
) -> tuple[str, list[RetrievedSymbol], bool]:
    """Render symbols (by rank) and relations within `max_chars`.

    Returns (context, symbols actually rendered, truncated). A symbol nested in one already
    shown (a method in its class) points to it instead of repeating the code. Relations are
    cheap and carry the graph: up to a quarter of the budget is kept for them.
    """
    blocks: list[str] = []
    used = 0
    truncated = False
    planned = "\n".join(["Relations:", *_relation_lines(relations, symbols, files)])
    symbol_budget = max_chars - min(len(planned) + 2, max_chars // 4) if relations else max_chars

    def fits(block: str, limit: int = symbol_budget) -> bool:
        return used + len(block) + 2 <= limit

    if files:
        header = "Files named in the question: " + ", ".join(files)
        if fits(header):
            blocks.append(header)
            used += len(header) + 2

    shown: list[RetrievedSymbol] = []
    with_code: list[RetrievedSymbol] = []
    for symbol in symbols:
        head = [
            f"[{symbol.label}] {symbol.kind} {symbol.full_name} ({symbol.location})",
            f"signature: {symbol.signature}",
        ]
        body = []
        if symbol.docstring:
            body.append(f"docstring: {_truncate(symbol.docstring, MAX_DOCSTRING_CHARS)}")
        container = next((c for c in with_code if c.contains(symbol)), None)
        full_code = False
        if container is not None:
            body.append(f"code: shown in [{container.label}]")
        elif symbol.kind == "class" and len(symbol.source) > MAX_SOURCE_CHARS:
            body.append(_outline(symbol))  # a big class: its methods, not its first lines
        else:
            body.append(_numbered(symbol))
            full_code = True
        block = "\n".join(head + body)
        if not fits(block):
            truncated = True
            # The start of the code if some room is left, else at least where it is.
            room = symbol_budget - used - len("\n".join(head + body[:-1])) - 4
            if full_code and room >= MIN_CODE_CHARS:
                block = "\n".join(head + body[:-1] + [_numbered(symbol, room)])
            else:
                block = "\n".join(head)
            if not fits(block):
                break
        elif full_code:
            with_code.append(symbol)
        blocks.append(block)
        used += len(block) + 2
        shown.append(symbol)

    lines = []
    for line in _relation_lines(relations, shown, files):
        if not fits("\n".join(["Relations:", *lines, line]), max_chars):
            truncated = True
            break
        lines.append(line)
    if lines:
        blocks.append("\n".join(["Relations:", *lines]))

    truncated = truncated or len(shown) < len(symbols)
    return "\n\n".join(blocks), shown, truncated


def _relation_lines(
    relations: list[Relation], shown: list[RetrievedSymbol], files: list[str]
) -> list[str]:
    """One line per (subject, verb), told from the best-ranked endpoint, e.g.
    "greet [S1] is called by: App.main [S2] (app.py, line 9), ..."."""
    names = {s.id: f"{s.full_name} [{s.label}]" for s in shown}
    rank = {s.id: i for i, s in enumerate(shown)} | {path: -1 for path in files}
    groups: dict[tuple[str, str], list[str]] = {}
    for rel in relations:
        where = f" ({rel.source_id.split('::')[0]}, line {rel.line})" if rel.line else ""
        active, passive = _VERBS[rel.type]
        if rank.get(rel.target_id, len(rank)) < rank.get(rel.source_id, len(rank)):
            subject, verb, other = rel.target_id, passive, rel.source_id
        else:
            subject, verb, other = rel.source_id, active, rel.target_id
        groups.setdefault((subject, verb), []).append(_node(other, names) + where)
    ordered = sorted(groups.items(), key=lambda item: rank.get(item[0][0], len(rank)))
    return [
        f"- {_node(subject, names)} {verb}: {', '.join(others)}"
        for (subject, verb), others in ordered
    ]


def _outline(symbol: RetrievedSymbol) -> str:
    """Class and def lines of a big class, with their line numbers."""
    lines = [
        f"{symbol.start_line + i:>4} | {line}"
        for i, line in enumerate(symbol.source.splitlines())
        if line.lstrip().startswith(("def ", "async def ", "class "))
    ]
    return "outline (code too long):\n" + _truncate("\n".join(lines), MAX_SOURCE_CHARS)


def _numbered(symbol: RetrievedSymbol, limit: int = MAX_SOURCE_CHARS) -> str:
    """Code with line numbers so the model can cite exact lines, within `limit` chars."""
    out, size = ["code:"], len("code:")
    for i, line in enumerate(symbol.source.splitlines()):
        numbered = f"{symbol.start_line + i:>4} | {line}"
        if size + len(numbered) + 1 > limit - 4:
            out.append("...")
            break
        out.append(numbered)
        size += len(numbered) + 1
    return "\n".join(out)


def _node(node_id: str, names: dict[str, str]) -> str:
    if node_id in names:
        return names[node_id]
    if node_id.startswith("module:"):
        return f"external module {node_id.removeprefix('module:')}"
    return node_id  # a file path, or a symbol not shown in the context


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "\n..."
