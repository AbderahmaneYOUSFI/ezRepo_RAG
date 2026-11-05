"""Neo4j storage for the code graph and its embeddings.

Graph model. Every node but Repository carries `repo_id`; `id` is the parser's stable id
(file path, "<path>::<qualified name>", "module:<name>"), unique within a repository, and
(repo_id, id) is unique per label, so two repositories never share a node:

    (:Repository {id})-[:CONTAINS]->(:File)
    (:File)-[:DEFINES]->(:Symbol)          top-level classes and functions
    (:Symbol)-[:DEFINES]->(:Symbol)        methods and nested definitions
    (:File)-[:IMPORTS]->(:File|:ExternalModule)
    (:Symbol:Function)-[:CALLS]->(:Symbol:Function|:Symbol:Class)
    (:Symbol:Class)-[:INHERITS]->(:Symbol:Class)

Functions and methods share the Function label (`kind` tells them apart). Symbols carry an
`embedding` (384 floats) searched through the `symbol_embeddings` vector index.

Re-indexing is idempotent: nodes and relations are MERGEd and stamped with the id of the
current run; whatever the run did not write is deleted at the end. A symbol whose indexed
text did not change keeps its embedding.
"""

import logging
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any

from neo4j import Driver, GraphDatabase, RoutingControl
from neo4j.exceptions import AuthError, DriverError, Neo4jError, ServiceUnavailable

from ez_repo.config import Settings, get_settings
from ez_repo.embeddings import Embedder, get_embedder, symbol_documents, text_hash
from ez_repo.ingest import RepoRef
from ez_repo.parser import CodeGraph, Relation, Symbol

log = logging.getLogger(__name__)

VECTOR_INDEX = "symbol_embeddings"
NODE_LABELS = ("File", "Symbol", "ExternalModule")  # all keyed on (repo_id, id)
RELATION_TYPES = ("CONTAINS", "DEFINES", "IMPORTS", "CALLS", "INHERITS")
SYMBOL_LABELS = {"class": "Class", "function": "Function", "method": "Function"}

_SCHEMA = [
    "CREATE CONSTRAINT repository_id IF NOT EXISTS FOR (r:Repository) REQUIRE r.id IS UNIQUE",
    *(
        f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE (n.repo_id, n.id) IS UNIQUE"
        for label in NODE_LABELS
    ),
    *(
        f"CREATE INDEX {label.lower()}_repo IF NOT EXISTS FOR (n:{label}) ON (n.repo_id)"
        for label in NODE_LABELS
    ),
    "CREATE INDEX symbol_name IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
]
_SYMBOL_FIELDS = (
    "repo_id", "id", "kind", "name", "qualified_name", "module", "path", "start_line", "end_line",
    "signature", "docstring", "parent_id", "decorators", "bases",
)  # fmt: skip


class GraphStoreError(Exception):
    """Base error for Neo4j storage."""


class GraphUnavailable(GraphStoreError):
    pass


class GraphAuthError(GraphStoreError):
    pass


class GraphQueryError(GraphStoreError):
    pass


@dataclass
class IndexStats:
    repo_id: str
    files: int = 0
    symbols: int = 0
    relations: int = 0
    external_modules: int = 0
    removed: int = 0  # stale nodes and relations from the previous indexing
    embedded: int = 0
    embeddings_reused: int = 0


def repo_id_for(repo: RepoRef) -> str:
    """GitHub names are case-insensitive: "Owner/Repo" and "owner/repo" are one repository."""
    return repo.slug.lower()


@contextmanager
def _neo4j_errors(action: str) -> Iterator[None]:
    try:
        yield
    except AuthError as exc:
        raise GraphAuthError(
            "Neo4j rejected the credentials: check NEO4J_USER / NEO4J_PASSWORD in .env."
        ) from exc
    except ServiceUnavailable as exc:
        raise GraphUnavailable(
            f"Neo4j is unreachable while {action}: {exc}. Is it running (docker compose up -d)?"
        ) from exc
    except Neo4jError as exc:
        raise GraphQueryError(f"Neo4j error while {action}: {exc}") from exc
    except DriverError as exc:
        raise GraphStoreError(f"Neo4j driver error while {action}: {exc}") from exc


class GraphStore:
    def __init__(self, settings: Settings | None = None, driver: Driver | None = None):
        self.settings = settings or get_settings()
        self.batch_size = self.settings.neo4j_batch_size
        if driver is None:
            auth = (self.settings.neo4j_user, self.settings.neo4j_password)
            try:
                driver = GraphDatabase.driver(self.settings.neo4j_uri, auth=auth)
            except (ValueError, DriverError) as exc:
                raise GraphStoreError(f"Invalid Neo4j configuration: {exc}") from exc
        self._driver = driver

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def verify(self) -> None:
        with _neo4j_errors("connecting"):
            self._driver.verify_connectivity()

    # --- schema --------------------------------------------------------------------------

    def ensure_schema(self, dimensions: int) -> None:
        for query in _SCHEMA:
            self._run(query, "creating the schema")
        self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (s:Symbol) ON s.embedding "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)}, "
            f"`vector.similarity_function`: 'cosine'}}}}",
            "creating the vector index",
        )
        rows = self._read(
            "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
            "reading the vector index",
            name=VECTOR_INDEX,
        )
        existing = rows[0]["options"]["indexConfig"]["vector.dimensions"]
        if existing != dimensions:
            raise GraphStoreError(
                f"Vector index {VECTOR_INDEX} has {existing} dimensions but the embedding model "
                f"produces {dimensions}. Drop it (DROP INDEX {VECTOR_INDEX}) and re-index."
            )

    # --- writes --------------------------------------------------------------------------

    def begin(self, repo_id: str, run: str, url: str | None = None) -> None:
        self._run(
            "MERGE (r:Repository {id: $repo_id}) "
            "SET r.url = coalesce($url, r.url), r.status = 'indexing', r.run = $run, "
            "r.updated_at = datetime()",
            "registering the repository",
            repo_id=repo_id,
            run=run,
            url=url,
        )

    def write_graph(self, repo_id: str, graph: CodeGraph, run: str) -> None:
        """Upsert all nodes and relations of `graph`, stamped with `run`."""
        files = [
            {"id": f.path, "module": f.module, "has_errors": f.has_errors} for f in graph.files
        ]
        self._write_batches(
            "MATCH (r:Repository {id: $repo_id}) "
            "UNWIND $rows AS row "
            "MERGE (f:File {repo_id: $repo_id, id: row.id}) "
            "SET f.path = row.id, f.module = row.module, f.has_errors = row.has_errors, "
            "f.run = $run "
            "MERGE (r)-[c:CONTAINS]->(f) SET c.run = $run",
            files,
            "writing files",
            repo_id=repo_id,
            run=run,
        )

        by_label: dict[str, list[dict]] = defaultdict(list)
        for f in graph.files:
            for s in f.symbols:
                by_label[SYMBOL_LABELS[s.kind]].append(symbol_properties(s, f.module))
        for label, rows in by_label.items():
            other = next(lbl for lbl in set(SYMBOL_LABELS.values()) if lbl != label)
            self._write_batches(
                "UNWIND $rows AS row "
                "MERGE (s:Symbol {repo_id: $repo_id, id: row.id}) "
                f"SET s += row, s.run = $run REMOVE s:{other} SET s:{label}",
                rows,
                f"writing {label.lower()} nodes",
                repo_id=repo_id,
                run=run,
            )

        modules = [{"id": f"module:{name}", "name": name} for name in graph.external_modules]
        self._write_batches(
            "UNWIND $rows AS row "
            "MERGE (m:ExternalModule {repo_id: $repo_id, id: row.id}) "
            "SET m.name = row.name, m.run = $run",
            modules,
            "writing external modules",
            repo_id=repo_id,
            run=run,
        )

        for (rel_type, source_label, target_label), rows in relation_groups(graph).items():
            self._write_batches(
                "UNWIND $rows AS row "
                f"MATCH (a:{source_label} {{repo_id: $repo_id, id: row.source}}) "
                f"MATCH (b:{target_label} {{repo_id: $repo_id, id: row.target}}) "
                f"MERGE (a)-[rel:{rel_type}]->(b) "
                "SET rel.line = row.line, rel.run = $run",
                rows,
                f"writing {rel_type} relations",
                repo_id=repo_id,
                run=run,
            )

    def remove_stale(self, repo_id: str, run: str) -> int:
        """Delete nodes and relations of `repo_id` not written by `run`. Returns the count."""
        removed = 0
        for label in NODE_LABELS:
            removed += self._delete_in_batches(
                f"MATCH (n:{label} {{repo_id: $repo_id}}) WHERE coalesce(n.run, '') <> $run "
                "WITH n LIMIT $batch DETACH DELETE n RETURN count(*) AS deleted",
                f"removing stale {label} nodes",
                repo_id=repo_id,
                run=run,
            )
        for label in ("File", "Symbol"):  # sources of every relation but CONTAINS
            removed += self._delete_in_batches(
                f"MATCH (:{label} {{repo_id: $repo_id}})-[rel]->() "
                "WHERE coalesce(rel.run, '') <> $run "
                "WITH rel LIMIT $batch DELETE rel RETURN count(*) AS deleted",
                "removing stale relations",
                repo_id=repo_id,
                run=run,
            )
        return removed

    def finish(
        self,
        repo_id: str,
        stats: IndexStats,
        embedding_model: str,
        commit_sha: str | None = None,
    ) -> None:
        # commit_sha is overwritten even when unknown: an older SHA no longer matches the data.
        self._run(
            "MATCH (r:Repository {id: $repo_id}) "
            "SET r.status = 'ready', r.indexed_at = datetime(), r.file_count = $files, "
            "r.symbol_count = $symbols, r.relation_count = $relations, "
            "r.embedding_model = $model, r.commit_sha = $commit_sha, r.error = null",
            "finishing the indexing",
            repo_id=repo_id,
            files=stats.files,
            symbols=stats.symbols,
            relations=stats.relations,
            model=embedding_model,
            commit_sha=commit_sha,
        )

    def mark_failed(self, repo_id: str, error: str) -> None:
        self._run(
            "MATCH (r:Repository {id: $repo_id}) SET r.status = 'failed', r.error = $error",
            "marking the indexing as failed",
            repo_id=repo_id,
            error=error[:500],
        )

    def embedding_hashes(self, repo_id: str) -> dict[str, str]:
        rows = self._read(
            "MATCH (s:Symbol {repo_id: $repo_id}) WHERE s.embedding IS NOT NULL "
            "RETURN s.id AS id, s.embedding_hash AS hash",
            "reading embedding hashes",
            repo_id=repo_id,
        )
        return {row["id"]: row["hash"] for row in rows}

    def save_embeddings(self, repo_id: str, rows: Sequence[dict[str, Any]]) -> None:
        """`rows`: {"id": symbol id, "vector": list[float], "hash": text hash}."""
        self._write_batches(
            "UNWIND $rows AS row "
            "MATCH (s:Symbol {repo_id: $repo_id, id: row.id}) "
            "SET s.embedding = row.vector, s.embedding_hash = row.hash",
            rows,
            "writing embeddings",
            repo_id=repo_id,
        )

    def delete_repository(self, repo_id: str) -> int:
        """Delete a repository and everything it contains. Returns the number of nodes."""
        removed = 0
        for label in NODE_LABELS:
            removed += self._delete_in_batches(
                f"MATCH (n:{label} {{repo_id: $repo_id}}) "
                "WITH n LIMIT $batch DETACH DELETE n RETURN count(*) AS deleted",
                f"deleting {label} nodes",
                repo_id=repo_id,
            )
        rows = self._run(
            "MATCH (r:Repository {id: $repo_id}) DETACH DELETE r RETURN count(*) AS deleted",
            "deleting the repository",
            repo_id=repo_id,
        )
        return removed + rows[0]["deleted"]

    # --- reads ---------------------------------------------------------------------------

    def list_repositories(self) -> list[dict[str, Any]]:
        return self._read(
            f"MATCH (r:Repository) RETURN {_REPOSITORY_FIELDS} ORDER BY r.id",
            "listing repositories",
        )

    def get_repository(self, repo_id: str) -> dict[str, Any] | None:
        rows = self._read(
            f"MATCH (r:Repository {{id: $repo_id}}) RETURN {_REPOSITORY_FIELDS}",
            "reading the repository",
            repo_id=repo_id,
        )
        return rows[0] if rows else None

    def get_files(self, repo_id: str) -> list[dict[str, Any]]:
        return self._read(
            "MATCH (f:File {repo_id: $repo_id}) "
            "RETURN f.path AS path, f.module AS module, f.has_errors AS has_errors, "
            "COUNT { (f)-[:DEFINES]->() } AS symbols ORDER BY path",
            "reading files",
            repo_id=repo_id,
        )

    def get_symbols(
        self,
        repo_id: str,
        kind: str | None = None,
        path: str | None = None,
        with_source: bool = False,
    ) -> list[dict[str, Any]]:
        """Classes, functions and methods, filtered by kind and/or file path."""
        rows = self._read(
            "MATCH (s:Symbol {repo_id: $repo_id}) "
            "WHERE ($kind IS NULL OR s.kind = $kind) AND ($path IS NULL OR s.path = $path) "
            f"RETURN {_symbol_projection(with_source)} AS symbol "
            "ORDER BY s.path, s.start_line",
            "reading symbols",
            repo_id=repo_id,
            kind=kind,
            path=path,
        )
        return [row["symbol"] for row in rows]

    def get_symbol(self, repo_id: str, symbol_id: str) -> dict[str, Any] | None:
        rows = self._read(
            "MATCH (s:Symbol {repo_id: $repo_id, id: $id}) "
            f"RETURN {_symbol_projection(True)} AS symbol",
            "reading a symbol",
            repo_id=repo_id,
            id=symbol_id,
        )
        return rows[0]["symbol"] if rows else None

    def find_symbols(
        self, repo_id: str, names: Iterable[str], limit: int = 20
    ) -> list[dict[str, Any]]:
        """Symbols whose name, qualified name or module-qualified name is one of `names`
        (case-sensitive: "App" is a class, "app" is a word)."""
        rows = self._read(
            "MATCH (s:Symbol {repo_id: $repo_id}) "
            "WHERE s.name IN $names OR s.qualified_name IN $names "
            "OR s.module + '.' + s.qualified_name IN $names "
            f"RETURN {_symbol_projection(True)} AS symbol "
            "ORDER BY s.path, s.start_line LIMIT $limit",
            "searching symbols by name",
            repo_id=repo_id,
            names=sorted(set(names)),
            limit=limit,
        )
        return [row["symbol"] for row in rows]

    def get_relations(
        self,
        repo_id: str,
        types: Iterable[str] | None = None,
        node_ids: Iterable[str] | None = None,
    ) -> list[Relation]:
        """Relations of a repository, or only those touching `node_ids` (either direction)."""
        if node_ids is not None:
            return self._relations_of(repo_id, list(node_ids), types)
        rows = self._read(
            "CALL () { "
            "  MATCH (a:Repository {id: $repo_id}) RETURN a "
            "  UNION MATCH (a:File {repo_id: $repo_id}) RETURN a "
            "  UNION MATCH (a:Symbol {repo_id: $repo_id}) RETURN a "
            "} "
            "MATCH (a)-[rel]->(b) WHERE $types IS NULL OR type(rel) IN $types "
            "RETURN type(rel) AS type, a.id AS source, b.id AS target, rel.line AS line "
            "ORDER BY type, source, target",
            "reading relations",
            repo_id=repo_id,
            types=list(types) if types is not None else None,
        )
        return [Relation(r["type"], r["source"], r["target"], r["line"]) for r in rows]

    def _relations_of(
        self, repo_id: str, node_ids: list[str], types: Iterable[str] | None
    ) -> list[Relation]:
        rows = self._read(
            "CALL () { "
            "  MATCH (n:File {repo_id: $repo_id}) WHERE n.id IN $ids RETURN n "
            "  UNION MATCH (n:Symbol {repo_id: $repo_id}) WHERE n.id IN $ids RETURN n "
            "} "
            "MATCH (n)-[rel]-() WHERE $types IS NULL OR type(rel) IN $types "
            "WITH DISTINCT rel "
            "RETURN type(rel) AS type, startNode(rel).id AS source, endNode(rel).id AS target, "
            "rel.line AS line ORDER BY type, source, target",
            "reading neighbouring relations",
            repo_id=repo_id,
            ids=node_ids,
            types=list(types) if types is not None else None,
        )
        return [Relation(r["type"], r["source"], r["target"], r["line"]) for r in rows]

    def vector_search(
        self, repo_id: str, vector: Sequence[float], k: int = 5
    ) -> list[dict[str, Any]]:
        """The `k` symbols of `repo_id` closest to `vector`, each with a cosine `score`.

        The index is shared by all repositories and Neo4j 5 filters after the top-N, so
        more candidates than `k` are requested.
        """
        rows = self._read(
            "CALL db.index.vector.queryNodes($index, $candidates, $vector) "
            "YIELD node AS s, score WHERE s.repo_id = $repo_id "
            f"RETURN {_symbol_projection(True)} AS symbol, score "
            "ORDER BY score DESC LIMIT $k",
            "searching the vector index",
            index=VECTOR_INDEX,
            candidates=min(max(k * 20, 100), 1000),
            vector=list(vector),
            repo_id=repo_id,
            k=k,
        )
        return [{**row["symbol"], "score": row["score"]} for row in rows]

    # --- helpers -------------------------------------------------------------------------

    def _run(self, query: str, action: str, **params: Any) -> list[dict[str, Any]]:
        return self._execute(query, action, RoutingControl.WRITE, params)

    def _read(self, query: str, action: str, **params: Any) -> list[dict[str, Any]]:
        return self._execute(query, action, RoutingControl.READ, params)

    def _execute(self, query, action, routing, params) -> list[dict[str, Any]]:
        # execute_query runs a managed transaction, retried on transient errors.
        with _neo4j_errors(action):
            records, _, _ = self._driver.execute_query(
                query,
                parameters_=params,
                routing_=routing,
                database_=self.settings.neo4j_database,
            )
        return [record.data() for record in records]

    def _write_batches(self, query: str, rows: Sequence[dict], action: str, **params) -> None:
        for batch in _chunks(rows, self.batch_size):
            self._run(query, action, rows=batch, **params)

    def _delete_in_batches(self, query: str, action: str, **params) -> int:
        total = 0
        while True:
            deleted = self._run(query, action, batch=self.batch_size, **params)[0]["deleted"]
            total += deleted
            if deleted < self.batch_size:
                return total


_REPOSITORY_FIELDS = (
    "r.id AS id, r.url AS url, r.status AS status, toString(r.indexed_at) AS indexed_at, "
    "r.file_count AS file_count, r.symbol_count AS symbol_count, "
    "r.relation_count AS relation_count, r.embedding_model AS embedding_model, "
    "r.commit_sha AS commit_sha, r.error AS error"
)


def _symbol_projection(with_source: bool) -> str:
    fields = [*_SYMBOL_FIELDS, "source"] if with_source else _SYMBOL_FIELDS
    return "s {" + ", ".join(f".{name}" for name in fields) + "}"


def symbol_properties(symbol: Symbol, module: str) -> dict[str, Any]:
    props = asdict(symbol)
    props["module"] = module
    props["decorators"] = list(symbol.decorators)
    props["bases"] = list(symbol.bases)
    return props


def node_labels(graph: CodeGraph) -> dict[str, str]:
    """Map every node id of `graph` to the label used to MATCH it."""
    labels = {f.path: "File" for f in graph.files}
    labels.update({s.id: "Symbol" for s in graph.symbols})
    labels.update({f"module:{m}": "ExternalModule" for m in graph.external_modules})
    return labels


def relation_groups(graph: CodeGraph) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Relations grouped by (type, source label, target label): one Cypher query per group."""
    labels = node_labels(graph)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rel in graph.relations:
        source, target = labels.get(rel.source_id), labels.get(rel.target_id)
        if rel.type not in RELATION_TYPES or source is None or target is None:
            log.warning("Skipping relation with an unknown endpoint: %s", rel)
            continue
        groups[(rel.type, source, target)].append(
            {"source": rel.source_id, "target": rel.target_id, "line": rel.line}
        )
    return dict(groups)


def index_repository(
    store: GraphStore,
    repo_id: str,
    graph: CodeGraph,
    *,
    url: str | None = None,
    embedder: Embedder | None = None,
    commit_sha: str | None = None,
) -> IndexStats:
    """Write `graph` and its embeddings under `repo_id`, replacing any previous indexing.

    If writing fails half-way, the repository is marked 'failed' (its data is incomplete).
    """
    embedder = embedder or get_embedder()  # loaded first: fail before touching the graph
    store.ensure_schema(embedder.dimensions)
    run = uuid.uuid4().hex
    stats = IndexStats(
        repo_id=repo_id,
        files=len(graph.files),
        symbols=len(graph.symbols),
        relations=len(graph.relations) + len(graph.files),  # + CONTAINS
        external_modules=len(graph.external_modules),
    )

    log.info("Indexing %s: %d files, %d symbols", repo_id, stats.files, stats.symbols)
    store.begin(repo_id, run, url)
    try:
        store.write_graph(repo_id, graph, run)
        stats.removed = store.remove_stale(repo_id, run)
        stats.embedded, stats.embeddings_reused = _embed_symbols(store, repo_id, graph, embedder)
        store.finish(repo_id, stats, embedder.model_name, commit_sha)
    except Exception as exc:
        try:
            store.mark_failed(repo_id, str(exc))
        except GraphStoreError:  # Neo4j itself is down: the status stays 'indexing'
            log.warning("Could not mark %s as failed", repo_id)
        raise
    log.info("Indexed %s: %s", repo_id, stats)
    return stats


def _embed_symbols(
    store: GraphStore, repo_id: str, graph: CodeGraph, embedder: Embedder
) -> tuple[int, int]:
    """Embed symbols whose text changed since the last indexing. Returns (embedded, reused)."""
    existing = store.embedding_hashes(repo_id)
    pending = (
        {"id": symbol.id, "text": text, "hash": text_hash(text, embedder.model_name)}
        for symbol, text in symbol_documents(graph)
    )
    embedded = reused = 0
    for chunk in _chunks(pending, store.batch_size):
        todo = [row for row in chunk if existing.get(row["id"]) != row["hash"]]
        reused += len(chunk) - len(todo)
        vectors = embedder.embed_documents([row["text"] for row in todo])
        rows = [
            {"id": r["id"], "vector": v, "hash": r["hash"]}
            for r, v in zip(todo, vectors, strict=True)
        ]
        store.save_embeddings(repo_id, rows)
        embedded += len(todo)
    return embedded, reused


def _chunks(items: Iterable, size: int) -> Iterator[list]:
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch
