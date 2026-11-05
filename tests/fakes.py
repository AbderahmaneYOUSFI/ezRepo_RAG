"""In-memory stand-ins for GraphStore and the embedder (unit tests, no services)."""

from pathlib import Path

from ez_repo.embeddings import EMBEDDING_DIMENSIONS
from ez_repo.graph import symbol_properties
from ez_repo.ingest import SourceFile, collect_python_files
from ez_repo.parser import CodeGraph, Relation, parse_repository

DEMO_REPO = Path(__file__).parent / "fixtures" / "demo_repo"


def demo_graph() -> CodeGraph:
    return parse_repository(collect_python_files(DEMO_REPO, 10**6, 100, 10**7).files)


def graph_from(sources: dict[str, str]) -> CodeGraph:
    return parse_repository(SourceFile(p, t, len(t)) for p, t in sources.items())


class FakeEmbedder:
    model_name = "fake"
    dimensions = EMBEDDING_DIMENSIONS

    def embed_query(self, text):
        return [0.0] * EMBEDDING_DIMENSIONS


class FakeStore:
    """Same read API as GraphStore over parsed graphs; `hits` scripts the vector search."""

    def __init__(self, graphs: dict[str, CodeGraph], hits=None, status="ready", meta=None):
        self.graphs = graphs
        self.hits = hits or {}
        self.status = status
        self.meta = meta or {}  # repo_id -> Repository properties (url, commit_sha, status...)
        self.deleted = []
        self.repo_ids = []  # repo_id of every call, to check isolation
        self.symbols = {
            repo_id: {
                s.id: {**symbol_properties(s, f.module), "repo_id": repo_id}
                for f in graph.files
                for s in f.symbols
            }
            for repo_id, graph in graphs.items()
        }

    def _use(self, repo_id):
        self.repo_ids.append(repo_id)
        return self.graphs.get(repo_id)

    def get_repository(self, repo_id):
        if not self._use(repo_id):
            return None
        return {"id": repo_id, "status": self.status, **self.meta.get(repo_id, {})}

    def list_repositories(self):
        return [self.get_repository(repo_id) for repo_id in sorted(self.graphs)]

    def delete_repository(self, repo_id):
        graph = self.graphs.pop(repo_id)
        self.deleted.append(repo_id)
        return 1 + len(graph.files) + len(graph.symbols)

    def verify(self):
        pass

    def get_files(self, repo_id):
        graph = self._use(repo_id)
        return [{"path": f.path, "module": f.module, "has_errors": f.has_errors}
                for f in graph.files]  # fmt: skip

    def get_symbols(self, repo_id, kind=None, path=None, with_source=False):
        self._use(repo_id)
        return [
            s for s in self.symbols[repo_id].values()
            if (kind is None or s["kind"] == kind) and (path is None or s["path"] == path)
        ]  # fmt: skip

    def get_symbol(self, repo_id, symbol_id):
        self._use(repo_id)
        return self.symbols[repo_id].get(symbol_id)

    def find_symbols(self, repo_id, names, limit=20):
        self._use(repo_id)
        names = set(names)
        found = [
            s for s in self.symbols[repo_id].values()
            if {s["name"], s["qualified_name"], f"{s['module']}.{s['qualified_name']}"} & names
        ]  # fmt: skip
        return found[:limit]

    def vector_search(self, repo_id, vector, k=5):
        self._use(repo_id)
        return [
            {**self.symbols[repo_id][symbol_id], "score": score}
            for symbol_id, score in self.hits.get(repo_id, [])[:k]
        ]

    def get_relations(self, repo_id, types=None, node_ids=None):
        graph = self._use(repo_id)
        relations = graph.relations + [Relation("CONTAINS", repo_id, f.path) for f in graph.files]
        ids = set(node_ids) if node_ids is not None else None
        return [
            r for r in relations
            if (types is None or r.type in types)
            and (ids is None or r.source_id in ids or r.target_id in ids)
        ]  # fmt: skip
