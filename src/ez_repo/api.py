"""FastAPI backend: repositories, background indexing and questions.

Routes validate input and translate errors; the work is done by the existing services
(`tasks.index_github_repository` -> ingest / parser / graph, and `qa.ask_repository`).

- One GraphStore (one Neo4j driver) per process, created on first use.
- The embedding model and the Ollama client are loaded once per process by their modules.
- Indexing runs in background threads (see tasks.py). Tasks are kept in memory: they are
  lost when the server restarts, so run a single process.
- Listens on 127.0.0.1 by default (API_HOST). No authentication: keep it local.
"""

import re
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from urllib.parse import quote

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ez_repo import __version__
from ez_repo.config import Settings, get_settings
from ez_repo.embeddings import EmbeddingError
from ez_repo.graph import GraphStore, GraphStoreError, GraphUnavailable, repo_id_for
from ez_repo.ingest import InvalidRepoURL, RepoRef, is_commit_sha, parse_github_url, remove_clone
from ez_repo.qa import (
    Answer,
    LLMError,
    LLMTimeout,
    LLMUnavailable,
    RepositoryNotIndexed,
    RepositoryNotReady,
    ask_repository,
)
from ez_repo.tasks import Task, TaskConflict, TaskManager, TooManyTasks, index_github_repository

REPO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}/[a-z0-9._-]{1,100}$")
TASKS_LOST_HINT = "Tasks are kept in memory and are lost when the API restarts."

Indexer = Callable[..., dict[str, Any]]  # (store, url, progress, settings) -> summary


# --- schemas -----------------------------------------------------------------------------


def _repo_id(value: str) -> str:
    value = value.strip().lower()
    if not REPO_ID_RE.match(value) or value.split("/")[1] in {".", ".."}:
        raise ValueError("Invalid repository id: expected 'owner/name'.")
    return value


class ImportRequest(BaseModel):
    url: str = Field(
        min_length=1, max_length=300, examples=["https://github.com/pypa/sampleproject"]
    )


class ChatRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=140, examples=["pypa/sampleproject"])
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("repo_id")
    @classmethod
    def _valid_repo_id(cls, value: str) -> str:
        return _repo_id(value)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("The question is empty.")
        return value.strip()


class IndexResultOut(BaseModel):
    repo_id: str
    files: int
    symbols: int
    relations: int
    external_modules: int
    removed: int
    embedded: int
    embeddings_reused: int
    commit_sha: str | None
    skipped_files: int
    files_with_errors: int
    truncated: bool
    warnings: list[str]


class TaskOut(BaseModel):
    id: str
    repo_id: str
    url: str
    status: Literal["pending", "running", "completed", "failed"]
    stage: str
    error: str | None
    result: IndexResultOut | None
    created_at: float  # Unix timestamps
    started_at: float | None
    finished_at: float | None
    elapsed_seconds: float | None

    @classmethod
    def of(cls, task: Task) -> "TaskOut":
        return cls(
            id=task.id,
            repo_id=task.repo_id,
            url=task.url,
            status=task.status,
            stage=task.stage,
            error=task.error,
            result=task.result,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
            elapsed_seconds=round(task.elapsed, 1) if task.elapsed is not None else None,
        )


class RepositoryOut(BaseModel):
    id: str
    url: str | None
    # "interrupted": left in 'indexing' by a task that no longer exists (API restarted).
    status: Literal["ready", "indexing", "failed", "interrupted"]
    indexed_at: str | None
    file_count: int | None
    symbol_count: int | None
    relation_count: int | None
    embedding_model: str | None
    commit_sha: str | None
    error: str | None
    task_id: str | None  # active indexing task, if any


class SourceOut(BaseModel):
    label: str
    symbol_id: str
    kind: str
    name: str
    path: str
    start_line: int
    end_line: int
    location: str
    cited: bool
    url: str | None  # GitHub link to the lines of the indexed commit, when it is known


class ChatResponse(BaseModel):
    repo_id: str
    question: str
    answer: str
    language: str
    sources: list[SourceOut]  # every source sent to the model; `cited` marks those used
    commit_sha: str | None
    invalid_citations: list[str]
    unknown_paths: list[str]
    context_truncated: bool
    incomplete: bool
    duration_seconds: float


class DeleteOut(BaseModel):
    repo_id: str
    deleted_nodes: int
    clone_removed: bool


# --- helpers -----------------------------------------------------------------------------


def github_line_url(
    repo_id: str, repo_url: str | None, commit_sha: str | None, path: str, start: int, end: int
) -> str | None:
    """Link to lines of the indexed commit, or None unless every part is verified."""
    if not repo_url or not is_commit_sha(commit_sha):
        return None
    try:
        repo = parse_github_url(repo_url)
    except InvalidRepoURL:
        return None
    parts = path.split("/")
    if repo_id_for(repo) != repo_id or not path or any(p in {"", ".", ".."} for p in parts):
        return None
    anchor = f"#L{start}" if start == end else f"#L{start}-L{end}"
    return f"{repo.url}/blob/{commit_sha}/{quote(path)}{anchor}"


def chat_response(answer: Answer, repo: dict[str, Any], duration: float) -> ChatResponse:
    commit_sha = repo.get("commit_sha") if is_commit_sha(repo.get("commit_sha")) else None
    sources = [
        SourceOut(
            label=s.label,
            symbol_id=s.symbol_id,
            kind=s.kind,
            name=s.name,
            path=s.path,
            start_line=s.start_line,
            end_line=s.end_line,
            location=s.location,
            cited=s.cited,
            url=github_line_url(
                answer.repo_id, repo.get("url"), commit_sha, s.path, s.start_line, s.end_line
            ),
        )
        for s in answer.sources
    ]
    return ChatResponse(
        repo_id=answer.repo_id,
        question=answer.question,
        answer=answer.answer,
        language=answer.language,
        sources=sources,
        commit_sha=commit_sha,
        invalid_citations=answer.invalid_citations,
        unknown_paths=answer.unknown_paths,
        context_truncated=answer.context_truncated,
        incomplete=answer.incomplete,
        duration_seconds=round(duration, 1),
    )


def _error(code: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": message, **extra})


_ERROR_CODES: list[tuple[type[Exception], int]] = [
    (InvalidRepoURL, status.HTTP_422_UNPROCESSABLE_CONTENT),
    (RepositoryNotReady, status.HTTP_409_CONFLICT),
    (RepositoryNotIndexed, status.HTTP_404_NOT_FOUND),
    (TooManyTasks, status.HTTP_429_TOO_MANY_REQUESTS),
    (LLMTimeout, status.HTTP_504_GATEWAY_TIMEOUT),
    (LLMUnavailable, status.HTTP_503_SERVICE_UNAVAILABLE),
    (LLMError, status.HTTP_502_BAD_GATEWAY),
    (EmbeddingError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (GraphUnavailable, status.HTTP_503_SERVICE_UNAVAILABLE),
    (GraphStoreError, status.HTTP_503_SERVICE_UNAVAILABLE),
]


# --- application -------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None,
    *,
    store: GraphStore | None = None,
    tasks: TaskManager | None = None,
    indexer: Indexer | None = None,
) -> FastAPI:
    """Build the API. Tests pass fakes for `store`, `tasks` and `indexer`."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.tasks.shutdown()
        if app.state.owns_store and app.state.store is not None:
            app.state.store.close()

    app = FastAPI(title="ez-repo", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.owns_store = store is None
    app.state.store_lock = threading.Lock()
    app.state.tasks = tasks or TaskManager.from_settings(settings)
    app.state.indexer = indexer or index_github_repository
    # One generation at a time: Ollama on CPU would only queue the others until they time out.
    app.state.chat_slot = threading.BoundedSemaphore(1)

    def get_store(request: Request) -> GraphStore:
        state = request.app.state
        if state.store is None:
            with state.store_lock:
                if state.store is None:
                    state.store = GraphStore(state.settings)
        return state.store

    def get_tasks(request: Request) -> TaskManager:
        return request.app.state.tasks

    Store = Annotated[GraphStore, Depends(get_store)]
    Tasks = Annotated[TaskManager, Depends(get_tasks)]

    for exc_type, code in _ERROR_CODES:
        app.add_exception_handler(exc_type, lambda _request, exc, code=code: _error(code, str(exc)))
    app.add_exception_handler(
        TaskConflict,
        lambda _request, exc: _error(status.HTTP_409_CONFLICT, str(exc), task_id=exc.task.id),
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/health/services")
    def services_health(store: Store) -> dict[str, dict[str, Any]]:
        """Neo4j and Ollama reachability (does not load any model)."""
        try:
            store.verify()
            neo4j: dict[str, Any] = {"ok": True}
        except GraphStoreError as exc:
            neo4j = {"ok": False, "error": str(exc)}
        try:
            tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3).json()
            names = {m.get("name") for m in tags.get("models", [])}
            ollama: dict[str, Any] = {"ok": settings.ollama_model in names}
            if not ollama["ok"]:
                ollama["error"] = f"Model missing: ollama pull {settings.ollama_model}"
        except (httpx.HTTPError, ValueError) as exc:
            ollama = {"ok": False, "error": f"Ollama unreachable at {settings.ollama_base_url}"}
            ollama["detail"] = str(exc)
        return {"neo4j": neo4j, "ollama": {**ollama, "model": settings.ollama_model}}

    @app.get("/repositories")
    def list_repositories(store: Store, tasks: Tasks) -> list[RepositoryOut]:
        repos = []
        for repo in store.list_repositories():
            task = tasks.active_for(repo["id"])
            state = repo.get("status")
            if task is not None:
                state = "indexing"
            elif state == "indexing":
                state = "interrupted"
            elif state not in {"ready", "failed"}:
                state = "failed"
            fields = {name: repo.get(name) for name in RepositoryOut.model_fields}
            repos.append(
                RepositoryOut(**{**fields, "status": state, "task_id": task.id if task else None})
            )
        return repos

    @app.post("/repositories", status_code=status.HTTP_202_ACCEPTED)
    def import_repository(body: ImportRequest, store: Store, tasks: Tasks) -> TaskOut:
        """Start indexing a public GitHub repository; poll GET /tasks/{id} for the result."""
        repo = parse_github_url(body.url)
        indexer = app.state.indexer
        task = tasks.submit(
            repo_id_for(repo),
            repo.url,
            lambda progress: indexer(store, repo.url, progress, settings),
        )
        return TaskOut.of(task)

    @app.delete("/repositories/{repo_id:path}")
    def delete_repository(repo_id: str, store: Store, tasks: Tasks) -> DeleteOut:
        try:
            repo_id = _repo_id(repo_id)
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
        if (task := tasks.active_for(repo_id)) is not None:
            raise TaskConflict(task)
        if store.get_repository(repo_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Repository {repo_id} is not indexed.")
        deleted = store.delete_repository(repo_id)
        owner, name = repo_id.split("/")
        removed = remove_clone(RepoRef(owner, name), settings.repos_dir)
        return DeleteOut(repo_id=repo_id, deleted_nodes=deleted, clone_removed=removed)

    @app.get("/tasks")
    def list_tasks(tasks: Tasks) -> list[TaskOut]:
        """Recent indexing tasks, newest first (in memory: lost when the API restarts)."""
        return [TaskOut.of(t) for t in tasks.list()]

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str, tasks: Tasks) -> TaskOut:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown task. {TASKS_LOST_HINT}")
        return TaskOut.of(task)

    @app.post("/chat")
    def chat(body: ChatRequest, store: Store, tasks: Tasks) -> ChatResponse:
        """Answer a question about an indexed repository (CPU: up to a few tens of seconds)."""
        if (task := tasks.active_for(body.repo_id)) is not None:
            raise TaskConflict(task)
        repo = store.get_repository(body.repo_id)
        if repo is None:
            raise RepositoryNotIndexed(f"Repository {body.repo_id} is not indexed.")
        if not app.state.chat_slot.acquire(blocking=False):
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "Another question is being answered. Retry when it is done.",
            )
        try:
            started = time.perf_counter()
            answer = ask_repository(store, body.repo_id, body.question, settings=settings)
        finally:
            app.state.chat_slot.release()
        return chat_response(answer, repo, time.perf_counter() - started)

    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run("ez_repo.api:app", host=settings.api_host, port=settings.api_port, workers=1)


if __name__ == "__main__":
    run()
