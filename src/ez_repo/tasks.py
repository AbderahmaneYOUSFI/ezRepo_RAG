"""Background indexing: an in-memory task registry over a few worker threads.

Tasks live in the API process only: they are lost when the server restarts (a repository
left in the 'indexing' state is then reported as interrupted and can be imported again).
Run the API as a single process.

Limits, to fit a 16 GB laptop: at most `max_workers` indexings run at once (1 by default),
at most `max_pending` wait in the queue, and a repository is never indexed twice at once.
"""

import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ez_repo.config import Settings, get_settings
from ez_repo.graph import GraphStore, index_repository, repo_id_for
from ez_repo.ingest import ingest_repository
from ez_repo.parser import parse_repository

log = logging.getLogger(__name__)

PENDING, RUNNING, COMPLETED, FAILED = "pending", "running", "completed", "failed"
ACTIVE_STATES = frozenset({PENDING, RUNNING})

Progress = Callable[[str], None]
Job = Callable[[Progress], dict[str, Any]]


class TaskError(Exception):
    """Base error for background tasks."""


class TaskConflict(TaskError):
    def __init__(self, task: "Task"):
        self.task = task
        super().__init__(f"{task.repo_id} is already being indexed (task {task.id}).")


class TooManyTasks(TaskError):
    pass


@dataclass
class Task:
    id: str
    repo_id: str
    url: str
    status: str = PENDING
    stage: str = "queued"
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATES

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at


class TaskManager:
    def __init__(self, max_workers: int = 1, max_pending: int = 3, history: int = 50):
        self.max_workers = max(1, max_workers)
        self.max_pending = max(0, max_pending)
        self.history = max(0, history)
        self._tasks: OrderedDict[str, Task] = OrderedDict()
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[Task, Job] | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "TaskManager":
        return cls(settings.index_max_concurrent, settings.index_max_pending, settings.task_history)

    def submit(self, repo_id: str, url: str, job: Job) -> Task:
        """Queue `job` for `repo_id`. Raises TaskConflict or TooManyTasks."""
        with self._lock:
            if self._closed:
                raise TooManyTasks("The server is shutting down.")
            for task in self._tasks.values():
                if task.active and task.repo_id == repo_id:
                    raise TaskConflict(replace(task))
            pending = sum(t.status == PENDING for t in self._tasks.values())
            if pending >= self.max_pending + self.max_workers - self._running():
                raise TooManyTasks(
                    f"Too many indexings queued ({pending}). Wait for one to finish."
                )
            task = Task(id=uuid.uuid4().hex, repo_id=repo_id, url=url)
            self._tasks[task.id] = task
            self._prune()
            self._start_workers()
            snapshot = replace(task)
        self._queue.put((task, job))
        return snapshot

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return replace(task) if task else None

    def list(self) -> list[Task]:
        """All known tasks, newest first."""
        with self._lock:
            return [replace(t) for t in reversed(self._tasks.values())]

    def active_for(self, repo_id: str) -> Task | None:
        with self._lock:
            for task in self._tasks.values():
                if task.active and task.repo_id == repo_id:
                    return replace(task)
        return None

    def shutdown(self) -> None:
        """Stop accepting tasks. Workers are daemon threads: a running indexing is abandoned
        when the process exits (its repository is then reported as interrupted)."""
        with self._lock:
            self._closed = True
            workers = len(self._workers)
        for _ in range(workers):
            self._queue.put(None)

    # --- internals -----------------------------------------------------------------------

    def _running(self) -> int:
        return sum(t.status == RUNNING for t in self._tasks.values())

    def _start_workers(self) -> None:
        while len(self._workers) < self.max_workers:
            worker = threading.Thread(
                target=self._work, name=f"ez-repo-index-{len(self._workers)}", daemon=True
            )
            self._workers.append(worker)
            worker.start()

    def _work(self) -> None:
        while (item := self._queue.get()) is not None:
            task, job = item
            if self._closed:
                self._update(task, status=FAILED, stage="cancelled", error="Server stopped.")
                continue
            self._run(task, job)

    def _run(self, task: Task, job: Job) -> None:
        self._update(task, status=RUNNING, stage="starting", started_at=time.time())
        try:
            result = job(lambda stage: self._update(task, stage=stage))
        except Exception as exc:  # reported through the task, never raised in the worker
            log.warning("Indexing %s failed: %s", task.repo_id, exc, exc_info=True)
            self._update(
                task,
                status=FAILED,
                stage="failed",
                error=str(exc) or type(exc).__name__,
                finished_at=time.time(),
            )
        else:
            self._update(
                task, status=COMPLETED, stage="done", result=result, finished_at=time.time()
            )

    def _update(self, task: Task, **changes: Any) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(task, name, value)

    def _prune(self) -> None:
        finished = [t.id for t in self._tasks.values() if not t.active]
        for task_id in finished[: max(0, len(finished) - self.history)]:
            del self._tasks[task_id]


def index_github_repository(
    store: GraphStore, url: str, progress: Progress, settings: Settings | None = None
) -> dict[str, Any]:
    """Clone, parse and index a public GitHub repository. Returns the indexing summary."""
    settings = settings or get_settings()
    progress("cloning the repository")
    ingested = ingest_repository(url, settings)
    progress(f"parsing {len(ingested.files)} Python files")
    graph = parse_repository(ingested.files)
    progress(f"writing the graph and embedding {len(graph.symbols)} symbols")
    stats = index_repository(
        store,
        repo_id_for(ingested.repo),
        graph,
        url=ingested.repo.url,
        commit_sha=ingested.commit_sha,
    )
    warnings = []
    if not ingested.files:
        warnings.append("No readable Python file was found in this repository.")
    if ingested.truncated:
        warnings.append("File limits were reached (MAX_FILES / MAX_TOTAL_SIZE): partial index.")
    if ingested.commit_sha is None:
        warnings.append("The indexed commit is unknown: sources will have no GitHub links.")
    return {
        **asdict(stats),
        "commit_sha": ingested.commit_sha,
        "skipped_files": len(ingested.skipped),
        "files_with_errors": sum(f.has_errors for f in graph.files),
        "truncated": ingested.truncated,
        "warnings": warnings,
    }
