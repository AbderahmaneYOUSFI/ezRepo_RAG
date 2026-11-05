import threading
import time
from pathlib import Path

import pytest

from ez_repo import tasks as tasks_module
from ez_repo.config import Settings
from ez_repo.graph import IndexStats
from ez_repo.ingest import IngestResult, RepoRef, SourceFile
from ez_repo.tasks import (
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    TaskConflict,
    TaskManager,
    TooManyTasks,
    index_github_repository,
)


def wait_for(manager, task_id, *states, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = manager.get(task_id)
        if task.status in states:
            return task
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} stayed {manager.get(task_id).status}")


def gated_job(gate: threading.Event, result=None):
    def job(progress):
        progress("working")
        assert gate.wait(5), "gate never opened"
        return result if result is not None else {"ok": True}

    return job


@pytest.fixture
def manager():
    manager = TaskManager(max_workers=1, max_pending=2, history=10)
    yield manager
    manager.shutdown()


def test_task_lifecycle(manager):
    gate = threading.Event()
    task = manager.submit("acme/demo", "https://github.com/acme/demo", gated_job(gate, {"n": 1}))
    assert task.status == PENDING and task.stage == "queued" and task.started_at is None

    running = wait_for(manager, task.id, RUNNING)
    assert running.stage == "working" and running.started_at is not None
    assert manager.active_for("acme/demo").id == task.id

    gate.set()
    done = wait_for(manager, task.id, COMPLETED)
    assert done.result == {"n": 1} and done.error is None and done.stage == "done"
    assert done.finished_at >= done.started_at and done.elapsed >= 0
    assert manager.active_for("acme/demo") is None


def test_failure_is_recorded(manager):
    def job(progress):
        raise RuntimeError("clone failed")

    task = manager.submit("acme/demo", "u", job)
    failed = wait_for(manager, task.id, FAILED)
    assert failed.error == "clone failed" and failed.stage == "failed" and failed.result is None


def test_failure_without_message_uses_the_exception_name(manager):
    def job(progress):
        raise KeyError()

    failed = wait_for(manager, manager.submit("acme/demo", "u", job).id, FAILED)
    assert failed.error == "KeyError"


def test_same_repository_cannot_be_indexed_twice_at_once(manager):
    gate = threading.Event()
    first = manager.submit("acme/demo", "u", gated_job(gate))
    with pytest.raises(TaskConflict) as info:
        manager.submit("acme/demo", "u", gated_job(gate))
    assert info.value.task.id == first.id

    gate.set()
    wait_for(manager, first.id, COMPLETED)
    again = manager.submit("acme/demo", "u", gated_job(gate))  # allowed once finished
    wait_for(manager, again.id, COMPLETED)


def test_concurrency_is_limited(manager):
    gate = threading.Event()
    first = manager.submit("acme/one", "u", gated_job(gate))
    wait_for(manager, first.id, RUNNING)
    second = manager.submit("acme/two", "u", gated_job(gate))
    time.sleep(0.1)
    assert manager.get(second.id).status == PENDING  # one worker: waits for the first

    gate.set()
    wait_for(manager, first.id, COMPLETED)
    wait_for(manager, second.id, COMPLETED)


def test_queue_is_bounded():
    manager = TaskManager(max_workers=1, max_pending=1)
    gate = threading.Event()
    try:
        first = manager.submit("acme/one", "u", gated_job(gate))
        wait_for(manager, first.id, RUNNING)
        manager.submit("acme/two", "u", gated_job(gate))
        with pytest.raises(TooManyTasks):
            manager.submit("acme/three", "u", gated_job(gate))
    finally:
        gate.set()
        manager.shutdown()


def test_history_is_pruned_and_listed_newest_first():
    manager = TaskManager(max_workers=1, max_pending=5, history=2)
    try:
        ids = []
        for i in range(4):
            ids.append(manager.submit(f"acme/r{i}", "u", lambda progress: {}).id)
            wait_for(manager, ids[-1], COMPLETED)
        listed = [t.id for t in manager.list()]
        assert listed[0] == ids[-1]
        assert len(listed) <= 3 and ids[0] not in listed
        assert manager.get(ids[0]) is None
    finally:
        manager.shutdown()


def test_returned_tasks_are_snapshots(manager):
    gate = threading.Event()
    task = manager.submit("acme/demo", "u", gated_job(gate))
    task.status = "completed"
    assert manager.get(task.id).status in {PENDING, RUNNING}
    gate.set()


def test_shutdown_refuses_new_tasks():
    manager = TaskManager()
    manager.shutdown()
    with pytest.raises(TooManyTasks, match="shutting down"):
        manager.submit("acme/demo", "u", lambda progress: {})


def test_from_settings():
    manager = TaskManager.from_settings(
        Settings(_env_file=None, index_max_concurrent=2, index_max_pending=4, task_history=7)
    )
    assert (manager.max_workers, manager.max_pending, manager.history) == (2, 4, 7)


# --- indexing pipeline -------------------------------------------------------------------


def test_index_github_repository_chains_the_services(monkeypatch, tmp_path):
    repo = RepoRef("Acme", "Demo")
    files = [SourceFile("pkg/a.py", "def f():\n    return 1\n", 22)]
    sha = "0123456789abcdef0123456789abcdef01234567"
    calls = {}

    def fake_ingest(url, settings):
        calls["ingest"] = url
        return IngestResult(repo=repo, root=Path(tmp_path), files=files, commit_sha=sha,
                            skipped=[("big.py", "too large")])  # fmt: skip

    def fake_index(store, repo_id, graph, *, url=None, commit_sha=None):
        calls["index"] = (store, repo_id, len(graph.symbols), url, commit_sha)
        return IndexStats(repo_id=repo_id, files=1, symbols=1, relations=2, embedded=1)

    monkeypatch.setattr(tasks_module, "ingest_repository", fake_ingest)
    monkeypatch.setattr(tasks_module, "index_repository", fake_index)
    stages = []

    result = index_github_repository("store", "https://github.com/Acme/Demo", stages.append,
                                     Settings(_env_file=None))  # fmt: skip

    assert calls["ingest"] == "https://github.com/Acme/Demo"
    assert calls["index"] == ("store", "acme/demo", 1, "https://github.com/Acme/Demo", sha)
    assert stages[0] == "cloning the repository" and "parsing 1 Python files" in stages
    assert result["repo_id"] == "acme/demo" and result["commit_sha"] == sha
    assert result["skipped_files"] == 1 and result["files_with_errors"] == 0
    assert result["warnings"] == []


def test_index_github_repository_warns_about_empty_or_unknown_commit(monkeypatch, tmp_path):
    repo = RepoRef("acme", "empty")
    monkeypatch.setattr(
        tasks_module,
        "ingest_repository",
        lambda url, settings: IngestResult(repo=repo, root=tmp_path, truncated=True),
    )
    monkeypatch.setattr(
        tasks_module,
        "index_repository",
        lambda store, repo_id, graph, **kw: IndexStats(repo_id=repo_id),
    )
    result = index_github_repository(None, "acme/empty", lambda s: None, Settings(_env_file=None))
    assert result["commit_sha"] is None and result["truncated"] is True
    assert len(result["warnings"]) == 3
