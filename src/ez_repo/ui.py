"""Streamlit frontend. Run with: streamlit run src/ez_repo/ui.py

A thin client of the FastAPI backend (API_URL): no model, Neo4j connection or heavy work
runs here. The chat history is kept per repository for the browser session only.
"""

import re
import time
from typing import Any

import httpx
import streamlit as st

from ez_repo.config import get_settings
from ez_repo.ingest import is_commit_sha

settings = get_settings()

CHAT_TIMEOUT = settings.ollama_timeout + 60  # the first question also loads the embedder
POLL_SECONDS = 2
ACTIVE = {"pending", "running"}
STATUS = {  # repository status -> (dot, label)
    "ready": (":green[●]", "Ready"),
    "indexing": (":orange[●]", "Indexing"),
    "failed": (":red[●]", "Failed"),
    "interrupted": (":gray[●]", "Interrupted"),
}
# Links written by the model itself are never trusted: only API-verified sources link out.
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((?:https?:)?//[^)\s]+\)")

CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 920px; padding-top: 2.5rem; }
[data-testid="stChatMessage"] { padding: 0.75rem 1rem; }
[data-testid="stChatMessage"] pre { font-size: 0.85rem; }
.ez-meta { opacity: 0.7; font-size: 0.85rem; }
</style>
"""


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# --- API client --------------------------------------------------------------------------


@st.cache_resource
def _client() -> httpx.Client:
    return httpx.Client(base_url=settings.api_url, timeout=10)


def api(method: str, path: str, *, timeout: float = 10, **kwargs: Any) -> Any:
    try:
        response = _client().request(method, path, timeout=timeout, **kwargs)
    except httpx.TimeoutException as exc:
        raise ApiError("The ez-repo API did not answer in time.") from exc
    except httpx.HTTPError as exc:
        raise ApiError(
            f"Cannot reach the ez-repo API at {settings.api_url}. Start it with `ez-repo-api`."
        ) from exc
    if response.is_success:
        return response.json()
    raise ApiError(_detail(response), response.status_code)


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", "")).removeprefix("Value error, ") for d in detail)
    return str(detail) if detail else f"API error (HTTP {response.status_code})."


@st.cache_data(ttl=30, show_spinner=False)
def services_status() -> dict[str, Any] | None:
    try:
        return api("GET", "/health/services", timeout=8)
    except ApiError:
        return None


# --- state -------------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("history", {})  # repo_id -> list of messages
    st.session_state.setdefault("watched", [])  # task ids started from this session
    st.session_state.setdefault("notices", [])  # (kind, text) shown once


def notify(kind: str, text: str) -> None:
    st.session_state.notices.append((kind, text))


def show_notices() -> None:
    for kind, text in st.session_state.notices:
        st.toast(text, icon=":material/check_circle:" if kind == "success" else ":material/error:")
    st.session_state.notices = []


def collect_finished(tasks: list[dict]) -> None:
    """Announce the end of tasks started from this session and select the new repository."""
    by_id = {t["id"]: t for t in tasks}
    for task_id in list(st.session_state.watched):
        task = by_id.get(task_id)
        if task is not None and task["status"] in ACTIVE:
            continue
        st.session_state.watched.remove(task_id)
        if task is None:  # the API restarted: in-memory tasks are gone
            notify("error", "An indexing task was lost (the API restarted). Import it again.")
        elif task["status"] == "completed":
            notify("success", f"{task['repo_id']} is ready.")
            st.session_state.repo = task["repo_id"]
        else:
            notify("error", f"Indexing {task['repo_id']} failed: {task['error']}")


# --- sidebar -----------------------------------------------------------------------------


def import_form(busy: bool) -> None:
    with st.form("import", clear_on_submit=False, border=False):
        url = st.text_input(
            "GitHub repository",
            placeholder="https://github.com/owner/repo",
            disabled=busy,
        )
        submitted = st.form_submit_button(
            "Analyze repository", type="primary", disabled=busy, width="stretch"
        )
    if busy:
        st.caption("One indexing at a time, to keep memory use low.")
    if submitted:
        if not url.strip():
            st.error("Enter a GitHub repository URL.")
            return
        try:
            task = api("POST", "/repositories", json={"url": url.strip()})
        except ApiError as exc:
            st.error(str(exc))
            return
        st.session_state.watched.append(task["id"])
        st.rerun()


@st.fragment(run_every=POLL_SECONDS)
def indexing_progress(task_ids: list[str]) -> None:
    """Polls the running tasks; reruns the whole page when one of them ends."""
    for task_id in task_ids:
        try:
            task = api("GET", f"/tasks/{task_id}")
        except ApiError:
            st.rerun()  # unknown task (API restarted) or API down: refresh everything
            return
        if task["status"] not in ACTIVE:
            st.rerun()
            return
        elapsed = task.get("elapsed_seconds")
        label = f"Indexing {task['repo_id']}"
        with st.status(label, state="running", expanded=True):
            stage = "waiting for a free slot" if task["status"] == "pending" else task["stage"]
            st.write(stage.capitalize())
            if elapsed is not None:
                st.caption(f"{elapsed:.0f} s elapsed · embeddings run on CPU, large "
                           "repositories can take several minutes.")  # fmt: skip


def repository_selector(repos: list[dict]) -> dict | None:
    if not repos:
        st.caption("No repository yet. Import one above.")
        return None
    ids = [r["id"] for r in repos]
    by_id = {r["id"]: r for r in repos}
    if st.session_state.get("repo") not in ids:
        st.session_state.repo = ids[0]
    selected = st.radio(
        "Repositories",
        ids,
        key="repo",
        format_func=lambda rid: f"{STATUS[by_id[rid]['status']][0]} {rid}",
        label_visibility="collapsed",
    )
    return by_id[selected]


def _close_delete_dialog() -> None:
    st.session_state.pop("confirm_delete", None)


@st.dialog("Delete repository", on_dismiss=_close_delete_dialog)
def confirm_delete(repo_id: str) -> None:
    st.write(
        f"Delete **{repo_id}** from ez-repo? Its graph, embeddings and local clone are "
        "removed. The GitHub repository itself is not affected."
    )
    cancel, delete = st.columns(2)
    if cancel.button("Cancel", width="stretch"):
        _close_delete_dialog()
        st.rerun()
    if delete.button("Delete", type="primary", width="stretch"):
        try:
            api("DELETE", f"/repositories/{repo_id}", timeout=60)
        except ApiError as exc:
            st.error(str(exc))
            return
        _close_delete_dialog()
        st.session_state.history.pop(repo_id, None)
        st.session_state.pop("repo", None)
        notify("success", f"{repo_id} was deleted.")
        st.rerun()


def services_panel() -> None:
    services = services_status()
    healthy = services is not None and all(s.get("ok") for s in services.values())
    with st.expander("System", expanded=services is not None and not healthy):
        if services is None:
            st.caption("Service status unavailable.")
            return
        model = services["ollama"]["model"]
        for name, label in (("neo4j", "Neo4j"), ("ollama", f"Ollama · {model}")):
            info = services[name]
            dot = ":green[●]" if info.get("ok") else ":red[●]"
            st.markdown(f"{dot} {label}")
            if not info.get("ok"):
                st.caption(info.get("error", "Unavailable"))
        st.caption(f"API {settings.api_url}")


# --- chat --------------------------------------------------------------------------------


def commit_link(repo: dict) -> str | None:
    sha, url = repo.get("commit_sha"), repo.get("url") or ""
    if is_commit_sha(sha) and url.startswith("https://github.com/"):
        return f"[`{sha[:7]}`]({url}/commit/{sha})"
    return None


def repo_header(repo: dict) -> None:
    st.markdown(f"### {repo['id']}")
    parts = []
    if repo.get("file_count") is not None:
        parts.append(f"{repo['file_count']} files · {repo['symbol_count']} symbols")
    if link := commit_link(repo):
        parts.append(f"commit {link}")
    if repo.get("url"):
        parts.append(f"[GitHub]({repo['url']})")
    if parts:
        st.markdown(f'<span class="ez-meta">{" · ".join(parts)}</span>', unsafe_allow_html=True)


def source_line(source: dict) -> str:
    location = f"`{source['location']}`"
    if source.get("url"):
        location = f"[{location}]({source['url']})"
    return f"**{source['label']}** · `{source['name']}` ({source['kind']}) — {location}"


def render_answer(message: dict) -> None:
    if message.get("error"):
        st.error(message["error"], icon=":material/error:")
        return
    st.markdown(_MARKDOWN_LINK.sub(r"\1", message["answer"]))
    if message.get("incomplete"):
        st.warning(
            "This answer is incomplete: the model reached its length limit "
            "(OLLAMA_NUM_PREDICT). Ask a narrower question.",
            icon=":material/warning:",
        )
    if message.get("unknown_paths"):
        paths = ", ".join(f"`{p}`" for p in message["unknown_paths"])
        st.warning(f"The answer mentions files that are not in the repository: {paths}.")
    notes = []
    if message.get("invalid_citations"):
        notes.append(f"invalid citations removed: {', '.join(message['invalid_citations'])}")
    if message.get("context_truncated"):
        notes.append("the code context was shortened to fit the model")
    sources = message.get("sources") or []
    cited = [s for s in sources if s["cited"]]
    others = [s for s in sources if not s["cited"]]
    if sources:
        with st.expander(f"Sources · {len(cited)} cited, {len(sources)} in context"):
            for source in cited:
                st.markdown(source_line(source))
            if others:
                st.caption("Also given to the model")
                for source in others:
                    st.markdown(source_line(source))
            if not message.get("commit_sha"):
                st.caption("No GitHub links: the indexed commit is unknown.")
    meta = [f"{message.get('duration_seconds', 0):.0f} s"] + notes
    st.markdown(f'<span class="ez-meta">{" · ".join(meta)}</span>', unsafe_allow_html=True)


def ask(repo_id: str, question: str) -> dict:
    try:
        data = api(
            "POST",
            "/chat",
            json={"repo_id": repo_id, "question": question},
            timeout=CHAT_TIMEOUT,
        )
    except ApiError as exc:
        return {"role": "assistant", "error": str(exc)}
    return {"role": "assistant", **data}


def chat(repo: dict, indexing: bool) -> None:
    repo_id = repo["id"]
    history = st.session_state.history.setdefault(repo_id, [])
    ready = repo["status"] == "ready" and not indexing

    if indexing or repo["status"] == "indexing":
        st.info("This repository is being indexed. Questions are disabled until it is done.")
    elif repo["status"] == "failed":
        st.error(f"The last indexing failed: {repo.get('error') or 'unknown error'}. "
                 "Import the repository again.")  # fmt: skip
    elif repo["status"] == "interrupted":
        st.warning("Indexing was interrupted (the API restarted). Import the repository again.")
    elif not history:
        st.caption(
            "Ask about the code: where something is defined, what a class does, who calls "
            "a function. Answers are generated locally on CPU and can take up to a few tens "
            "of seconds."
        )

    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message)

    question = st.chat_input(
        "Ask a question about this repository" if ready else "Questions are unavailable",
        disabled=not ready,
        max_chars=2000,
    )
    if not question:
        return
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        started = time.monotonic()
        with st.spinner("Generating the answer on CPU… this can take a few tens of seconds."):
            answer = ask(repo_id, question)
            history.append(answer)  # stored before any widget call: survives a rerun
        answer.setdefault("duration_seconds", time.monotonic() - started)
        render_answer(answer)


def empty_state() -> None:
    st.markdown("## ez-repo")
    st.markdown(
        "Ask questions about any public GitHub repository. ez-repo parses the Python code, "
        "builds a dependency graph and answers with verified file and line references — "
        "everything runs locally."
    )
    st.markdown(
        "1. Paste a repository URL in the sidebar and click **Analyze repository**.\n"
        "2. Wait for the indexing to finish (a few minutes on CPU for a large repository).\n"
        "3. Ask your questions."
    )


# --- page --------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="ez-repo", page_icon=":material/hub:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    try:
        repos = api("GET", "/repositories")
        tasks = api("GET", "/tasks")
    except ApiError as exc:
        with st.sidebar:
            st.markdown("## ez-repo")
        st.error(str(exc), icon=":material/cloud_off:")
        st.caption("The page reconnects when you interact with it or reload it.")
        return

    collect_finished(tasks)
    active = [t for t in tasks if t["status"] in ACTIVE]
    indexing_ids = {t["repo_id"] for t in active}

    with st.sidebar:
        st.markdown("## ez-repo")
        st.caption("Chat with a GitHub repository — 100% local.")
        import_form(busy=bool(active))
        if active:
            indexing_progress([t["id"] for t in active])
        st.divider()
        st.markdown("**Repositories**")
        repo = repository_selector(repos)
        if repo is not None:
            busy = repo["id"] in indexing_ids
            if st.button(
                "Delete repository",
                icon=":material/delete:",
                type="tertiary",
                disabled=busy,
                help="Unavailable while the repository is being indexed." if busy else None,
            ):
                st.session_state.confirm_delete = repo["id"]
            if st.session_state.get("confirm_delete") == repo["id"] and not busy:
                confirm_delete(repo["id"])
        st.divider()
        services_panel()

    show_notices()
    if repo is None:
        empty_state()
        return
    repo_header(repo)
    chat(repo, indexing=repo["id"] in indexing_ids)


main()
