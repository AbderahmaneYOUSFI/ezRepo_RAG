"""Connectivity checks for local services. Run with: pytest -m integration"""

import httpx
import pytest

pytestmark = pytest.mark.integration


def test_neo4j_reachable(settings):
    from neo4j import GraphDatabase

    auth = (settings.neo4j_user, settings.neo4j_password)
    with GraphDatabase.driver(settings.neo4j_uri, auth=auth) as driver:
        driver.verify_connectivity()


def test_ollama_model_available(settings):
    tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5).json()
    names = {m["name"] for m in tags.get("models", [])}
    assert settings.ollama_model in names, f"Run: ollama pull {settings.ollama_model}"


def test_ollama_generates_with_configured_model(settings):
    payload = {
        "model": settings.ollama_model,
        "prompt": "Reply with the single word OK.",
        "stream": False,
        "options": {"num_predict": 5},
    }
    response = httpx.post(f"{settings.ollama_base_url}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    body = response.json()
    assert body["model"] == settings.ollama_model
    assert body["response"].strip()
