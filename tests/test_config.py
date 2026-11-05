from ez_repo.config import Settings


def test_defaults_target_local_services(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
    s = Settings(_env_file=None)
    assert s.neo4j_uri.startswith("bolt://localhost")
    assert s.ollama_model == "qwen2.5-coder:3b"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "other:1b")
    monkeypatch.setenv("API_PORT", "9000")
    s = Settings(_env_file=None)
    assert s.ollama_model == "other:1b"
    assert s.api_port == 9000


def test_dotenv_wins_over_os_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLLAMA_MODEL=qwen2.5-coder:3b\n", encoding="utf-8")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    assert Settings(_env_file=env_file).ollama_model == "qwen2.5-coder:3b"
