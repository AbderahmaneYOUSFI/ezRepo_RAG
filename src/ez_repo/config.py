"""Application settings, loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # The project's .env wins over OS variables: generic names such as OLLAMA_MODEL
        # are often set globally for other tools.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "change-me-please"
    neo4j_database: str = "neo4j"
    neo4j_batch_size: int = 500  # rows per write transaction

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:3b"
    ollama_timeout: float = 180.0  # seconds; a 3B model on CPU is slow on long prompts
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 4096  # tokens: prompt + context + answer
    ollama_num_predict: int = 512  # max answer tokens

    # Retrieval / question answering
    retrieval_top_k: int = 5  # vector search hits
    retrieval_max_symbols: int = 8  # symbols sent to the LLM
    retrieval_depth: int = 1  # graph expansion: 1 or 2 hops
    retrieval_min_score: float = 0.6  # Neo4j cosine score in [0, 1]: (1 + cos) / 2
    qa_max_context_chars: int = 6000

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Ingestion
    repos_dir: Path = Path("./data/repos")
    max_file_size: int = 500_000
    max_files: int = 2_000
    max_total_size: int = 20_000_000
    clone_timeout: int = 120

    # Background indexing (in memory: tasks are lost when the API restarts)
    index_max_concurrent: int = 1  # indexings running at once (RAM: embeddings + parsing)
    index_max_pending: int = 3  # queued indexings beyond that are refused
    task_history: int = 50  # finished tasks kept for GET /tasks

    # Backend / frontend
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_url: str = "http://127.0.0.1:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
