from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str

    PDF_STORAGE_DIRECTORY: str
    MAX_PDF_SIZE_BYTES: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    EMBEDDING_MODEL: str
    EMBEDDING_DIM: int
    EMBEDDING_BATCH_SIZE: int
    EMBEDDING_QUERY_INSTRUCTION: str = "为这个句子生成表示以用于检索相关文章："

    DEVICE: str

    TOP_K: int
    THRESHOLD: float

    VECTOR_DB: str

    LLM_MODEL: str
    LLM_API_KEY: str
    LLM_API_URL: str
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 2

    MAX_TOKENS: int
    TEMPERATURE: float
    RAG_MAX_CONTEXT_CHARS: int = 6000

    RERANK_MODEL: str
    RERANK_TOP_K: int
    RERANK_THRESHOLD: float

    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str = "fastapi-rag-dev"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_HIDE_INPUTS: bool = False
    LANGSMITH_HIDE_OUTPUTS: bool = False

    @model_validator(mode="after")
    def validate_langsmith_configuration(self):
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY is None:
            raise ValueError(
                "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled"
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()