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

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()