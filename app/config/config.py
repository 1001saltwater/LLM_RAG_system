from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_HOST: str
    DATABASE_PORT: int
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str

    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    EMBEDDING_MODEL: str
    EMBEDDING_DIM: int
    EMBEDDING_BATCH_SIZE: int

    DEVICE: str

    TOP_K: int
    THRESHOLD: float

    VECTOR_DB: str

    LLM_MODEL: str
    LLM_API_KEY: str
    LLM_API_URL: str

    MAX_TOKENS: int
    TEMPERATURE: float

    RERANK_MODEL: str
    RERANK_TOP_K: int
    RERANK_THRESHOLD: float

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()