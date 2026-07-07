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

    VECTOR_DB: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()