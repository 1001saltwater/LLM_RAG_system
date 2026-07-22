from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.config.config import settings


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=settings.TOP_K, ge=1, le=100)
    max_distance: float = Field(default=settings.THRESHOLD, ge=0.0, le=2.0)
    rerank: bool = True
    rerank_top_k: int = Field(default=settings.RERANK_TOP_K, ge=1, le=100)
    rerank_threshold: float | None = Field(
        default=settings.RERANK_THRESHOLD,
        ge=0.0,
        le=1.0,
    )

    @field_validator("question")
    @classmethod
    def strip_and_validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SearchResult(BaseModel):
    article_id: int
    chunk_id: int
    content: str
    chunk_index: int
    page_number: int
    chunk_metadata: dict[str, Any]
    distance: float
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    question: str
    results: list[SearchResult]
