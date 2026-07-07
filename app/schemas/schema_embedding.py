# app/schemas/schema_embedding.py
from pydantic import BaseModel, ConfigDict, field_validator
from typing import Any

class BaseEmbedding(BaseModel):
    chunk_id: int
    embedding: list[float]

class CreateEmbedding(BaseEmbedding):
    pass

class UpdateEmbedding(BaseEmbedding):
    chunk_id: int|None = None
    embedding: list[float]|None = None

class ResponseEmbedding(BaseEmbedding):
    id: int

    model_config = ConfigDict(from_attributes=True)

    # @field_validator('embedding', mode='before')
    # @classmethod
    # def convert_vector_to_list(cls, v: Any) -> list[float]:
    #     if isinstance(v, list):
    #         return v
    #     try:
    #         return list(v)
    #     except (TypeError, ValueError):
    #         raise ValueError("embedding must be convertible to list[float]")