# app/schemas/schema_article.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class BaseArticle(BaseModel):
    file_name: str
    file_size: int
    storage_path: str

class CreateArticle(BaseArticle):
    pass

class UpdateArticle(BaseArticle):
    file_name: str|None = None
    file_size: int|None = None
    storage_path: str|None = None

class ResponseArticle(BaseArticle):
    id: int
    created_time: datetime
    updated_time: datetime

    model_config = ConfigDict(from_attributes=True)

class UploadPDFResponse(BaseModel):
    article: ResponseArticle
    chunk_count: int | None = None