# app/schemas/schema_article.py
from pydantic import BaseModel, ConfigDict
from datetime import datetime

class BaseArticle(BaseModel):
    title: str
    content: str

class CreateArticle(BaseArticle):
    pass

class UpdateArticle(BaseArticle):
    title: str|None = None
    content: str|None = None

class ResponseArticle(BaseArticle):
    id: int
    created_time: datetime
    updated_time: datetime

    model_config = ConfigDict(from_attributes=True)