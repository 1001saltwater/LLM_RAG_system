from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime

class BaseChunk(BaseModel):
    article_id: int
    content: str
    chunk_index: int
    page_number: int
    chunk_metadata: dict = Field(default_factory=dict)

class CreateChunk(BaseChunk):
    pass

class UpdateChunk(BaseChunk):
    article_id: int|None = None
    content: str|None = None
    chunk_index: int|None = None
    page_number: int|None = None
    chunk_metadata: dict|None = None

class ResponseChunk(BaseChunk):
    id: int
    created_time: datetime
    updated_time: datetime

    model_config = ConfigDict(from_attributes=True)
