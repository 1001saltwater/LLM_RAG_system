from pydantic import BaseModel, ConfigDict
from datetime import datetime

class BaseChunk(BaseModel):
    article_id: int
    content: str

class CreateChunk(BaseChunk):
    pass

class UpdateChunk(BaseChunk):
    article_id: int|None = None
    content: str|None = None

class ResponseChunk(BaseChunk):
    id: int
    created_time: datetime
    updated_time: datetime

    model_config = ConfigDict(from_attributes=True)
