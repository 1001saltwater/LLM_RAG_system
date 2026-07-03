# app/models/model_article_chunks.py
from app.models.base import Base
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy import ForeignKey, Text, DateTime
import datetime

class Chunk(Base):
    __tablename__ = "chunk"

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    article_id : Mapped[int] = mapped_column(ForeignKey("article.id"), index=True)
    content : Mapped[str] = mapped_column(Text, index=True)
    created_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now)
    updated_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now, onupdate=datetime.datetime.now)
