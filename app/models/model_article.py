# app/models/model_article.py
from app.models.base import Base
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy import String, Text, DateTime
import datetime

class Article(Base):
    __tablename__ = "article"

    id : Mapped[int] = mapped_column(primary_key=True, index=True)
    file_name : Mapped[str] = mapped_column(String(255), index=True)
    file_size : Mapped[int] = mapped_column(index=True)
    storage_path : Mapped[str] = mapped_column(String(255), index=True)
    created_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now)
    updated_time : Mapped[datetime.datetime] = mapped_column(DateTime,default=datetime.datetime.now, onupdate=datetime.datetime.now)
