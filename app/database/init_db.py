# app/database/init_db.py
from app.models.base import Base
from app.database.session import engine

from app.models.model_article import Article
from app.models.model_chunk import Chunk

Base.metadata.create_all(bind=engine)