# app/database/init_db.py
from app.models.base import Base
from app.database.session import engine

from app.models.model_article import Article
from app.models.model_chunk import Chunk
from app.models.model_embedding import Embedding


Base.metadata.create_all(bind=engine)