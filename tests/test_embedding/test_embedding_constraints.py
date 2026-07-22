import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config.config import settings
from app.models.base import Base
from app.models.model_article import Article
from app.models.model_chunk import Chunk
from app.models.model_embedding import Embedding
from app.services.service_embedding import ServiceEmbedding


@pytest.fixture
def db() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def create_chunk(db: Session) -> tuple[Article, Chunk]:
    article = Article(
        file_name="test.pdf",
        file_size=9,
        storage_path="test.pdf",
    )
    db.add(article)
    db.flush()

    chunk = Chunk(
        article_id=article.id,
        content="content",
        chunk_index=0,
        page_number=1,
        chunk_metadata={},
    )
    db.add(chunk)
    db.commit()
    db.refresh(article)
    db.refresh(chunk)
    return article, chunk


def test_chunk_id_is_unique_and_failed_batch_is_rolled_back(db: Session):
    _article, chunk = create_chunk(db)
    vector = [0.1] * settings.EMBEDDING_DIM
    db.add_all(
        [
            Embedding(chunk_id=chunk.id, embedding=vector),
            Embedding(chunk_id=chunk.id, embedding=vector),
        ]
    )

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()

    assert db.query(Embedding).count() == 0


def test_deleting_chunk_cascades_embedding(db: Session):
    article, chunk = create_chunk(db)
    embedding = Embedding(
        chunk_id=chunk.id,
        embedding=[0.1] * settings.EMBEDDING_DIM,
    )
    db.add(embedding)
    db.commit()

    assert ServiceEmbedding().count_by_article_id(db, article.id) == 1

    db.delete(chunk)
    db.commit()

    assert db.query(Embedding).count() == 0
