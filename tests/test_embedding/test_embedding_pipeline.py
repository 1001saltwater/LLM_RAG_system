import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config.config import settings
from app.models.base import Base
from app.models.model_article import Article
from app.models.model_chunk import Chunk
from app.models.model_embedding import Embedding
from app.rag.embedding.pipeline import EmbeddingPipeline


class FakeEmbedder:
    def __init__(self, dimension: int = settings.EMBEDDING_DIM):
        self.dimension_value = dimension
        self.batch_sizes: list[int] = []

    def dimension(self) -> int:
        return self.dimension_value

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[0.1] * self.dimension_value for _text in texts]


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


def create_article_with_chunks(db: Session, chunk_count: int) -> Article:
    article = Article(
        file_name="test.pdf",
        file_size=9,
        storage_path="test.pdf",
    )
    db.add(article)
    db.flush()

    db.add_all(
        [
            Chunk(
                article_id=article.id,
                content=f"chunk {chunk_index}",
                chunk_index=chunk_index,
                page_number=1,
                chunk_metadata={},
            )
            for chunk_index in range(chunk_count)
        ]
    )
    db.commit()
    db.refresh(article)
    return article


def test_embed_article_batches_chunks_and_rejects_duplicate(db: Session):
    article = create_article_with_chunks(
        db,
        settings.EMBEDDING_BATCH_SIZE + 1,
    )
    embedder = FakeEmbedder()
    pipeline = EmbeddingPipeline(embedder=embedder)

    assert pipeline.embed_article(db, article.id) == settings.EMBEDDING_BATCH_SIZE + 1
    assert embedder.batch_sizes == [settings.EMBEDDING_BATCH_SIZE, 1]
    assert (
        db.query(Embedding)
        .join(Chunk, Embedding.chunk_id == Chunk.id)
        .filter(Chunk.article_id == article.id)
        .count()
        == settings.EMBEDDING_BATCH_SIZE + 1
    )

    with pytest.raises(HTTPException) as exc_info:
        pipeline.embed_article(db, article.id)
    assert exc_info.value.status_code == 409


def test_embed_article_rejects_dimension_mismatch_without_writing(db: Session):
    article = create_article_with_chunks(db, 1)
    pipeline = EmbeddingPipeline(
        embedder=FakeEmbedder(settings.EMBEDDING_DIM + 1)
    )

    with pytest.raises(HTTPException) as exc_info:
        pipeline.embed_article(db, article.id)

    assert exc_info.value.status_code == 500
    assert db.query(Embedding).count() == 0


def test_embed_article_requires_chunks(db: Session):
    article = create_article_with_chunks(db, 0)
    pipeline = EmbeddingPipeline(embedder=FakeEmbedder())

    with pytest.raises(HTTPException) as exc_info:
        pipeline.embed_article(db, article.id)

    assert exc_info.value.status_code == 409
    assert db.query(Embedding).count() == 0
