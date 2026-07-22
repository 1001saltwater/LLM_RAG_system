from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.model_article import Article
from app.models.model_chunk import Chunk
from app.rag.ingestion.pipeline.langchain_pipeline import IngestionPipeline
from app.schemas.schema_chunk import CreateChunk
from app.services.service_chunk import ServiceChunk


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


def create_article(db: Session, storage_path: str) -> Article:
    article = Article(
        file_name="test.pdf",
        file_size=9,
        storage_path=storage_path,
    )
    db.add(article)
    db.commit()
    db.refresh(article)
    return article


def test_ingest_preserves_order_page_metadata_and_rejects_duplicate(
    db: Session,
    tmp_path,
):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-test")
    article = create_article(db, str(pdf_path))

    documents = [
        Document(page_content="page one, chunk one", metadata={"page": 0}),
        Document(page_content="page one, chunk two", metadata={"page": 0}),
        Document(page_content="page two", metadata={"page": 1}),
    ]
    pipeline = IngestionPipeline()
    pipeline.pdf_loader = SimpleNamespace(load=lambda _data: documents)
    pipeline.pdf_cleaner = SimpleNamespace(clean=lambda loaded: loaded)
    pipeline.text_splitter = SimpleNamespace(split=lambda cleaned: cleaned)

    assert pipeline.ingest(db, article.id) == 3

    chunks = (
        db.query(Chunk)
        .filter(Chunk.article_id == article.id)
        .order_by(Chunk.chunk_index)
        .all()
    )
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 2]
    assert all(chunk.content for chunk in chunks)
    for chunk in chunks:
        assert chunk.chunk_metadata["article_id"] == article.id
        assert chunk.chunk_metadata["source"] == str(pdf_path)
        assert chunk.chunk_metadata["page"] == chunk.page_number - 1

    with pytest.raises(HTTPException) as exc_info:
        pipeline.ingest(db, article.id)

    assert exc_info.value.status_code == 409
    assert (
        db.query(Chunk).filter(Chunk.article_id == article.id).count()
        == len(chunks)
    )


def test_batch_insert_rolls_back_all_chunks_on_unique_constraint_failure(
    db: Session,
    tmp_path,
):
    article = create_article(db, str(tmp_path / "test.pdf"))
    chunks_data = [
        CreateChunk(
            article_id=article.id,
            content="first",
            chunk_index=0,
            page_number=1,
            chunk_metadata={},
        ),
        CreateChunk(
            article_id=article.id,
            content="duplicate index",
            chunk_index=0,
            page_number=1,
            chunk_metadata={},
        ),
    ]

    with pytest.raises(IntegrityError):
        ServiceChunk().create_chunks_batch(db, chunks_data)

    assert db.query(Chunk).filter(Chunk.article_id == article.id).count() == 0
