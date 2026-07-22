# tests/test_embedding_pipeline.py

from app.database.session import sessionmaker
from app.rag.embedding.pipeline import EmbeddingPipeline


def test_embedding_pipeline():

    db = sessionmaker()

    try:
        pipeline = EmbeddingPipeline()

        count = pipeline.embed_article(
            db=db,
            article_id=3
        )

        print(f"Generated {count} embeddings")

        assert isinstance(count, int)
        assert count > 0

    finally:
        db.close()