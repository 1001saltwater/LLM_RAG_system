import json

from app.database.session import sessionmaker
from app.services.service_article import ServiceArticle
from app.schemas.schema_article import CreateArticle
from app.config.config import settings

from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.embedding.pipeline import EmbeddingPipeline


def main():

    db = sessionmaker()

    article_service = ServiceArticle()
    ingestion_pipeline = IngestionPipeline()
    embedding_pipeline = EmbeddingPipeline()

    with open(
        "article_import/data/documents.json",
        "r",
        encoding="utf-8"
    ) as f:
        documents = json.load(f)


    for document in documents:

        # 1. 创建文章
        article = article_service.create_article(
            db,
            CreateArticle(
                title=document["title"],
                content=document["content"]
            )
        )

        print(
            f"Created article: {article.id}"
        )


        # 2. Chunk切分
        chunk_count = ingestion_pipeline.ingest(
            db,
            article.id
        )

        print(
            f"Created chunks: {chunk_count}"
        )


        # 3. 生成Embedding
        embedding_count = embedding_pipeline.embed_article(
            db,
            article.id
        )

        print(
            f"Created embeddings: {embedding_count}"
        )


    db.close()


if __name__ == "__main__":
    main()