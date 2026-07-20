# app/rag/ingestion/pipeline.py
from sqlalchemy.orm import Session
from fastapi import HTTPException
from app.services.service_chunk import ServiceChunk
from app.services.service_article import ServiceArticle
from app.rag.ingestion.preprocessor import TextPreprocessor
from app.rag.ingestion.splitter.splitter import TextSplitter
from app.schemas.schema_chunk import CreateChunk
from app.config.config import settings



class IngestionPipeline:
    def __init__(self):
        self.article_service = ServiceArticle()
        self.chunk_service = ServiceChunk()

        self.preprocessor = TextPreprocessor()
        self.splitter = TextSplitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    def ingest(self, db: Session, article_id: int):
        article = self.article_service.get_article_by_id(db, article_id)

        if article is None:
            raise HTTPException(status_code=404, detail="Article not found")

        # 预处理和分块
        clean_text = self.preprocessor.preprocess(article.content)

        chunks = self.splitter.split(clean_text)

        for chunk_index, content in enumerate(chunks):
            self.chunk_service.create_chunk(db=db, chunk_data=CreateChunk(article_id=article_id, content=content, chunk_index=chunk_index, page_number=1, metadata={"article_id": article_id}))
        
        return len(chunks)