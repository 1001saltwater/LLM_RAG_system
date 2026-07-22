from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.rag.ingestion.cleaners.pdf_cleaner import PDFCleaner
from app.rag.ingestion.loaders.pdf_loader import PDFLoader
from app.rag.ingestion.splitter.LangChain_splitter import TextSplitter
from app.schemas.schema_chunk import CreateChunk
from app.services.service_article import ServiceArticle
from app.services.service_chunk import ServiceChunk

class IngestionPipeline:
    def __init__(self):
        self.article_service = ServiceArticle()
        self.chunk_service = ServiceChunk()
        self.pdf_loader = PDFLoader()
        self.pdf_cleaner = PDFCleaner()
        self.text_splitter = TextSplitter()

    def ingest(self, db: Session, article_id: int) -> int:
        article = self.article_service.get_article_by_id(db, article_id)
        if article is None:
            raise HTTPException(status_code=404, detail="Article not found")

        storage_path = Path(article.storage_path)
        if not storage_path.is_file():
            raise HTTPException(status_code=404, detail="PDF file not found")

        documents = self.pdf_loader.load(storage_path.read_bytes())
        documents = self.pdf_cleaner.clean(documents)
        chunk_documents = self.text_splitter.split(documents)

        chunks_data = []
        for chunk_index, document in enumerate(chunk_documents):
            metadata = dict(document.metadata)
            page_number = int(metadata.get("page", 0)) + 1
            metadata.update(
                {
                    "article_id": article_id,
                    "source": str(storage_path),
                }
            )
            chunks_data.append(
                CreateChunk(
                    article_id=article_id,
                    content=document.page_content,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    chunk_metadata=metadata,
                )
            )

        self.chunk_service.create_chunks_batch(db=db, chunks_data=chunks_data)
        return len(chunks_data)
