# app/services/service_article.py
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session
from app.config.config import settings
from app.models.model_article import Article
from app.rag.ingestion.loaders.pdf_loader import PDFLoader
from app.schemas.schema_article import CreateArticle, UpdateArticle

class ServiceArticle:
    def create_article(self, db: Session, article_data: CreateArticle) -> Article:
        db_article = Article( file_name=article_data.file_name, file_size=article_data.file_size, storage_path=article_data.storage_path)
        db.add(db_article)
        db.commit()
        db.refresh(db_article)
        return db_article

    def create_article_from_pdf(self, db: Session, file_bytes: bytes, filename: str,) -> Article:
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        if not file_bytes.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="Invalid PDF file")

        documents = PDFLoader().load(file_bytes)
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in PDF. Scanned PDFs are not supported yet.",
            )

        # 存储文件到指定目录
        storage_dir = Path(f"{settings.PDF_STORAGE_DIRECTORY}")
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / filename
        with open(storage_path, "wb") as f:
            f.write(file_bytes)

        article_title = Path(filename).stem
        return self.create_article(db, CreateArticle(title=article_title, file_name=filename, file_size=len(file_bytes), storage_path=str(storage_path)))

    def get_article_by_id(self, db: Session, article_id: int) -> Article | None:
        return db.query(Article).filter(Article.id == article_id).first()

    def get_all_article(self, db: Session) -> list[Article]:
        return db.query(Article).all()

    def update_article(self, db: Session, article_id: int, article_data: UpdateArticle) -> Article | None:
        db_article = self.get_article_by_id(db, article_id)
        if db_article:
            if article_data.title is not None:
                db_article.title = article_data.title
            if article_data.content is not None:
                db_article.content = article_data.content
            db.commit()
            db.refresh(db_article)
        return db_article

    def delete_article(self, db: Session, article_id: int) -> bool:
        db_article = self.get_article_by_id(db, article_id)
        if db_article:
            db.delete(db_article)
            db.commit()
            return True
        return False
