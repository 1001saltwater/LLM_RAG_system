# app/routers/router_article.py
from app.rag.ingestion.pipeline.langchain_pipeline import IngestionPipeline
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.services.service_article import ServiceArticle
from app.schemas.schema_article import CreateArticle, UpdateArticle, ResponseArticle, UploadPDFResponse
from sqlalchemy.orm import Session
from app.database.session import get_db
from fastapi import Depends
from app.config.config import settings

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post("/", response_model=ResponseArticle)
def create_article(article_data: CreateArticle, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    return service_article.create_article(db, article_data)

@router.post("/upload-pdf", response_model=UploadPDFResponse)
async def upload_pdf_article(
    file: UploadFile = File(...),
    auto_ingest: bool = Form(default=False),
    db: Session = Depends(get_db),
):
    if file.content_type not in (None, "application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_bytes) > settings.MAX_PDF_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"PDF file exceeds {settings.MAX_PDF_SIZE_BYTES / 1024 / 1024} MB limit")   

    service_article = ServiceArticle()
    article = service_article.create_article_from_pdf(
        db=db,
        file_bytes=file_bytes,
        filename=file.filename or "upload.pdf"
    )

    chunk_count = None
    if auto_ingest:
        pipeline = IngestionPipeline()
        chunk_count = pipeline.ingest(db, article.id)

    return UploadPDFResponse(article=article, chunk_count=chunk_count)

@router.post("/{article_id}/ingest", response_model=int)
def ingest_article(article_id: int, db: Session = Depends(get_db)):
    pipeline = IngestionPipeline()
    return pipeline.ingest(db, article_id)

@router.get("/{article_id}", response_model=ResponseArticle)
def get_article_by_id(article_id: int, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.get_article_by_id(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@router.get("/", response_model=list[ResponseArticle])
def get_all_article(db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    return service_article.get_all_article(db)

@router.put("/{article_id}", response_model=ResponseArticle)
def update_article(article_id: int, article_data: UpdateArticle, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.update_article(db, article_id, article_data)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@router.delete("/{article_id}", response_model=ResponseArticle)
def delete_article(article_id: int, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.delete_article(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return  article
