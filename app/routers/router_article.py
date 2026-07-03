# app/routers/router_article.py
from fastapi import APIRouter, HTTPException
from app.services.service_article import ServiceArticle
from app.schemas.schema_article import CreateArticle, UpdateArticle, ResponseArticle
from sqlalchemy.orm import Session
from app.database.session import sessionmaker, get_db
from fastapi import Depends

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("/", response_model=ResponseArticle)
def create_article(article_data: CreateArticle, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    return service_article.create_article(db, article_data)

@router.get("/{article_id}", response_model=ResponseArticle)
def get_article(article_id: int, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.get_article(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article

@router.get("/", response_model=list[ResponseArticle])
def get_all_article(db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    return service_article.get_all_article(db)

@router.get("/article_id/{article_id}", response_model=list[ResponseArticle])
def get_article_by_id(article_id: int, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    return service_article.get_article_by_id(db, article_id)

@router.put("/{article_id}", response_model=ResponseArticle)
def update_article(article_id: int, article_data: UpdateArticle, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.update_article(db, article_id, article_data)
    if not article:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@router.delete("/{article_id}", response_model=ResponseArticle)
def delete_article(article_id: int, db: Session = Depends(get_db)):
    service_article = ServiceArticle()
    article = service_article.delete_article(db, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return  article
