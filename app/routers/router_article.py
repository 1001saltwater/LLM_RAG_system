# app/routers/router_article.py
from fastapi import APIRouter
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
    