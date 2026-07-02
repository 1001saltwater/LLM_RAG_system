# app/services/service_article.py
from sqlalchemy.orm import Session
from app.models.model_article import Article
from app.schemas.schema_article import CreateArticle, UpdateArticle, ResponseArticle

class ServiceArticle:
    def create_article(self,db:Session,article_data:CreateArticle) -> Article:
        db_article = Article(title=article_data.title,content=article_data.content)
        db.add(db_article)
        db.commit()
        db.refresh(db_article)
        return db_article
