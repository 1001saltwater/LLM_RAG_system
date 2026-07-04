# app/services/service_article.py
from sqlalchemy.orm import Session
from app.models.model_article import Article
from app.schemas.schema_article import CreateArticle, UpdateArticle, ResponseArticle

class ServiceArticle:
    def create_article(self, db: Session, article_data: CreateArticle) -> Article:
        db_article = Article(title=article_data.title, content=article_data.content)
        db.add(db_article)
        db.commit()
        db.refresh(db_article)
        return db_article

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
