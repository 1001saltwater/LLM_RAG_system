# app/database/session.py
from sqlalchemy import create_engine
from app.config.config import settings
from sqlalchemy.orm import sessionmaker

DATABASE_URL = f"postgresql://{settings.DATABASE_USER}:{settings.DATABASE_PASSWORD}@{settings.DATABASE_HOST}:{settings.DATABASE_PORT}/{settings.DATABASE_NAME}"
engine = create_engine(DATABASE_URL)
sessionmaker = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = sessionmaker()
    try:
        yield db
    finally:
        db.close()