from sqlalchemy.orm import Session
from app.models.model_chunk import Chunk
from app.schemas.schema_chunk import CreateChunk, UpdateChunk, ResponseChunk

class ServiceChunk:
    def create_chunk(self, db: Session, chunk_data: CreateChunk) -> ResponseChunk:
        db_chunk = ArticleChunks(article_id=chunk_data.article_id, content=chunk_data.content)
        db.add(db_chunk)
        db.commit() 
        db.refresh(db_chunk)
        return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content)

    def get_chunk(self, db: Session, chunk_id: int) -> ResponseChunk | None:
        return db.query(Chunk).filter(Chunk.id == chunk_id).first()
    
    def get_all_chunks(self, db: Session) -> list[ResponseChunk]:
        return db.query(ArticleChunk).all()

    def get_chunk_by_id(self, db: Session, chunk_id: int) -> list[ResponseChunk]:        
        return db.query(Chunk).filter(Chunk.id == chunk_id).all()

    def update_chunk(self, db: Session, chunk_id: int, chunk_data: UpdateChunk) -> ResponseChunk | None:
        db_chunk = self.get_chunk(db, chunk_id)
        if db_chunk:
            if chunk_data.article_id is not None:
                db_chunk.article_id = chunk_data.article_id
            if chunk_data.content is not None:
                db_chunk.content = chunk_data.content
            db.commit()
            db.refresh(db_chunk)
        return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content)

    def delete_chunk(self, db: Session, chunk_id: int) -> ResponseChunk | None:
        db_chunk = self.get_chunk(db, chunk_id)   
        if db_chunk:
            db.delete(db_chunk)
            db.commit()
            return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content)
        return None
