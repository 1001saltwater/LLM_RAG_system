# app/services/service_chunk.py
from sqlalchemy.orm import Session
from app.models.model_chunk import Chunk
from app.schemas.schema_chunk import CreateChunk, UpdateChunk, ResponseChunk
from datetime import datetime


class ServiceChunk:
    def create_chunk(self, db: Session, chunk_data: CreateChunk) -> ResponseChunk:
        db_chunk = Chunk(article_id=chunk_data.article_id, content=chunk_data.content,created_time=datetime.now(),updated_time=datetime.now())
        db.add(db_chunk)
        db.commit() 
        db.refresh(db_chunk)
        return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content,created_time=db_chunk.created_time,updated_time=db_chunk.updated_time)

    def get_chunk(self, db: Session, chunk_id: int) -> ResponseChunk | None:
        return db.query(Chunk).filter(Chunk.id == chunk_id).first()
    
    def get_all_chunk(self, db: Session) -> list[ResponseChunk]:
        return db.query(Chunk).all()
        
    def get_chunk_by_id(self, db: Session, chunk_id: int) -> ResponseChunk | None:        
        return db.query(Chunk).filter(Chunk.id == chunk_id).first()

    def get_chunk_by_article_id(self, db: Session, article_id: int) -> list[ResponseChunk]:
        return db.query(Chunk).filter(Chunk.article_id == article_id).all()

    def get_chunk_by_id_batch(self,db: Session,chunk_ids: list[int]):
        db_chunks = (db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all())
        chunk_map = {chunk.id: chunk for chunk in db_chunks}
        return [ResponseChunk.model_validate(chunk_map[chunk_id]) for chunk_id in chunk_ids if chunk_id in chunk_map]

    def update_chunk(self, db: Session, chunk_id: int, chunk_data: UpdateChunk) -> ResponseChunk | None:
        db_chunk = self.get_chunk(db, chunk_id)
        if db_chunk:
            if chunk_data.article_id is not None:
                db_chunk.article_id = chunk_data.article_id
            if chunk_data.content is not None:
                db_chunk.content = chunk_data.content
            db.commit()
            db.refresh(db_chunk)
        return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content,created_time=db_chunk.created_time,updated_time=db_chunk.updated_time)

    def delete_chunk(self, db: Session, chunk_id: int) -> ResponseChunk | None:
        db_chunk = self.get_chunk(db, chunk_id)   
        if db_chunk:
            db.delete(db_chunk)
            db.commit()
            return ResponseChunk(id=db_chunk.id, article_id=db_chunk.article_id, content=db_chunk.content,created_time=db_chunk.created_time,updated_time=db_chunk.updated_time)
        return None
