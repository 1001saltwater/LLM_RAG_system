# app/services/service_chunk.py
from sqlalchemy.orm import Session
from app.models.model_chunk import Chunk
from app.schemas.schema_chunk import CreateChunk, UpdateChunk, ResponseChunk


class ServiceChunk:
    def create_chunk(self, db: Session, chunk_data: CreateChunk) -> ResponseChunk:
        db_chunk = self._build_chunk(chunk_data)
        db.add(db_chunk)
        try:
            db.commit()
            db.refresh(db_chunk)
        except Exception:
            db.rollback()
            raise
        return ResponseChunk.model_validate(db_chunk)

    def create_chunks_batch(
        self,
        db: Session,
        chunks_data: list[CreateChunk],
    ) -> list[ResponseChunk]:
        if not chunks_data:
            return []

        db_chunks = [self._build_chunk(chunk_data) for chunk_data in chunks_data]
        db.add_all(db_chunks)
        try:
            db.commit()
            for db_chunk in db_chunks:
                db.refresh(db_chunk)
        except Exception:
            db.rollback()
            raise

        return [ResponseChunk.model_validate(db_chunk) for db_chunk in db_chunks]

    @staticmethod
    def _build_chunk(chunk_data: CreateChunk) -> Chunk:
        return Chunk(
            article_id=chunk_data.article_id,
            content=chunk_data.content,
            chunk_index=chunk_data.chunk_index,
            page_number=chunk_data.page_number,
            chunk_metadata=chunk_data.chunk_metadata,
        )

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
        if db_chunk is None:
            return None

        if chunk_data.article_id is not None:
            db_chunk.article_id = chunk_data.article_id
        if chunk_data.content is not None:
            db_chunk.content = chunk_data.content
        if chunk_data.chunk_index is not None:
            db_chunk.chunk_index = chunk_data.chunk_index
        if chunk_data.page_number is not None:
            db_chunk.page_number = chunk_data.page_number
        if chunk_data.chunk_metadata is not None:
            db_chunk.chunk_metadata = chunk_data.chunk_metadata
        try:
            db.commit()
            db.refresh(db_chunk)
        except Exception:
            db.rollback()
            raise
        return ResponseChunk.model_validate(db_chunk)

    def delete_chunk(self, db: Session, chunk_id: int) -> ResponseChunk | None:
        db_chunk = self.get_chunk(db, chunk_id)   
        if db_chunk:
            response = ResponseChunk.model_validate(db_chunk)
            db.delete(db_chunk)
            try:
                db.commit()
            except Exception:
                db.rollback()
                raise
            return response
        return None
