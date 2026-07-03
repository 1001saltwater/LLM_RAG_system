from fastapi import APIRouter, HTTPException
from app.services.service_chunk import ServiceChunk
from app.schemas.schema_chunk import CreateChunk, UpdateChunk, ResponseChunk
from sqlalchemy.orm import Session
from app.database.session import get_db
from fastapi import Depends

router = APIRouter(prefix="/chunks", tags=["chunks"])

@router.post("/", response_model=ResponseChunk)
def create_chunk(chunk_data: CreateChunk, db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    return service_chunk.create_chunk(db, chunk_data)

@router.get("/{chunk_id}", response_model=ResponseChunk)
def get_chunk(chunk_id: int, db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    chunk = service_chunk.get_chunk(db, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@router.get("/", response_model=list[ResponseChunk])
def get_all_chunk(db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    return service_chunk.get_all_chunk(db)

@router.get("/chunk_id/{chunk_id}", response_model=list[ResponseChunk])
def get_chunk_by_id(chunk_id: int, db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    return service_chunk.get_chunk_by_id(db, chunk_id)

@router.put("/{chunk_id}", response_model=ResponseChunk)
def update_chunk(chunk_id: int, chunk_data: UpdateChunk, db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    chunk = service_chunk.update_chunk(db, chunk_id, chunk_data)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk

@router.delete("/{chunk_id}", response_model=ResponseChunk)
def delete_chunk(chunk_id: int, db: Session = Depends(get_db)):
    service_chunk = ServiceChunk()
    chunk = service_chunk.delete_chunk(db, chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return  chunk
