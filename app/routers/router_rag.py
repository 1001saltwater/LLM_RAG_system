from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.rag.generation.langchain_generator import LangChainGenerator
from app.schemas.schema_rag import RAGGenerationResult, RAGQueryRequest


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGGenerationResult)
def query(
    request: RAGQueryRequest,
    db: Session = Depends(get_db),
):
    generator = LangChainGenerator()
    return generator.generate(
        db=db,
        question=request.question,
        top_k=request.top_k,
        max_distance=request.max_distance,
        rerank_top_k=request.rerank_top_k,
        rerank_threshold=request.rerank_threshold,
    )
