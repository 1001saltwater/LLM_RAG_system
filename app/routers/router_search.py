from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.rag.retrieval.retriever import Retriever
from app.schemas.schema_search import SearchRequest, SearchResponse


router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    search_request: SearchRequest,
    db: Session = Depends(get_db),
):
    retriever = Retriever()
    results = retriever.retrieve(
        db=db,
        question=search_request.question,
        top_k=search_request.top_k,
        max_distance=search_request.max_distance,
    )
    return SearchResponse(
        question=search_request.question,
        results=results,
    )
