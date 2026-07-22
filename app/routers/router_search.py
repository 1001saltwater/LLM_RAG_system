from fastapi import APIRouter, Depends
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.rag.rerank.langchain_rerank import LangChainReranker
from app.rag.retrieval.langchain_retriever import LangChainRetriever
from app.schemas.schema_search import SearchRequest, SearchResponse, SearchResult


router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(
    search_request: SearchRequest,
    db: Session = Depends(get_db),
):
    retriever = LangChainRetriever(
        db=db,
        top_k=search_request.top_k,
        max_distance=search_request.max_distance,
    )
    if search_request.rerank:
        reranker = LangChainReranker(
            top_n=search_request.rerank_top_k,
            score_threshold=search_request.rerank_threshold,
        )
        compression_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=reranker,
        )
        documents = compression_retriever.invoke(search_request.question)
    else:
        documents = retriever.invoke(search_request.question)

    results = [
        SearchResult(
            content=document.page_content,
            **document.metadata,
        )
        for document in documents
    ]
    return SearchResponse(
        question=search_request.question,
        results=results,
    )
