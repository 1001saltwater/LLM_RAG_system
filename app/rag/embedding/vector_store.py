from typing import Any

from langsmith import traceable
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.config.config import settings
from app.models.model_chunk import Chunk
from app.models.model_embedding import Embedding
from app.schemas.schema_search import SearchResult


def sanitize_vector_search_inputs(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    query_vector = inputs.get("query_vector") or []
    return {
        "top_k": inputs.get("top_k"),
        "max_distance": inputs.get("max_distance"),
        "query_vector_dimension": len(query_vector),
    }


def summarize_vector_search_outputs(
    outputs: Any,
) -> dict[str, Any]:
    if not isinstance(outputs, list):
        return {"result": outputs}
    return {
        "result_count": len(outputs),
        "results": [
            {
                "chunk_id": result.chunk_id,
                "distance": result.distance,
            }
            for result in outputs
            if isinstance(result, SearchResult)
        ],
    }


class VectorStore:
    @traceable(
        name="pgvector-search",
        run_type="retriever",
        process_inputs=sanitize_vector_search_inputs,
        process_outputs=summarize_vector_search_outputs,
    )
    def similarity_search(
        self,
        db: Session,
        query_vector: list[float],
        top_k: int | None = None,
        max_distance: float | None = None,
    ) -> list[SearchResult]:
        statement = self.build_similarity_statement(
            query_vector=query_vector,
            top_k=settings.TOP_K if top_k is None else top_k,
            max_distance=(
                settings.THRESHOLD if max_distance is None else max_distance
            ),
        )
        rows = db.execute(statement).mappings().all()

        return [
            SearchResult(
                article_id=row["article_id"],
                chunk_id=row["chunk_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                page_number=row["page_number"],
                chunk_metadata=row["chunk_metadata"],
                distance=float(row["distance"]),
            )
            for row in rows
        ]

    @staticmethod
    def build_similarity_statement(
        query_vector: list[float],
        top_k: int,
        max_distance: float,
    ) -> Select:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 <= max_distance <= 2.0:
            raise ValueError("max_distance must be between 0 and 2")

        distance = Embedding.embedding.cosine_distance(query_vector)
        return (
            select(
                Chunk.article_id,
                Chunk.id.label("chunk_id"),
                Chunk.content,
                Chunk.chunk_index,
                Chunk.page_number,
                Chunk.chunk_metadata,
                distance.label("distance"),
            )
            .join(Embedding, Embedding.chunk_id == Chunk.id)
            .where(distance <= max_distance)
            .order_by(distance.asc())
            .limit(top_k)
        )