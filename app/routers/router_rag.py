import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.config.config import settings
from app.observability.langsmith import langsmith_tracing_context
from app.rag.generation.langchain_generator import LangChainGenerator
from app.schemas.schema_rag import RAGQueryRequest


router = APIRouter(prefix="/rag", tags=["rag"])


def format_sse_event(event: str, data: dict) -> str:
    return (
        f"event: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


@router.post("/query", response_class=StreamingResponse)
async def query(request: RAGQueryRequest):
    generator = LangChainGenerator()

    def prepare_context():
        return generator.prepare(
            question=request.question,
            top_k=request.top_k,
            max_distance=request.max_distance,
            rerank_top_k=request.rerank_top_k,
            rerank_threshold=request.rerank_threshold,
        )

    async def event_stream():
        with langsmith_tracing_context(
            tags=["rag", "api", "stream"],
            metadata={
                "embedding_model": settings.EMBEDDING_MODEL,
                "rerank_model": settings.RERANK_MODEL,
                "llm_model": settings.LLM_MODEL,
            },
        ):
            prepared = await run_in_threadpool(prepare_context)
            async for token in generator.astream(
                request.question,
                prepared,
            ):
                yield format_sse_event("token", {"content": token})

            yield format_sse_event(
                "sources",
                {
                    "sources": [
                        source.model_dump()
                        for source in prepared.sources
                    ]
                },
            )
            yield format_sse_event("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
