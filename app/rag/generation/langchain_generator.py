from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langsmith import traceable

from app.config.config import settings
from app.database.session import session_scope
from app.rag.generation.langchain_LLM import LangChainLLM
from app.rag.generation.langchain_prompt import (
    LangChainPrompt,
    format_context,
)
from app.rag.rerank.langchain_rerank import LangChainReranker
from app.rag.retrieval.langchain_retriever import LangChainRetriever
from app.schemas.schema_rag import RAGGenerationResult, RAGSource


NO_CONTEXT_ANSWER = "根据已有资料无法回答该问题。"


@dataclass(frozen=True)
class PreparedRAGContext:
    documents: list[Document]
    sources: list[RAGSource]


def sanitize_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value
        for key, value in inputs.items()
        if key not in {"self", "db", "prepared"}
    }
    prepared = inputs.get("prepared")
    if isinstance(prepared, PreparedRAGContext):
        sanitized["document_count"] = len(prepared.documents)
    return sanitized


def serialize_trace_output(output: Any) -> Any:
    if isinstance(output, RAGGenerationResult):
        return output.model_dump()
    return output


class LangChainGenerator:
    def __init__(
        self,
        prompt: LangChainPrompt | None = None,
        llm: LangChainLLM | None = None,
        retriever_factory: Callable[..., BaseRetriever] = LangChainRetriever,
        reranker_factory: Callable[
            ..., BaseDocumentCompressor
        ] = LangChainReranker,
    ):
        self.prompt = prompt or LangChainPrompt()
        self.llm = llm or LangChainLLM()
        self.retriever_factory = retriever_factory
        self.reranker_factory = reranker_factory
        self.chain = (
            self.prompt.template
            | self.llm.runnable
            | StrOutputParser()
        )

    @traceable(
        name="rag-query",
        run_type="chain",
        process_inputs=sanitize_trace_inputs,
        process_outputs=serialize_trace_output,
    )
    def generate(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        max_distance: float = settings.THRESHOLD,
        rerank_top_k: int = settings.RERANK_TOP_K,
        rerank_threshold: float | None = settings.RERANK_THRESHOLD,
    ) -> RAGGenerationResult:
        prepared = self.prepare(
            question=question,
            top_k=top_k,
            max_distance=max_distance,
            rerank_top_k=rerank_top_k,
            rerank_threshold=rerank_threshold,
        )
        question = question.strip()
        if not prepared.documents:
            return RAGGenerationResult(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        answer = self.chain.invoke(
            {
                "question": question,
                "context": format_context(prepared.documents),
            }
        ).strip()
        return RAGGenerationResult(
            answer=answer,
            sources=prepared.sources,
        )

    def prepare(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        max_distance: float = settings.THRESHOLD,
        rerank_top_k: int = settings.RERANK_TOP_K,
        rerank_threshold: float | None = settings.RERANK_THRESHOLD,
    ) -> PreparedRAGContext:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")

        documents = self.retrieve(
            question=question,
            top_k=top_k,
            max_distance=max_distance,
        )
        documents = self.rerank(
            question=question,
            documents=documents,
            rerank_top_k=rerank_top_k,
            rerank_threshold=rerank_threshold,
        )
        documents = self._limit_documents(
            documents,
            settings.RAG_MAX_CONTEXT_CHARS,
        )
        return PreparedRAGContext(
            documents=documents,
            sources=self._build_sources(documents),
        )

    def retrieve(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        max_distance: float = settings.THRESHOLD,
    ) -> list[Document]:
        with session_scope() as db:
            base_retriever = self.retriever_factory(
                db=db,
                top_k=top_k,
                max_distance=max_distance,
            )
            return list(base_retriever.invoke(question))

    def rerank(
        self,
        question: str,
        documents: Sequence[Document],
        rerank_top_k: int = settings.RERANK_TOP_K,
        rerank_threshold: float | None = settings.RERANK_THRESHOLD,
    ) -> list[Document]:
        reranker = self.reranker_factory(
            top_n=rerank_top_k,
            score_threshold=rerank_threshold,
        )
        return list(reranker.compress_documents(documents, question))

    @traceable(
        name="rag-query-stream",
        run_type="chain",
        process_inputs=sanitize_trace_inputs,
    )
    async def astream(
        self,
        question: str,
        prepared: PreparedRAGContext,
    ) -> AsyncIterator[str]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")

        if not prepared.documents:
            yield NO_CONTEXT_ANSWER
            return

        async for chunk in self.chain.astream(
            {
                "question": question,
                "context": format_context(prepared.documents),
            }
        ):
            if chunk:
                yield chunk

    @staticmethod
    def _limit_documents(
        documents: Sequence[Document],
        max_characters: int,
    ) -> list[Document]:
        if max_characters < 1:
            raise ValueError("max_characters must be at least 1")

        limited_documents = []
        remaining = max_characters
        for document in documents:
            content = document.page_content.strip()
            if not content:
                continue
            content = content[:remaining]
            limited_documents.append(
                Document(
                    page_content=content,
                    metadata=document.metadata,
                )
            )
            remaining -= len(content)
            if remaining == 0:
                break
        return limited_documents

    @staticmethod
    def _build_sources(documents: Sequence[Document]) -> list[RAGSource]:
        return [
            RAGSource(
                article_id=document.metadata["article_id"],
                chunk_id=document.metadata["chunk_id"],
                page_number=document.metadata["page_number"],
                distance=document.metadata["distance"],
                rerank_score=document.metadata.get("rerank_score"),
            )
            for document in documents
        ]
