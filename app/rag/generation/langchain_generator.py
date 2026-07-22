from collections.abc import Callable, Sequence

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.generation.langchain_LLM import LangChainLLM
from app.rag.generation.langchain_prompt import (
    LangChainPrompt,
    format_context,
)
from app.rag.rerank.langchain_rerank import LangChainReranker
from app.rag.retrieval.langchain_retriever import LangChainRetriever
from app.schemas.schema_rag import RAGGenerationResult, RAGSource


NO_CONTEXT_ANSWER = "根据已有资料无法回答该问题。"


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

    def generate(
        self,
        db: Session,
        question: str,
        top_k: int = settings.TOP_K,
        max_distance: float = settings.THRESHOLD,
        rerank_top_k: int = settings.RERANK_TOP_K,
        rerank_threshold: float | None = settings.RERANK_THRESHOLD,
    ) -> RAGGenerationResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")

        base_retriever = self.retriever_factory(
            db=db,
            top_k=top_k,
            max_distance=max_distance,
        )
        reranker = self.reranker_factory(
            top_n=rerank_top_k,
            score_threshold=rerank_threshold,
        )
        retriever = ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,
        )
        documents = self._limit_documents(
            retriever.invoke(question),
            settings.RAG_MAX_CONTEXT_CHARS,
        )

        if not documents:
            return RAGGenerationResult(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
            )

        answer = self.chain.invoke(
            {
                "question": question,
                "context": format_context(documents),
            }
        ).strip()
        return RAGGenerationResult(
            answer=answer,
            sources=self._build_sources(documents),
        )

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
