import asyncio
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.rag.generation import langchain_generator as generator_module
from app.rag.generation.langchain_generator import (
    LangChainGenerator,
    NO_CONTEXT_ANSWER,
    PreparedRAGContext,
)


class FakeLLM:
    def __init__(self):
        self.inputs = []
        self.runnable = RunnableLambda(self.invoke)

    def invoke(self, prompt_value):
        self.inputs.append(prompt_value)
        return AIMessage(content="基于资料的回答。[资料1]")


class FakeRetriever:
    def __init__(self, documents: list[Document], session_state: dict):
        self.documents = documents
        self.session_state = session_state
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        assert self.session_state["open"]
        self.queries.append(query)
        return self.documents


class FakeReranker:
    def __init__(self, session_state: dict):
        self.session_state = session_state

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        assert not self.session_state["open"]
        return documents


def build_generator(monkeypatch, documents):
    retriever_calls = []
    reranker_calls = []
    session_state = {"open": False}
    db = object()
    retriever = FakeRetriever(documents, session_state)

    @contextmanager
    def fake_session_scope():
        session_state["open"] = True
        try:
            yield db
        finally:
            session_state["open"] = False

    def retriever_factory(**kwargs):
        retriever_calls.append(kwargs)
        return retriever

    def reranker_factory(**kwargs):
        reranker_calls.append(kwargs)
        return FakeReranker(session_state)

    monkeypatch.setattr(generator_module, "session_scope", fake_session_scope)
    llm = FakeLLM()
    generator = LangChainGenerator(
        llm=llm,
        retriever_factory=retriever_factory,
        reranker_factory=reranker_factory,
    )
    return generator, llm, retriever_calls, reranker_calls, db


def test_generator_builds_rag_answer_and_sources(monkeypatch):
    documents = [
        Document(
            page_content="参考内容",
            metadata={
                "article_id": 1,
                "chunk_id": 2,
                "page_number": 3,
                "distance": 0.2,
                "rerank_score": 0.9,
            },
        )
    ]
    generator, llm, retriever_calls, reranker_calls, db = build_generator(
        monkeypatch,
        documents,
    )

    result = generator.generate(
        question="  问题  ",
        top_k=10,
        max_distance=0.4,
        rerank_top_k=2,
        rerank_threshold=0.6,
    )

    assert result.answer == "基于资料的回答。[资料1]"
    assert result.sources[0].chunk_id == 2
    assert result.sources[0].rerank_score == 0.9
    assert retriever_calls == [
        {"db": db, "top_k": 10, "max_distance": 0.4}
    ]
    assert reranker_calls == [
        {"top_n": 2, "score_threshold": 0.6}
    ]
    prompt_text = "\n".join(
        str(message.content) for message in llm.inputs[0].messages
    )
    assert "问题" in prompt_text
    assert "[资料1｜文章1｜第3页｜Chunk 2]" in prompt_text


def test_generator_skips_llm_when_no_context(monkeypatch):
    generator, llm, _retriever_calls, _reranker_calls, _db = build_generator(
        monkeypatch,
        [],
    )

    result = generator.generate(question="问题")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.inputs == []


def test_generator_limits_context_without_losing_source_metadata():
    documents = [
        Document(
            page_content="12345",
            metadata={"chunk_id": 1},
        ),
        Document(
            page_content="67890",
            metadata={"chunk_id": 2},
        ),
    ]

    limited = LangChainGenerator._limit_documents(documents, 7)

    assert [document.page_content for document in limited] == ["12345", "67"]
    assert [document.metadata["chunk_id"] for document in limited] == [1, 2]


def test_generator_astream_uses_async_lcel_and_handles_empty_context(
    monkeypatch,
):
    document = Document(
        page_content="参考内容",
        metadata={
            "article_id": 1,
            "chunk_id": 2,
            "page_number": 3,
            "distance": 0.2,
            "rerank_score": 0.9,
        },
    )
    generator, _llm, _retriever_calls, _reranker_calls, _db = build_generator(
        monkeypatch,
        [document],
    )
    prepared = PreparedRAGContext(
        documents=[document],
        sources=LangChainGenerator._build_sources([document]),
    )

    async def collect(context):
        return [
            chunk
            async for chunk in generator.astream("问题", context)
        ]

    assert "".join(asyncio.run(collect(prepared))) == "基于资料的回答。[资料1]"
    empty = PreparedRAGContext(documents=[], sources=[])
    assert asyncio.run(collect(empty)) == [NO_CONTEXT_ANSWER]
