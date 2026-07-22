from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from sqlalchemy.orm import Session

from app.rag.generation import langchain_generator as generator_module
from app.rag.generation.langchain_generator import (
    LangChainGenerator,
    NO_CONTEXT_ANSWER,
)


class FakeLLM:
    def __init__(self):
        self.inputs = []
        self.runnable = RunnableLambda(self.invoke)

    def invoke(self, prompt_value):
        self.inputs.append(prompt_value)
        return AIMessage(content="基于资料的回答。[资料1]")


class FakeRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.documents


class FakeCompressionRetriever:
    def __init__(self, base_retriever, base_compressor):
        self.base_retriever = base_retriever
        self.base_compressor = base_compressor

    def invoke(self, query: str) -> list[Document]:
        return self.base_retriever.invoke(query)


def build_generator(monkeypatch, documents):
    retriever_calls = []
    reranker_calls = []
    retriever = FakeRetriever(documents)

    def retriever_factory(**kwargs):
        retriever_calls.append(kwargs)
        return retriever

    def reranker_factory(**kwargs):
        reranker_calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        generator_module,
        "ContextualCompressionRetriever",
        FakeCompressionRetriever,
    )
    llm = FakeLLM()
    generator = LangChainGenerator(
        llm=llm,
        retriever_factory=retriever_factory,
        reranker_factory=reranker_factory,
    )
    return generator, llm, retriever_calls, reranker_calls


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
    generator, llm, retriever_calls, reranker_calls = build_generator(
        monkeypatch,
        documents,
    )
    db = Session()

    result = generator.generate(
        db=db,
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
    db.close()


def test_generator_skips_llm_when_no_context(monkeypatch):
    generator, llm, _retriever_calls, _reranker_calls = build_generator(
        monkeypatch,
        [],
    )
    db = Session()

    result = generator.generate(db=db, question="问题")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert llm.inputs == []
    db.close()


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
