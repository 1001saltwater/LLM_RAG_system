from app.config.config import settings
from app.rag.embedding import embedder as embedder_module


class FakeHuggingFaceEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


def test_embedder_uses_instruction_for_queries_only(monkeypatch):
    monkeypatch.setattr(
        embedder_module,
        "HuggingFaceEmbeddings",
        FakeHuggingFaceEmbeddings,
    )
    monkeypatch.setattr(embedder_module.Embedder, "_embeddings", None)
    monkeypatch.setattr(embedder_module.Embedder, "_dimension", None)

    embedder = embedder_module.Embedder()

    assert "prompt" not in embedder.embeddings.kwargs["encode_kwargs"]
    assert (
        embedder.embeddings.kwargs["query_encode_kwargs"]["prompt"]
        == settings.EMBEDDING_QUERY_INSTRUCTION
    )
    assert embedder.embed_documents(["document"]) == [[1.0, 0.0]]
    assert embedder.embed_query("query") == [0.0, 1.0]
    assert embedder.dimension() == 2
