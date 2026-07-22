from langchain_huggingface import HuggingFaceEmbeddings

from app.config.config import settings


class Embedder:
    _embeddings: HuggingFaceEmbeddings | None = None
    _dimension: int | None = None

    def __init__(self):
        if Embedder._embeddings is None:
            Embedder._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.DEVICE},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": settings.EMBEDDING_BATCH_SIZE,
                },
                query_encode_kwargs={
                    "normalize_embeddings": True,
                    "prompt": settings.EMBEDDING_QUERY_INSTRUCTION,
                },
            )
        self.embeddings = Embedder._embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """生成文档向量。"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """生成查询向量。"""
        return self.embeddings.embed_query(text)

    def dimension(self) -> int:
        """返回模型生成的向量维度。"""
        if Embedder._dimension is None:
            Embedder._dimension = len(self.embed_query("dimension check"))
        return Embedder._dimension