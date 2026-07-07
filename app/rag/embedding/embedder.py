from sentence_transformers import SentenceTransformer
from app.config.config import settings


class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.DEVICE,
            )
        self.model = Embedder._model

    def embed(self, text: str) -> list[float]:
        """
        对单段文本生成向量
        """
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        批量生成向量
        """
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return vectors.tolist()

    def dimension(self) -> int:
        """
        返回模型向量维度
        """
        return self.model.get_sentence_embedding_dimension()