from sentence_transformers import CrossEncoder
from app.config.config import settings

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(settings.RERANK_MODEL)
    def rerank(self, question: str, documents: list[str], top_k: int = settings.RERANK_TOP_K) -> list:
        pairs = [(question, document) for document in documents]
        scores = self.model.predict(pairs)
        results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [item[0] for item in results[:top_k]]