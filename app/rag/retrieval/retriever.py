from sqlalchemy.orm import Session

from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.rag.retrieval.similarity_search import SimilaritySearch
from app.services.service_chunk import ServiceChunk


class Retriever:

    def __init__(self):
        self.embedder = Embedder()
        self.search = SimilaritySearch()
        self.chunk_service = ServiceChunk()

    def retrieve(self, db: Session, question: str, top_k: int = settings.TOP_K):

        # 1. 问题生成向量
        query_vector = self.embedder.embed(question)

        # 2. 相似度检索（返回 Embedding）
        embeddings = self.search.search(
            db=db,
            query_vector=query_vector,
            top_k=top_k
        )

        chunk_ids = [embedding.chunk_id for embedding in embeddings]
        return self.chunk_service.get_chunk_by_id_batch(db, chunk_ids)