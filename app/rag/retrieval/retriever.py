from sqlalchemy.orm import Session
from app.config.config import settings
from app.rag.embedding.embedder import Embedder
from app.services.service_chunk import ServiceChunk
from app.services.service_embedding import ServiceEmbedding
from app.rag.embedding.vector_store import VectorStore
from app.rag.rerank.reranker import Reranker


class Retriever:

    def __init__(self):
        self.embedder = Embedder()
        self.embedding_service = ServiceEmbedding()
        self.chunk_service = ServiceChunk()
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def retrieve(self, db: Session, question: str, top_k: int = settings.TOP_K):

        # 1. 问题生成向量
        query_vector = self.embedder.embed(question)

        # 2. 相似度检索（返回 Embedding）
        embeddings = self.vector_store.similarity_search(
            db=db,
            query_vector=query_vector,
            top_k=top_k
        )

        chunk_ids = [embedding.chunk_id for embedding in embeddings]

        chunks = self.chunk_service.get_chunk_by_id_batch(db, chunk_ids)

        documents = [chunk.content for chunk in chunks]
        reranked_documents = self.reranker.rerank(question, documents)

        result = []
        for item in reranked_documents:
            for chunk in chunks:
                if chunk.content == item:
                    result.append(chunk)
                    break

        return result
