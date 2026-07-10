from sqlalchemy.orm import Session
from app.rag.embedding.embedder import Embedder
from app.services.service_chunk import ServiceChunk
from app.services.service_embedding import ServiceEmbedding
from app.rag.embedding.vector_store import VectorStore
from app.schemas.schema_embedding import CreateEmbedding

class EmbeddingPipeline:

    def __init__(self):
        self.chunk_service = ServiceChunk()
        self.embedding_service = ServiceEmbedding()
        self.embedder = Embedder()

    def embed_article(self,db: Session,article_id: int) -> int:

        # 1. 获取文章所有 Chunk
        chunks = self.chunk_service.get_chunk_by_article_id(db,article_id)

        if not chunks:
            return 0

        # 2. 提取所有文本
        texts = [chunk.content for chunk in chunks]

        # 3. Batch Embedding
        vectors = self.embedder.embed_batch(texts)

        # 4. 构造 Schema
        embedding_data = []

        for chunk, vector in zip(chunks, vectors):

            embedding_data.append(
                CreateEmbedding(
                    chunk_id=chunk.id,
                    embedding=vector
                )

            )

        # 5. 保存数据库
        self.embedding_service.create_embedding_batch(db,embedding_data)
        
        return len(embedding_data)