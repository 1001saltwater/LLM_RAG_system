from app.rag.embedding.vector_store import VectorStore



class SimilaritySearch:

    def __init__(self):
        self.vector_store = VectorStore()

    def search(self, db, query_vector, top_k):
        return self.vector_store.similarity_search(
            db=db,
            query_vector=query_vector,
            top_k=top_k,
        )