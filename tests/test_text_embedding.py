# tests/test_embedder.py

from app.rag.embedding.embedder import Embedder

def test_embed():

    embedder = Embedder()

    vector = embedder.embed("Hello World")

    assert isinstance(vector, list)

    assert len(vector) > 0

    assert isinstance(vector[0], float)