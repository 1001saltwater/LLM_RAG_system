from app.database.session import sessionmaker
from app.rag.retrieval.retriever import Retriever
from tests.test_retriver.utils import load_questions
from app.config.config import settings


def test_retrieval():

    db = sessionmaker()

    retriever = Retriever()

    questions = load_questions()

    try:
        for item in questions:

            print("\n问题:", item["question"])

            chunks = retriever.retrieve(
                db=db,
                question=item["question"],
                top_k=settings.TOP_K
            )

            assert len(chunks) > 0

            for chunk in chunks:
                print("----")
                print(chunk.content[:200])

    finally:
        db.close()