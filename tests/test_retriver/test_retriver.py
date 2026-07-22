# tests/test_retriver/test_retriver.py

from app.database.session import sessionmaker
from app.rag.retrieval.retriever import Retriever
from tests.test_retriver.utils import load_questions
from app.config.config import settings


def print_chunk_result(index, chunk):

    print("=" * 80)
    print(f"Top {index}")
    print(f"Chunk ID: {chunk.id}")
    print(f"Article ID: {chunk.article_id}")
    print(f"Created Time: {chunk.created_time}")

    print("-" * 80)
    print("Content:")
    print(chunk.content[:500])

    print("=" * 80)


def test_retrieval():

    db = sessionmaker()

    retriever = Retriever()

    questions = load_questions()

    try:

        for idx, item in enumerate(questions, start=1):

            question = item["question"]

            print("\n\n")
            print("#" * 80)
            print(f"测试问题 {idx}")
            print(f"Question: {question}")
            print("#" * 80)


            chunks = retriever.retrieve(
                db=db,
                question=question,
                top_k=settings.TOP_K
            )


            print(f"\n召回数量: {len(chunks)}")

            for rank, chunk in enumerate(chunks, start=1):

                print_chunk_result(
                    rank,
                    chunk
                )


    finally:
        db.close()