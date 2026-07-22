from app.rag.generation.generator import Generator
from app.rag.retrieval.retriever import Retriever
from app.database.session import sessionmaker
from tests.test_retriver.utils import load_questions

def test_generator():


    db = sessionmaker()

    retriever = Retriever()
    generator = Generator()

    questions = [item["question"] for item in load_questions()]
    
    for question in questions:
        answer = generator.generate(
            question
        )
        print("\n================")
        print("问题:")
        print(question)
        print("\n回答:")
        print(answer)


    db.close()