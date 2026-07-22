from app.rag.generation.prompt import Prompt
from app.rag.retrieval.retriever import Retriever
from app.database.session import sessionmaker
from app.config.config import settings

def test_prompt():

    prompt = Prompt()
    retriever = Retriever()
    db = sessionmaker()

    question = "请介绍一下RAG系统。"

    contexts = retriever.retrieve(
        db=db,
        question=question,
        top_k=settings.TOP_K
    )

    rag_prompt = prompt.build_rag_prompt(
        question=question,
        contexts=contexts
    )

    print(rag_prompt)
