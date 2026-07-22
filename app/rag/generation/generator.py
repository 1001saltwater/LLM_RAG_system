from app.rag.generation.prompt import Prompt
from app.rag.generation.LLM import LLM
from app.config.config import settings
from app.rag.retrieval.retriever import Retriever
from sqlalchemy.orm import Session



class Generator:
    def __init__(self, ):
        self.prompt = Prompt()
        self.llm = LLM()
        self.retriever = Retriever()

    def generate(self, db: Session, question: str) -> str:
        chunks = self.retriever.retrieve(
            db=db,
            question=question,
            top_k=settings.TOP_K
        )
        prompt = self.prompt.build_rag_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return answer