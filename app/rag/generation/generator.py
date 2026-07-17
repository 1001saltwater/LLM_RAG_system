from app.rag.generation.prompt import Prompt
from app.rag.generation.LLM import LLM
from app.config.config import settings
from app.database.session import sessionmaker
from app.rag.retrieval.retriever import Retriever



class Generator:
    def __init__(self, ):
        self.prompt = Prompt()
        self.llm = LLM()
        self.retriever = Retriever()
        self.db = sessionmaker()

    def generate(self, question: str) -> str:
        chunks = self.retriever.retrieve(
            db=self.db,
            question=question,
            top_k=settings.TOP_K
        )
        prompt = self.prompt.build_rag_prompt(question, chunks)
        answer = self.llm.generate(prompt)
        return answer