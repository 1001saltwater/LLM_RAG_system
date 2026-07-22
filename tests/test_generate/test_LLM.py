from app.rag.generation.LLM import LLM

def test_llm():

    llm = LLM()

    answer = llm.generate(
        """
        请介绍一下RAG系统。
        """
    )

    print(answer)