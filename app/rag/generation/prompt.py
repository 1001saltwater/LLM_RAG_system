SYSTEM_PROMPT = """
你是一个专业的问答系统，你的任务是根据用户的问题，从数据库中检索相关的信息并回答用户的问题。
回答要求：
1. 回答必须基于参考资料中的信息
2. 回答必须是中文。
3. 回答必须是准确的，不能包含任何错误的信息。
4. 不允许编造任何不存在的信息。
5. 如果参考资料中没有足够信息回答问题，请回答："根据已有资料无法回答该问题。"
"""

class Prompt:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_rag_prompt(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n".join(f"资料{i+1}: {item.content}" for i, item in enumerate(contexts))

        user_prompt = f"""
        问题：{question}
        参考资料：{context_text}
        请根据以上参考资料回答问题。
        """

        return self.system_prompt + user_prompt

