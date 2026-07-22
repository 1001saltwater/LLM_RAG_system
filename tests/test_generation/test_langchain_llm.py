from langchain_core.messages import AIMessage, HumanMessage

from app.config.config import settings
from app.rag.generation import langchain_LLM as llm_module


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.invocations: list = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content="  测试回答。  ")


def test_langchain_llm_uses_settings_cache_and_string_parser(monkeypatch):
    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatModel)
    llm_module.get_chat_model.cache_clear()

    try:
        first = llm_module.LangChainLLM()
        second = llm_module.LangChainLLM()

        assert first.model is second.model
        assert first.model.kwargs == {
            "model": settings.LLM_MODEL,
            "api_key": settings.LLM_API_KEY,
            "base_url": settings.LLM_API_URL,
            "temperature": settings.TEMPERATURE,
            "max_tokens": settings.MAX_TOKENS,
            "timeout": settings.LLM_TIMEOUT_SECONDS,
            "max_retries": settings.LLM_MAX_RETRIES,
        }

        messages = [HumanMessage(content="问题")]
        assert first.generate(messages) == "测试回答。"
        assert first.model.invocations == [messages]
        assert first.runnable is first.model
    finally:
        llm_module.get_chat_model.cache_clear()
