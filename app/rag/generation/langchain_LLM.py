from collections.abc import Sequence
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config.config import settings


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_API_URL,
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
    )


class LangChainLLM:
    def __init__(self, model: BaseChatModel | None = None):
        self.model = model or get_chat_model()
        self.output_parser = StrOutputParser()

    @property
    def runnable(self) -> BaseChatModel:
        """Expose the chat model for LCEL composition."""
        return self.model

    def invoke(self, messages: Sequence[BaseMessage]) -> BaseMessage:
        return self.model.invoke(list(messages))

    def generate(self, messages: Sequence[BaseMessage]) -> str:
        response = self.invoke(messages)
        return self.output_parser.invoke(response).strip()
