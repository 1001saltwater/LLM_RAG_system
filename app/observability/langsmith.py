from collections.abc import Generator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langsmith import Client, tracing_context

from app.config.config import settings


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client | None:
    if not settings.LANGSMITH_TRACING:
        return None

    api_key = settings.LANGSMITH_API_KEY
    if api_key is None:
        raise RuntimeError(
            "LANGSMITH_API_KEY is required when tracing is enabled"
        )

    return Client(
        api_url=settings.LANGSMITH_ENDPOINT,
        api_key=api_key.get_secret_value(),
        hide_inputs=settings.LANGSMITH_HIDE_INPUTS,
        hide_outputs=settings.LANGSMITH_HIDE_OUTPUTS,
    )


@contextmanager
def langsmith_tracing_context(
    *,
    tags: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    client = get_langsmith_client()
    if client is None:
        yield
        return

    with tracing_context(
        enabled=True,
        client=client,
        project_name=settings.LANGSMITH_PROJECT,
        tags=list(tags or []),
        metadata=metadata or {},
    ):
        yield
