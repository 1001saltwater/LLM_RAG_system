import os
import random
import time

from locust import HttpUser, between, events, task


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


QUESTIONS = [
    question.strip()
    for question in os.getenv(
        "RAG_QUESTIONS",
        "MSBERT是什么？",
    ).split("||")
    if question.strip()
]


class RAGUser(HttpUser):
    host = os.getenv("RAG_HOST", "http://localhost:8000")
    wait_time = between(
        env_float("RAG_WAIT_MIN", 1.0),
        env_float("RAG_WAIT_MAX", 3.0),
    )

    @task
    def query_rag(self) -> None:
        payload = {
            "question": random.choice(QUESTIONS),
            "top_k": env_int("RAG_TOP_K", 15),
            "max_distance": env_float("RAG_MAX_DISTANCE", 0.5),
            "rerank_top_k": env_int("RAG_RERANK_TOP_K", 3),
            "rerank_threshold": env_float(
                "RAG_RERANK_THRESHOLD",
                0.5,
            ),
        }
        headers = {"Accept": "text/event-stream"}
        timeout = env_float("RAG_TIMEOUT", 180.0)
        started_at = time.perf_counter()

        with self.client.post(
            "/rag/query",
            json=payload,
            headers=headers,
            name="/rag/query complete",
            stream=True,
            timeout=timeout,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return

            current_event = ""
            token_count = 0
            received_bytes = 0
            received_sources = False
            received_done = False

            try:
                for line in response.iter_lines(
                    chunk_size=1,
                    decode_unicode=True,
                ):
                    if not line:
                        current_event = ""
                        continue

                    received_bytes += len(line.encode("utf-8"))
                    if line.startswith("event:"):
                        current_event = line.removeprefix("event:").strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    if current_event == "token":
                        token_count += 1
                        if token_count == 1:
                            events.request.fire(
                                request_type="SSE",
                                name="/rag/query TTFT",
                                response_time=(
                                    time.perf_counter() - started_at
                                )
                                * 1000,
                                response_length=received_bytes,
                                exception=None,
                                context={},
                            )
                    elif current_event == "sources":
                        received_sources = True
                    elif current_event == "done":
                        received_done = True
                    elif current_event == "error":
                        response.failure(f"Server SSE error: {line}")
                        return
            except Exception as exc:
                response.failure(f"Streaming failed: {exc}")
                return
            finally:
                response.request_meta["response_time"] = (
                    time.perf_counter() - started_at
                ) * 1000
                response.request_meta["response_length"] = received_bytes

            if token_count == 0:
                response.failure("Stream returned no token events")
            elif not received_sources:
                response.failure("Stream returned no sources event")
            elif not received_done:
                response.failure("Stream returned no done event")
            else:
                response.success()
