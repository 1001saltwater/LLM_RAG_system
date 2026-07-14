FROM python:3.14-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install --no-cache-dir uv

RUN uv sync

COPY app ./app
COPY main.py .
COPY .env .

CMD ["uv", "run", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]