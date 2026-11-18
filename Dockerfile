FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "agentic-cv"]
