FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY modal_run/ modal_run/

RUN pip install --no-cache-dir .

COPY src/agdg/data_pipeline/aws/schema.sql /root/agdg/data_pipeline/aws/schema.sql

ENTRYPOINT ["python", "-m", "agdg.cli"]
