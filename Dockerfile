# Docker image: Python + cài thư viện ở requirements.txt
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY config ./config
COPY data ./data
COPY src ./src

CMD ["python", "-c", "import spacy; print('spaCy', spacy.__version__, 'ready')"]
