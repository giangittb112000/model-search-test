# Docker image: CUDA runtime + Python + cài thư viện ở requirements.txt
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install -U pip setuptools wheel
RUN pip install -r requirements.txt
RUN python -m pip check
RUN python -c "import cupy, thinc.compat as c; major = int(cupy.__version__.split('.')[0]); print('CuPy', cupy.__version__); print('Thinc has_cupy', c.has_cupy); assert major < 13, 'spaCy/Thinc 8.3 expects CuPy < 13'; assert c.has_cupy, 'Thinc cannot import CuPy backend'"

COPY config ./config
COPY data ./data
COPY src ./src

CMD ["python", "-c", "import spacy; print('spaCy', spacy.__version__, 'ready')"]
