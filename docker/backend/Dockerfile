FROM python:3.11-slim

WORKDIR /app

# 默认 CPU 运行，GPU 用户可在本地安装对应 CUDA 版本的 PyTorch。
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py ./main.py
COPY agent ./agent
COPY rag ./rag
COPY configs ./configs
COPY docker/backend/api.py ./docker/backend/api.py

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    EMBEDDING_DEVICE=cpu \
    RERANKER_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "docker.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
