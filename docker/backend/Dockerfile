FROM python:3.10-slim

WORKDIR /app

COPY  docker/backend/requirements.txt .

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

COPY docker/backend/api.py ./api.py
COPY agent ./agent

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]