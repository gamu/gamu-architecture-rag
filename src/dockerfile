FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install faiss-cpu
RUN pip install llama-cpp-python

COPY . .

RUN mkdir -p /app/vector_db

CMD ["python", "rag.py"]