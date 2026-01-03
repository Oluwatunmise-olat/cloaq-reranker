FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than full PyTorch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cleanup build dependencies
RUN apt-get purge -y build-essential && apt-get autoremove -y

COPY protos ./protos
COPY src ./src

ENV PYTHONUNBUFFERED=1

EXPOSE 50051

CMD ["python", "-m", "src.reranker.main"]