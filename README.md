# cloaq-reranker

gRPC service that reranks documents by relevance using a Sentence-Transformers CrossEncoder model (e.g. MS-MARCO MiniLM).

---

## Features

- Exposes a `RerankerService.Rerank` RPC: input a query + list of documents, returns documents scored & sorted by relevance
- Based on HuggingFace `CrossEncoder` for state-of-the-art retrieval reranking
- gRPC Reflection & Health Checking for easy integration & monitoring
- Graceful shutdown, logging, and thread-safe GPU inference

---

## Prerequisites

- Python 3.8 or newer
- `protoc` (optional, for regenerating gRPC stubs)
- A virtual-environment tool (venv)

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Oluwatunmise-olat/cloaq-reranker.git
   cd cloaq-reranker
   ```
2. Create and activate a virtual environment:

```bash
  python3 -m venv .venv
  source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

1. Copy the example environment file and edit:
   ```bash
   cp .env.example .env
   ```

---

## Running the Service

```bash
python -m src.reranker.main
```

---

## Notes

If you update protos/reranker.proto, regenerate Python code with:

```bash
python -m grpc_tools.protoc \
  -I protos \
  --python_out=protos \
  --grpc_python_out=protos \
  protos/reranker.proto
```
