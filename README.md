# RAG Service (FastAPI)

A lightweight RAG microservice — PDF ingestion, ChromaDB vector storage, and retrieval — designed to pair with an existing Node.js chat backend.

## Setup

```bash
cd rag-service
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Upload a PDF (multipart/form-data) |
| POST | `/retrieve` | Get top-K chunks for a query |
| POST | `/prompt` | Retrieve + build full LLM prompt (one call) |
| GET | `/documents` | List ingested documents |
| DELETE | `/documents/{doc_id}` | Remove a document |

## How Node.js uses this

### On PDF upload
```js
const form = new FormData();
form.append('file', pdfBuffer, { filename: 'doc.pdf', contentType: 'application/pdf' });
const res = await fetch('http://localhost:8001/ingest', { method: 'POST', body: form });
const { doc_id, chunk_count } = await res.json();
```

### On every chat message
```js
const res = await fetch('http://localhost:8001/prompt', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: userMessage, top_k: 4 })
});
const { system_prompt, user_message, sources } = await res.json();

// Then send system_prompt + user_message to your HuggingFace LLM
// and stream the response back through WebSocket
```

## Project structure

```
rag-service/
├── main.py            # FastAPI app & routes
├── ingest.py          # PDF parse → chunk → embed → ChromaDB
├── retriever.py       # Query embed → ChromaDB similarity search
├── prompt_builder.py  # Assemble context + question into LLM prompt
├── config.py          # Settings (model, paths, chunk size)
├── requirements.txt
└── .env.example
```