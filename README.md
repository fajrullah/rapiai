# 🗂️ RapiAI — Turn Messy Documents into Clear Answers

> Stop digging through messy PDFs. Just ask.

RapiAI is an AI-powered RAG (Retrieval-Augmented Generation) application 
that lets you upload messy, unstructured documents and chat with them 
using natural language. Get precise answers with source citations — 
no more manual reading.

## 🎯 The Problem
We deal with messy documents every day — long PDFs, unstructured reports, 
scattered files. Finding the right information wastes hours.

## ✅ The Solution
Upload your documents to RapiAI, ask anything, and get clear answers 
instantly — with references to the exact source.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
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
rag-pipeline/
├── app/
│   ├── main.py         # Entry point (FastAPI initialization)
│   ├── api/            # API Endpoints & schemas
│   ├── core/           # Configuration & database singletons
│   └── services/       # RAG logic (ingestion, retrieval, prompting)
├── data/               # Persistent storage (chroma_db)
├── docs/               # Document storage
├── requirements.txt
└── .env
```