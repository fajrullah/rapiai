import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from ingest import ingest_pdf, delete_document, list_documents, peek_chunks, get_document_chunks
from retriever import retrieve
from prompt_builder import build_prompt

app = FastAPI(title="RAG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(settings.tmp_dir, exist_ok=True)


# ── Request / Response models ──────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = settings.default_top_k
    doc_id: str | None = None  # optional: scope to one document


class RetrieveResponse(BaseModel):
    chunks: list[dict]


class PromptRequest(BaseModel):
    query: str
    top_k: int = settings.default_top_k
    doc_id: str | None = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Accept a PDF, parse, chunk, embed, and store in ChromaDB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = os.path.join(settings.tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = ingest_pdf(tmp_path, file.filename)
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(body: RetrieveRequest):
    """Embed a query and return top-K relevant chunks from ChromaDB."""
    chunks = retrieve(query=body.query, top_k=body.top_k, doc_id=body.doc_id)
    return {"chunks": chunks}


@app.post("/prompt")
def build_rag_prompt(body: PromptRequest):
    """
    Convenience endpoint: retrieve chunks AND build the full prompt in one call.
    Your Node.js backend can call this instead of /retrieve + building the prompt itself.
    """
    chunks = retrieve(query=body.query, top_k=body.top_k, doc_id=body.doc_id)
    prompt = build_prompt(query=body.query, chunks=chunks)
    return prompt


@app.get("/documents")
def get_documents():
    """List all ingested documents."""
    return {"documents": list_documents()}


@app.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    """Delete all chunks for a given document."""
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"doc_id": doc_id, "deleted_chunks": deleted}


@app.get("/chunks")
def list_chunks(limit: int = 10):
    """
    Peek into the database to see stored chunks.
    Useful for debugging what exactly is stored in ChromaDB.
    """
    return {"chunks": peek_chunks(limit)}


@app.get("/documents/{doc_id}/chunks")
def list_document_chunks(doc_id: str):
    """
    List all chunks for a specific document.
    """
    chunks = get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no chunks.")
    return {"doc_id": doc_id, "chunks": chunks}