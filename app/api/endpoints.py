import os
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.api.schemas import RetrieveRequest, RetrieveResponse, PromptRequest
from app.core.config import settings
from app.services.ingestion import ingest_pdf, delete_document, list_documents, peek_chunks, get_document_chunks
from app.services.retrieval import retrieve, compress_chunk
from app.services.prompting import build_prompt

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
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

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(body: RetrieveRequest):
    chunks = retrieve(query=body.query, top_k=body.top_k, doc_id=body.doc_id)
    return {"chunks": chunks}

@router.post("/prompt")
def build_rag_prompt(body: PromptRequest):
    chunks = retrieve(query=body.query, top_k=body.top_k, doc_id=body.doc_id)

    # Compress each chunk — keep only query-relevant sentences
    for chunk in chunks:
        chunk["text"] = compress_chunk(query=body.query, chunk_text=chunk["text"])

    prompt = build_prompt(query=body.query, chunks=chunks)
    return prompt


@router.get("/documents")
def get_documents():
    return {"documents": list_documents()}

@router.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    deleted = delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"doc_id": doc_id, "deleted_chunks": deleted}

@router.get("/chunks")
def list_chunks(limit: int = 10):
    return {"chunks": peek_chunks(limit)}

@router.get("/documents/{doc_id}/chunks")
def list_document_chunks(doc_id: str):
    chunks = get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="Document not found or has no chunks.")
    return {"doc_id": doc_id, "chunks": chunks}
