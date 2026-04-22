import uuid
import time
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.database import get_embedder, get_collection

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text page by page from a PDF file."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages

def extract_header(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        word_count = len(line.split())
        is_short = len(line) < 60
        has_no_sentence_end = not line.endswith((".", ",", ":", ";"))
        is_not_sentence = word_count <= 6

        if is_short and has_no_sentence_end and is_not_sentence:
            return line
    return ""

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into smaller overlapping chunks with contextual headers."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = []
    current_header = ""

    for page in pages:
        text = page["text"].strip()
        if not text:
            continue

        detected_header = extract_header(text)
        if detected_header:
            current_header = detected_header

        splits = splitter.split_text(text)

        for idx, split in enumerate(splits):
            enriched_text = (
                f"[{current_header}]\n{split}"
                if current_header and current_header not in split
                else split
            )

            chunks.append({
                "page": page["page"],
                "chunk_index": idx,
                "text": enriched_text,
                "header": current_header,
            })

    return chunks

def ingest_pdf(pdf_path: str, filename: str) -> dict:
    """The full RAG ingestion pipeline for a PDF."""
    doc_id = str(uuid.uuid4())

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("No readable text found in PDF.")

    chunks = chunk_pages(pages)

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection = get_collection()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "page": c["page"],
            "chunk_index": c["chunk_index"],
            "timestamp": time.time(),
        }
        for c in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}

def delete_document(doc_id: str) -> int:
    """Remove all stored information for a given document."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if not results["ids"]:
        return 0
    collection.delete(ids=results["ids"])
    return len(results["ids"])

def list_documents() -> list[dict]:
    """Fetch all unique document IDs and filenames current stored."""
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        doc_id = meta["doc_id"]
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": meta["filename"]}
    return list(seen.values())

def peek_chunks(limit: int = 10) -> list[dict]:
    """Peek into the database to see the latest chunks stored."""
    collection = get_collection()
    results = collection.get(include=["documents", "metadatas"])

    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
        })

    chunks.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
    return chunks[:limit]

def get_document_chunks(doc_id: str) -> list[dict]:
    """Retrieve all chunks for a specific document."""
    collection = get_collection()
    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"]
    )

    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return chunks
