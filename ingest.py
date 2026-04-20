import uuid
import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from config import settings

# Global singletons for the embedder and database collection to avoid re-initializing them
_embedder = None
_chroma_collection = None


def get_embedder() -> SentenceTransformer:
    """
    Lazy loader for the SentenceTransformer model.
    Loads the model into memory only when first requested.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def get_collection():
    """
    Lazy loader for the ChromaDB collection.
    Initializes the persistent client and returns (or creates) the specified collection.
    """
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        _chroma_collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection


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
    """Extract the first header-like line from a page."""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Header = short line, no period at end, title/uppercase
        if len(line) < 100 and not line.endswith(".") and not line.endswith(","):
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

        # Update header if this page has one
        detected_header = extract_header(text)
        if detected_header:
            current_header = detected_header

        splits = splitter.split_text(text)

        for idx, split in enumerate(splits):
            # Prepend header context to each chunk
            enriched_text = (
                f"[{current_header}]\n{split}"
                if current_header and current_header not in split
                else split
            )

            chunks.append({
                "page": page["page"],
                "chunk_index": idx,
                "text": enriched_text,
                "header": current_header,  # store for debugging/metadata
            })

    return chunks


def ingest_pdf(pdf_path: str, filename: str) -> dict:
    """
    The full RAG ingestion pipeline for a PDF:
    1. Parse: Extract text from the PDF file.
    2. Chunk: Split text into small, manageable pieces.
    3. Embed: Convert text chunks into numerical vectors (embeddings).
    4. Store: Save vectors and original text into ChromaDB for later retrieval.
    """
    doc_id = str(uuid.uuid4())  # Generate a unique ID for this document

    # 1. Extract text
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError("No readable text found in PDF.")

    # 2. Chunk
    chunks = chunk_pages(pages)

    # 3. Embed
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # 4. Store in ChromaDB
    collection = get_collection()
    # Unique IDs for every chunk so we can manage them individually
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    # Metadata helps us filter by doc_id or find which page a chunk came from
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "page": c["page"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    # Add to the vector database
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}


def delete_document(doc_id: str) -> int:
    """
    Remove all stored information (vectors and text) for a given document.
    """
    collection = get_collection()
    # Find all chunks specifically for this document ID
    results = collection.get(where={"doc_id": doc_id})
    if not results["ids"]:
        return 0
    # Delete them from ChromaDB
    collection.delete(ids=results["ids"])
    return len(results["ids"])


def list_documents() -> list[dict]:
    """
    Fetch all metadata and return a unique list of document IDs and filenames
    currently stored in the database.
    """
    collection = get_collection()
    # Retrieve only the metadata for all entries in the collection
    results = collection.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        doc_id = meta["doc_id"]
        # Deduplicate: many chunks belong to the same document
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": meta["filename"]}
    return list(seen.values())


def peek_chunks(limit: int = 10) -> list[dict]:
    """
    Peek into the database to see the first few chunks, their text, and metadata.
    """
    collection = get_collection()
    results = collection.get(limit=limit, include=["documents", "metadatas"])

    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return chunks


def get_document_chunks(doc_id: str) -> list[dict]:
    """
    Retrieve all chunks for a specific document.
    """
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