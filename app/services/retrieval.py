import re
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from app.core.config import settings
from app.core.database import get_embedder, get_collection
import numpy as np


def compress_chunk(query: str, chunk_text: str, threshold: float = 0.3) -> str:
    """Compress a chunk by keeping only sentences relevant to the query.

    1. Split chunk into sentences.
    2. Score each sentence against the query using cosine similarity.
    3. Keep only sentences scoring above *threshold*.
    4. Return the filtered text (or the original if nothing passes).
    """
    # Split on sentence-ending punctuation while keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return chunk_text

    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False)
    sentence_embeddings = embedder.encode(sentences, show_progress_bar=False)

    # Cosine similarity between the query and every sentence
    query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    sent_norms = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    similarities = (sent_norms @ query_norm.T).flatten()

    # Keep sentences above the threshold
    kept = [s for s, sim in zip(sentences, similarities) if sim >= threshold]

    # Fall back to the original text if nothing passed the filter
    if not kept:
        return chunk_text

    return " ".join(kept)

def retrieve(query: str, top_k: int = None, doc_id: str = None) -> list[dict]:
    """Embed the query and return top-K similar chunks using Hybrid Search."""
    if top_k is None:
        top_k = settings.default_top_k

    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    collection = get_collection()
    where = {"doc_id": doc_id} if doc_id else None

    # 1. Semantic (Dense) Search
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k * 2,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    dense_chunks = []
    if dense_results and dense_results["documents"] and dense_results["documents"][0]:
        for text, meta, distance in zip(
            dense_results["documents"][0],
            dense_results["metadatas"][0],
            dense_results["distances"][0],
        ):
            dense_chunks.append({
                "text": text,
                "filename": meta.get("filename"),
                "page": meta.get("page"),
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "dense_score": round(1 - distance, 4),
            })

    # 2. Sparse (BM25) Search
    all_docs = collection.get(where=where, include=["documents", "metadatas"])
    sparse_chunks = []

    if all_docs and all_docs["documents"]:
        documents = []
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            documents.append(Document(page_content=text, metadata=meta))

        if documents:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = top_k * 2
            sparse_results = bm25_retriever.invoke(query)
            
            for doc in sparse_results:
                sparse_chunks.append({
                    "text": doc.page_content,
                    "filename": doc.metadata.get("filename"),
                    "page": doc.metadata.get("page"),
                    "doc_id": doc.metadata.get("doc_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                })

    # 3. Reciprocal Rank Fusion (RRF)
    combined_scores = defaultdict(float)
    chunk_map = {}

    def populate_rrf(chunk_list, k=60):
        for rank, chunk in enumerate(chunk_list):
            chunk_id = f"{chunk['doc_id']}_{chunk['chunk_index']}"
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk
            combined_scores[chunk_id] += 1.0 / (k + rank + 1)

    populate_rrf(dense_chunks)
    populate_rrf(sparse_chunks)

    sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    final_chunks = []
    for chunk_id, rrf_score in sorted_chunks[:top_k]:
        chunk = chunk_map[chunk_id]
        final_chunks.append({
            "text": chunk["text"],
            "filename": chunk.get("filename"),
            "page": chunk.get("page"),
            "doc_id": chunk.get("doc_id"),
            "score": round(rrf_score, 4),
        })

    return final_chunks
