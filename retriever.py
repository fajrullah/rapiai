from ingest import get_embedder, get_collection
from config import settings


def retrieve(query: str, top_k: int = None, doc_id: str = None) -> list[dict]:
    """
    Embed the query and return top-K similar chunks from ChromaDB.

    Args:
        query:  The user's question.
        top_k:  Number of chunks to return (defaults to settings.default_top_k).
        doc_id: Optional — restrict search to a specific document.
    """
    if top_k is None:
        top_k = settings.default_top_k

    embedder = get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False).tolist()

    collection = get_collection()

    where = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": text,
                "filename": meta.get("filename"),
                "page": meta.get("page"),
                "doc_id": meta.get("doc_id"),
                "score": round(1 - distance, 4),  # cosine similarity (higher = better)
            }
        )

    return chunks