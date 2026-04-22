import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings

# Global singletons
_embedder = None
_chroma_collection = None

def get_embedder() -> SentenceTransformer:
    """Lazy loader for SentenceTransformer."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder

def get_collection():
    """Lazy loader for ChromaDB collection."""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_path)
        _chroma_collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _chroma_collection
