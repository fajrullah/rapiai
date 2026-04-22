import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # HuggingFace token (get from huggingface.co/settings/tokens)
    hf_token: str | None = None

    # Embedding model from HuggingFace
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ChromaDB persistent storage path
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "documents"

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval settings
    default_top_k: int = 4

    # Temp folder for uploaded PDFs
    tmp_dir: str = "./tmp"

    class Config:
        env_file = ".env"


settings = Settings()

# Apply HF token to environment so sentence-transformers picks it up
if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token