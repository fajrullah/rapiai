from pydantic import BaseModel
from app.core.config import settings

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = settings.default_top_k
    doc_id: str | None = None

class RetrieveResponse(BaseModel):
    chunks: list[dict]

class PromptRequest(BaseModel):
    query: str
    top_k: int = settings.default_top_k
    doc_id: str | None = None
