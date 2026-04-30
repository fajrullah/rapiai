"""Integration tests for API endpoints.

Tests cover:
- POST /ingest: file upload, validation, deduplication
- POST /prompt: prompt building pipeline
- POST /retrieve: chunk retrieval
- GET /health: health check
- GET /documents: list documents
- DELETE /documents/{doc_id}: document removal
- GET /chunks: peek chunks
- GET /documents/{doc_id}/chunks: document-specific chunks
"""
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIngestEndpoint:

    def test_rejects_non_pdf(self):
        file_content = b"not a pdf"
        response = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    @patch("app.api.endpoints.ingest_pdf")
    def test_successful_ingest(self, mock_ingest):
        mock_ingest.return_value = {"doc_id": "test-uuid", "filename": "test.pdf", "chunk_count": 5}
        file_content = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/ingest",
            files={"file": ("test.pdf", io.BytesIO(file_content), "application/pdf")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "test-uuid"
        assert data["chunk_count"] == 5
        mock_ingest.assert_called_once()

    @patch("app.api.endpoints.ingest_pdf")
    def test_duplicate_ingest(self, mock_ingest):
        mock_ingest.return_value = {"status": "skipped", "message": "Document already uploaded"}
        file_content = b"%PDF-1.4 duplicate"
        response = client.post(
            "/ingest",
            files={"file": ("dup.pdf", io.BytesIO(file_content), "application/pdf")},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "skipped"

    @patch("app.api.endpoints.ingest_pdf")
    def test_ingest_empty_pdf(self, mock_ingest):
        mock_ingest.side_effect = ValueError("No readable text found in PDF.")
        file_content = b"%PDF-1.4 empty"
        response = client.post(
            "/ingest",
            files={"file": ("empty.pdf", io.BytesIO(file_content), "application/pdf")},
        )
        assert response.status_code == 422
        assert "No readable text" in response.json()["detail"]

    def test_ingest_no_file(self):
        response = client.post("/ingest")
        assert response.status_code == 422


class TestRetrieveEndpoint:

    @patch("app.api.endpoints.retrieve")
    def test_retrieve_chunks(self, mock_retrieve):
        mock_retrieve.return_value = [
            {"text": "chunk1", "filename": "doc.pdf", "page": 1, "doc_id": "d1", "score": 0.9},
        ]
        response = client.post("/retrieve", json={"query": "test query"})
        assert response.status_code == 200
        assert len(response.json()["chunks"]) == 1

    @patch("app.api.endpoints.retrieve")
    def test_retrieve_with_doc_id(self, mock_retrieve):
        mock_retrieve.return_value = []
        response = client.post("/retrieve", json={"query": "test", "doc_id": "specific-doc"})
        assert response.status_code == 200
        mock_retrieve.assert_called_once_with(query="test", top_k=4, doc_id="specific-doc")

    def test_retrieve_missing_query(self):
        response = client.post("/retrieve", json={})
        assert response.status_code == 422


class TestPromptEndpoint:

    @patch("app.api.endpoints.build_prompt")
    @patch("app.api.endpoints.compress_chunk")
    @patch("app.api.endpoints.retrieve")
    def test_prompt_pipeline(self, mock_retrieve, mock_compress, mock_build):
        mock_retrieve.return_value = [
            {"text": "Revenue grew 20%.", "filename": "report.pdf", "page": 1, "doc_id": "d1", "score": 0.9},
        ]
        mock_compress.return_value = "Revenue grew 20%."
        mock_build.return_value = {
            "system_prompt": "You are a helpful assistant...",
            "user_message": "What is the revenue?",
            "sources": [{"filename": "report.pdf", "page": 1, "score": 0.9}],
        }
        response = client.post("/prompt", json={"query": "What is the revenue?"})
        assert response.status_code == 200
        data = response.json()
        assert "system_prompt" in data
        assert "user_message" in data
        assert "sources" in data
        mock_retrieve.assert_called_once()
        mock_compress.assert_called_once()
        mock_build.assert_called_once()

    @patch("app.api.endpoints.build_prompt")
    @patch("app.api.endpoints.retrieve")
    def test_prompt_no_results(self, mock_retrieve, mock_build):
        mock_retrieve.return_value = []
        mock_build.return_value = {
            "system_prompt": "No relevant context found.",
            "user_message": "Unknown query",
            "sources": [],
        }
        response = client.post("/prompt", json={"query": "Unknown query"})
        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_prompt_missing_query(self):
        response = client.post("/prompt", json={})
        assert response.status_code == 422


class TestDocumentsEndpoint:

    @patch("app.api.endpoints.list_documents")
    def test_list_documents(self, mock_list):
        mock_list.return_value = [
            {"doc_id": "d1", "filename": "file1.pdf"},
            {"doc_id": "d2", "filename": "file2.pdf"},
        ]
        response = client.get("/documents")
        assert response.status_code == 200
        assert len(response.json()["documents"]) == 2

    @patch("app.api.endpoints.list_documents")
    def test_list_empty(self, mock_list):
        mock_list.return_value = []
        response = client.get("/documents")
        assert response.status_code == 200
        assert response.json()["documents"] == []


class TestDeleteDocumentEndpoint:

    @patch("app.api.endpoints.delete_document")
    def test_delete_existing(self, mock_delete):
        mock_delete.return_value = 5
        response = client.delete("/documents/doc-123")
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc-123"
        assert data["deleted_chunks"] == 5

    @patch("app.api.endpoints.delete_document")
    def test_delete_nonexistent(self, mock_delete):
        mock_delete.return_value = 0
        response = client.delete("/documents/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestChunksEndpoint:

    @patch("app.api.endpoints.peek_chunks")
    def test_peek_chunks(self, mock_peek):
        mock_peek.return_value = [{"id": "c1", "document": "text", "metadata": {}}]
        response = client.get("/chunks")
        assert response.status_code == 200
        assert len(response.json()["chunks"]) == 1

    @patch("app.api.endpoints.peek_chunks")
    def test_peek_chunks_with_limit(self, mock_peek):
        mock_peek.return_value = []
        response = client.get("/chunks?limit=5")
        assert response.status_code == 200
        mock_peek.assert_called_once_with(5)


class TestDocumentChunksEndpoint:

    @patch("app.api.endpoints.get_document_chunks")
    def test_get_document_chunks(self, mock_get_chunks):
        mock_get_chunks.return_value = [
            {"id": "c1", "document": "text1", "metadata": {"page": 1}},
        ]
        response = client.get("/documents/doc-1/chunks")
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc-1"
        assert len(data["chunks"]) == 1

    @patch("app.api.endpoints.get_document_chunks")
    def test_no_chunks_404(self, mock_get_chunks):
        mock_get_chunks.return_value = []
        response = client.get("/documents/nonexistent/chunks")
        assert response.status_code == 404
