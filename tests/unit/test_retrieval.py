"""Unit tests for app.services.retrieval module.

Tests cover:
- compress_chunk(): sentence filtering by cosine similarity
- retrieve(): hybrid search (dense + sparse + RRF)
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.services.retrieval import compress_chunk, retrieve


class TestCompressChunk:
    """Tests for the compress_chunk sentence-filtering function."""

    @patch("app.services.retrieval.get_embedder")
    def test_keeps_relevant_sentences(self, mock_embedder_fn):
        mock_embedder = MagicMock()
        query_emb = np.array([[1.0, 0.0, 0.0]])
        sent_embs = np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
        mock_embedder.encode.side_effect = [query_emb, sent_embs]
        mock_embedder_fn.return_value = mock_embedder

        result = compress_chunk(query="revenue", chunk_text="Revenue grew 20%. Unrelated weather info.", threshold=0.3)
        assert "Revenue grew 20%" in result
        assert "weather" not in result

    @patch("app.services.retrieval.get_embedder")
    def test_fallback_when_nothing_passes(self, mock_embedder_fn):
        mock_embedder = MagicMock()
        query_emb = np.array([[1.0, 0.0, 0.0]])
        sent_embs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        mock_embedder.encode.side_effect = [query_emb, sent_embs]
        mock_embedder_fn.return_value = mock_embedder

        original = "Sentence one. Sentence two."
        result = compress_chunk(query="unrelated query", chunk_text=original, threshold=0.9)
        assert result == original

    @patch("app.services.retrieval.get_embedder")
    def test_empty_text(self, mock_embedder_fn):
        result = compress_chunk(query="test", chunk_text="")
        assert result == ""

    @patch("app.services.retrieval.get_embedder")
    def test_single_sentence(self, mock_embedder_fn):
        mock_embedder = MagicMock()
        query_emb = np.array([[1.0, 0.0]])
        sent_embs = np.array([[0.95, 0.05]])
        mock_embedder.encode.side_effect = [query_emb, sent_embs]
        mock_embedder_fn.return_value = mock_embedder

        result = compress_chunk(query="test", chunk_text="Single sentence here.", threshold=0.3)
        assert "Single sentence here." in result


class TestRetrieve:
    """Tests for the hybrid retrieve function with mocked dependencies."""

    @patch("app.services.retrieval.BM25Retriever")
    @patch("app.services.retrieval.get_collection")
    @patch("app.services.retrieval.get_embedder")
    def test_basic_retrieval(self, mock_embedder_fn, mock_collection_fn, mock_bm25_cls):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Dense chunk text"]],
            "metadatas": [[{"filename": "doc.pdf", "page": 1, "doc_id": "d1", "chunk_index": 0}]],
            "distances": [[0.2]],
        }
        mock_collection.get.return_value = {
            "documents": ["Sparse chunk text"],
            "metadatas": [{"filename": "doc.pdf", "page": 2, "doc_id": "d1", "chunk_index": 1}],
        }
        mock_collection_fn.return_value = mock_collection

        mock_bm25_instance = MagicMock()
        mock_bm25_doc = MagicMock()
        mock_bm25_doc.page_content = "Sparse chunk text"
        mock_bm25_doc.metadata = {"filename": "doc.pdf", "page": 2, "doc_id": "d1", "chunk_index": 1}
        mock_bm25_instance.invoke.return_value = [mock_bm25_doc]
        mock_bm25_cls.from_documents.return_value = mock_bm25_instance

        results = retrieve(query="test query", top_k=2)
        assert len(results) >= 1
        for chunk in results:
            assert "text" in chunk
            assert "filename" in chunk
            assert "score" in chunk

    @patch("app.services.retrieval.BM25Retriever")
    @patch("app.services.retrieval.get_collection")
    @patch("app.services.retrieval.get_embedder")
    def test_retrieval_with_doc_id_filter(self, mock_embedder_fn, mock_collection_fn, mock_bm25_cls):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Filtered text"]],
            "metadatas": [[{"filename": "target.pdf", "page": 1, "doc_id": "target-doc", "chunk_index": 0}]],
            "distances": [[0.1]],
        }
        mock_collection.get.return_value = {
            "documents": ["Filtered text"],
            "metadatas": [{"filename": "target.pdf", "page": 1, "doc_id": "target-doc", "chunk_index": 0}],
        }
        mock_collection_fn.return_value = mock_collection

        mock_bm25_instance = MagicMock()
        mock_bm25_instance.invoke.return_value = []
        mock_bm25_cls.from_documents.return_value = mock_bm25_instance

        results = retrieve(query="query", top_k=2, doc_id="target-doc")
        call_kwargs = mock_collection.query.call_args
        assert call_kwargs.kwargs.get("where") == {"doc_id": "target-doc"} or \
               (call_kwargs[1].get("where") == {"doc_id": "target-doc"})

    @patch("app.services.retrieval.BM25Retriever")
    @patch("app.services.retrieval.get_collection")
    @patch("app.services.retrieval.get_embedder")
    def test_empty_collection(self, mock_embedder_fn, mock_collection_fn, mock_bm25_cls):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection_fn.return_value = mock_collection

        results = retrieve(query="query", top_k=2)
        assert results == []

    @patch("app.services.retrieval.BM25Retriever")
    @patch("app.services.retrieval.get_collection")
    @patch("app.services.retrieval.get_embedder")
    def test_rrf_fusion_scores(self, mock_embedder_fn, mock_collection_fn, mock_bm25_cls):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Text A", "Text B"]],
            "metadatas": [[
                {"filename": "f.pdf", "page": 1, "doc_id": "d1", "chunk_index": 0},
                {"filename": "f.pdf", "page": 2, "doc_id": "d1", "chunk_index": 1},
            ]],
            "distances": [[0.1, 0.3]],
        }
        mock_collection.get.return_value = {
            "documents": ["Text A", "Text B"],
            "metadatas": [
                {"filename": "f.pdf", "page": 1, "doc_id": "d1", "chunk_index": 0},
                {"filename": "f.pdf", "page": 2, "doc_id": "d1", "chunk_index": 1},
            ],
        }
        mock_collection_fn.return_value = mock_collection

        mock_bm25_instance = MagicMock()
        mock_doc_a = MagicMock()
        mock_doc_a.page_content = "Text A"
        mock_doc_a.metadata = {"filename": "f.pdf", "page": 1, "doc_id": "d1", "chunk_index": 0}
        mock_bm25_instance.invoke.return_value = [mock_doc_a]
        mock_bm25_cls.from_documents.return_value = mock_bm25_instance

        results = retrieve(query="query", top_k=2)
        assert len(results) >= 1
        assert results[0]["text"] == "Text A"
        if len(results) >= 2:
            assert results[0]["score"] > results[1]["score"]

    @patch("app.services.retrieval.get_collection")
    @patch("app.services.retrieval.get_embedder")
    def test_default_top_k(self, mock_embedder_fn, mock_collection_fn):
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_collection.get.return_value = {"documents": [], "metadatas": []}
        mock_collection_fn.return_value = mock_collection

        retrieve(query="test")
        call_kwargs = mock_collection.query.call_args
        from app.core.config import settings
        expected_n = settings.default_top_k * 2
        assert call_kwargs.kwargs.get("n_results") == expected_n or \
               call_kwargs[1].get("n_results") == expected_n
