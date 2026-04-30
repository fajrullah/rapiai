"""Unit tests for app.services.ingestion module.

Tests cover:
- extract_header(): header detection heuristics
- chunk_pages(): text chunking with header enrichment
- extract_text_from_pdf(): PDF text extraction (mocked)
- ingest_pdf(): full pipeline with deduplication (mocked)
"""
import hashlib
import pytest
from unittest.mock import patch, MagicMock, mock_open

from app.services.ingestion import (
    extract_header,
    chunk_pages,
    extract_text_from_pdf,
    ingest_pdf,
    delete_document,
    list_documents,
    peek_chunks,
    get_document_chunks,
)


# ──────────────────────────────────────────────
# extract_header()
# ──────────────────────────────────────────────

class TestExtractHeader:
    """Tests for the extract_header heuristic."""

    def test_simple_header(self):
        text = "Introduction\nThis is the body of the section."
        assert extract_header(text) == "Introduction"

    def test_multi_word_header(self):
        text = "Executive Summary\nLorem ipsum dolor sit amet."
        assert extract_header(text) == "Executive Summary"

    def test_header_with_leading_blank_lines(self):
        text = "\n\n  \nChapter One\nSome content here."
        assert extract_header(text) == "Chapter One"

    def test_no_header_long_line(self):
        """Lines longer than 60 chars should not be detected as headers."""
        text = "This is a very long line that definitely exceeds sixty characters and should not be a header.\nMore text."
        assert extract_header(text) == ""

    def test_no_header_sentence_ending(self):
        """Lines ending with punctuation should not be detected as headers."""
        text = "This ends with a period.\nMore text."
        assert extract_header(text) == ""

    def test_no_header_too_many_words(self):
        """Lines with more than 6 words should not be detected as headers."""
        text = "This has seven words in this line\nBody text."
        assert extract_header(text) == ""

    def test_empty_text(self):
        assert extract_header("") == ""

    def test_only_whitespace(self):
        assert extract_header("   \n  \n  ") == ""

    def test_header_ending_with_comma(self):
        """Lines ending with comma are skipped, but next qualifying line is returned."""
        text = "Hello,\nWorld"
        # "Hello," ends with comma → skipped. "World" qualifies.
        assert extract_header(text) == "World"

    def test_header_ending_with_colon(self):
        """Lines ending with colon are skipped, but next qualifying line is returned."""
        text = "Summary:\nContent here"
        # "Summary:" ends with colon → skipped. "Content here" qualifies.
        assert extract_header(text) == "Content here"

    def test_single_word_header(self):
        text = "Abstract\nLorem ipsum dolor sit amet."
        assert extract_header(text) == "Abstract"

    def test_exactly_six_words(self):
        """Exactly 6 words is within the <=6 threshold."""
        text = "One Two Three Four Five Six\nBody."
        assert extract_header(text) == "One Two Three Four Five Six"


# ──────────────────────────────────────────────
# chunk_pages()
# ──────────────────────────────────────────────

class TestChunkPages:
    """Tests for the chunk_pages function."""

    def test_single_page_single_chunk(self):
        pages = [{"page": 1, "text": "Short text."}]
        chunks = chunk_pages(pages)
        assert len(chunks) >= 1
        assert chunks[0]["page"] == 1
        assert chunks[0]["chunk_index"] == 0
        assert "Short text." in chunks[0]["text"]

    def test_header_enrichment(self):
        """Chunks should be prefixed with the detected header."""
        pages = [{"page": 1, "text": "Introduction\nSome body text that is part of the introduction section."}]
        chunks = chunk_pages(pages)
        assert chunks[0]["header"] == "Introduction"
        # The header should appear in the text as enrichment prefix
        assert "[Introduction]" in chunks[0]["text"] or "Introduction" in chunks[0]["text"]

    def test_empty_page_skipped(self):
        pages = [
            {"page": 1, "text": "   "},
            {"page": 2, "text": "Valid text content here."},
        ]
        chunks = chunk_pages(pages)
        # Only page 2 should produce chunks
        for chunk in chunks:
            assert chunk["page"] == 2

    def test_multiple_pages(self):
        pages = [
            {"page": 1, "text": "Page one content."},
            {"page": 2, "text": "Page two content."},
        ]
        chunks = chunk_pages(pages)
        page_numbers = {c["page"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_index_resets_per_page(self):
        """Each page should start its chunk_index at 0."""
        pages = [
            {"page": 1, "text": "First page text."},
            {"page": 2, "text": "Second page text."},
        ]
        chunks = chunk_pages(pages)
        for page_num in [1, 2]:
            page_chunks = [c for c in chunks if c["page"] == page_num]
            if page_chunks:
                assert page_chunks[0]["chunk_index"] == 0

    def test_long_text_produces_multiple_chunks(self):
        """A large text block should be split into multiple chunks."""
        long_text = "This is a sentence. " * 200
        pages = [{"page": 1, "text": long_text}]
        chunks = chunk_pages(pages)
        assert len(chunks) > 1


# ──────────────────────────────────────────────
# extract_text_from_pdf() — mocked
# ──────────────────────────────────────────────

class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf with mocked PyMuPDF."""

    @patch("app.services.ingestion.fitz")
    def test_extracts_pages(self, mock_fitz):
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Page 1 text"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Page 2 text"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        result = extract_text_from_pdf("dummy.pdf")
        assert len(result) == 2
        assert result[0] == {"page": 1, "text": "Page 1 text"}
        assert result[1] == {"page": 2, "text": "Page 2 text"}

    @patch("app.services.ingestion.fitz")
    def test_skips_empty_pages(self, mock_fitz):
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "  "
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Content"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_fitz.open.return_value = mock_doc

        result = extract_text_from_pdf("dummy.pdf")
        assert len(result) == 1
        assert result[0]["page"] == 2


# ──────────────────────────────────────────────
# ingest_pdf() — mocked
# ──────────────────────────────────────────────

class TestIngestPdf:
    """Tests for the full ingest_pdf pipeline with mocked dependencies."""

    @patch("app.services.ingestion.get_collection")
    @patch("app.services.ingestion.get_embedder")
    @patch("app.services.ingestion.extract_text_from_pdf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"fake pdf content")
    def test_successful_ingestion(self, mock_file, mock_extract, mock_embedder_fn, mock_collection_fn):
        # Setup mocks
        mock_extract.return_value = [{"page": 1, "text": "Introduction\nSample text content."}]

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = MagicMock(tolist=MagicMock(return_value=[[0.1, 0.2]]))
        mock_embedder_fn.return_value = mock_embedder

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}  # No duplicates
        mock_collection_fn.return_value = mock_collection

        result = ingest_pdf("test.pdf", "test.pdf")

        assert result["filename"] == "test.pdf"
        assert "doc_id" in result
        assert result["chunk_count"] >= 1
        mock_collection.add.assert_called_once()

    @patch("app.services.ingestion.get_collection")
    @patch("builtins.open", new_callable=mock_open, read_data=b"duplicate content")
    def test_duplicate_detection(self, mock_file, mock_collection_fn):
        """Already-ingested documents should be skipped."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["existing_id_1"]}
        mock_collection_fn.return_value = mock_collection

        result = ingest_pdf("test.pdf", "test.pdf")

        assert result["status"] == "skipped"
        assert "already uploaded" in result["message"].lower()

    @patch("app.services.ingestion.get_collection")
    @patch("app.services.ingestion.get_embedder")
    @patch("app.services.ingestion.extract_text_from_pdf")
    @patch("builtins.open", new_callable=mock_open, read_data=b"empty pdf")
    def test_empty_pdf_raises_error(self, mock_file, mock_extract, mock_embedder_fn, mock_collection_fn):
        """PDFs with no readable text should raise ValueError."""
        mock_extract.return_value = []  # No pages

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_collection_fn.return_value = mock_collection

        with pytest.raises(ValueError, match="No readable text"):
            ingest_pdf("empty.pdf", "empty.pdf")


# ──────────────────────────────────────────────
# delete_document() — mocked
# ──────────────────────────────────────────────

class TestDeleteDocument:

    @patch("app.services.ingestion.get_collection")
    def test_delete_existing_document(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
        mock_collection_fn.return_value = mock_collection

        deleted = delete_document("doc-123")
        assert deleted == 3
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])

    @patch("app.services.ingestion.get_collection")
    def test_delete_nonexistent_document(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mock_collection_fn.return_value = mock_collection

        deleted = delete_document("nonexistent")
        assert deleted == 0
        mock_collection.delete.assert_not_called()


# ──────────────────────────────────────────────
# list_documents() — mocked
# ──────────────────────────────────────────────

class TestListDocuments:

    @patch("app.services.ingestion.get_collection")
    def test_list_unique_documents(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"doc_id": "a", "filename": "file_a.pdf"},
                {"doc_id": "a", "filename": "file_a.pdf"},
                {"doc_id": "b", "filename": "file_b.pdf"},
            ]
        }
        mock_collection_fn.return_value = mock_collection

        docs = list_documents()
        assert len(docs) == 2
        doc_ids = {d["doc_id"] for d in docs}
        assert doc_ids == {"a", "b"}

    @patch("app.services.ingestion.get_collection")
    def test_list_empty(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"metadatas": []}
        mock_collection_fn.return_value = mock_collection

        docs = list_documents()
        assert docs == []


# ──────────────────────────────────────────────
# get_document_chunks() — mocked
# ──────────────────────────────────────────────

class TestGetDocumentChunks:

    @patch("app.services.ingestion.get_collection")
    def test_returns_chunks(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": ["chunk1", "chunk2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"page": 1}, {"page": 2}],
        }
        mock_collection_fn.return_value = mock_collection

        chunks = get_document_chunks("doc-1")
        assert len(chunks) == 2
        assert chunks[0]["id"] == "chunk1"
        assert chunks[1]["document"] == "text2"

    @patch("app.services.ingestion.get_collection")
    def test_no_chunks(self, mock_collection_fn):
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }
        mock_collection_fn.return_value = mock_collection

        chunks = get_document_chunks("nonexistent")
        assert chunks == []
