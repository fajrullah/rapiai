"""Unit tests for app.services.prompting module.

Tests cover:
- build_prompt(): prompt assembly with chunks, empty chunks fallback, source formatting
"""
import pytest

from app.services.prompting import build_prompt


class TestBuildPrompt:
    """Tests for the build_prompt function."""

    def test_with_chunks(self):
        chunks = [
            {"filename": "report.pdf", "page": 1, "text": "Revenue grew 20%.", "score": 0.92},
            {"filename": "report.pdf", "page": 3, "text": "Expenses decreased.", "score": 0.85},
        ]
        result = build_prompt(query="What is the revenue?", chunks=chunks)

        assert "system_prompt" in result
        assert "user_message" in result
        assert "sources" in result
        assert "Revenue grew 20%" in result["system_prompt"]
        assert "Expenses decreased." in result["system_prompt"]
        assert result["user_message"] == "What is the revenue?"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["filename"] == "report.pdf"
        assert result["sources"][0]["page"] == 1
        assert result["sources"][0]["score"] == 0.92

    def test_empty_chunks(self):
        """When no chunks are provided, the prompt should indicate no context found."""
        result = build_prompt(query="What is X?", chunks=[])
        assert "No relevant context found" in result["system_prompt"]
        assert result["user_message"] == "What is X?"
        assert result["sources"] == []

    def test_system_prompt_contains_instruction(self):
        """System prompt should instruct the LLM to use only provided context."""
        result = build_prompt(query="test", chunks=[])
        assert "ONLY" in result["system_prompt"]
        assert "don't know" in result["system_prompt"].lower() or "do not make up" in result["system_prompt"].lower()

    def test_source_labels_in_context(self):
        chunks = [{"filename": "doc.pdf", "page": 5, "text": "Important finding.", "score": 0.9}]
        result = build_prompt(query="findings?", chunks=chunks)
        assert "doc.pdf" in result["system_prompt"]
        assert "page 5" in result["system_prompt"]

    def test_multiple_chunks_numbered(self):
        chunks = [
            {"filename": "a.pdf", "page": 1, "text": "First chunk.", "score": 0.9},
            {"filename": "b.pdf", "page": 2, "text": "Second chunk.", "score": 0.8},
            {"filename": "c.pdf", "page": 3, "text": "Third chunk.", "score": 0.7},
        ]
        result = build_prompt(query="test", chunks=chunks)
        assert "[1]" in result["system_prompt"]
        assert "[2]" in result["system_prompt"]
        assert "[3]" in result["system_prompt"]

    def test_return_structure(self):
        result = build_prompt(query="q", chunks=[])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"system_prompt", "user_message", "sources"}
        assert isinstance(result["system_prompt"], str)
        assert isinstance(result["user_message"], str)
        assert isinstance(result["sources"], list)
