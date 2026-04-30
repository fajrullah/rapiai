"""Unit tests for app.services.prompting module.

Tests cover:
- build_prompt(): prompt assembly, LLM answer synthesis, source formatting
"""
import pytest
from unittest.mock import patch

from app.services.prompting import build_prompt


class TestBuildPrompt:
    """Tests for the build_prompt function (now includes LLM answer synthesis)."""

    @patch("app.services.prompting.generate_answer")
    def test_with_chunks(self, mock_generate):
        mock_generate.return_value = "Revenue grew by 20%."
        chunks = [
            {"filename": "report.pdf", "page": 1, "text": "Revenue grew 20%.", "score": 0.92},
            {"filename": "report.pdf", "page": 3, "text": "Expenses decreased.", "score": 0.85},
        ]
        result = build_prompt(query="What is the revenue?", chunks=chunks)

        assert result["answer"] == "Revenue grew by 20%."
        assert "user_message" in result
        assert "sources" in result
        assert result["user_message"] == "What is the revenue?"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["filename"] == "report.pdf"
        assert result["sources"][0]["page"] == 1
        assert result["sources"][0]["score"] == 0.92
        mock_generate.assert_called_once()

    @patch("app.services.prompting.generate_answer")
    def test_empty_chunks(self, mock_generate):
        """When no chunks are provided, the prompt should indicate no context found."""
        mock_generate.return_value = "I don't know."
        result = build_prompt(query="What is X?", chunks=[])
        assert result["user_message"] == "What is X?"
        assert result["sources"] == []
        assert result["answer"] == "I don't know."

    @patch("app.services.prompting.generate_answer")
    def test_system_prompt_contains_instruction(self, mock_generate):
        """System prompt should instruct the LLM to use only provided context."""
        mock_generate.return_value = "answer"
        build_prompt(query="test", chunks=[])
        system_prompt = mock_generate.call_args.kwargs["system_prompt"]
        assert "ONLY" in system_prompt
        assert "don't know" in system_prompt.lower() or "do not make up" in system_prompt.lower()

    @patch("app.services.prompting.generate_answer")
    def test_source_labels_in_context(self, mock_generate):
        mock_generate.return_value = "answer"
        chunks = [{"filename": "doc.pdf", "page": 5, "text": "Important finding.", "score": 0.9}]
        build_prompt(query="findings?", chunks=chunks)
        system_prompt = mock_generate.call_args.kwargs["system_prompt"]
        assert "doc.pdf" in system_prompt
        assert "page 5" in system_prompt

    @patch("app.services.prompting.generate_answer")
    def test_multiple_chunks_numbered(self, mock_generate):
        mock_generate.return_value = "answer"
        chunks = [
            {"filename": "a.pdf", "page": 1, "text": "First chunk.", "score": 0.9},
            {"filename": "b.pdf", "page": 2, "text": "Second chunk.", "score": 0.8},
            {"filename": "c.pdf", "page": 3, "text": "Third chunk.", "score": 0.7},
        ]
        build_prompt(query="test", chunks=chunks)
        system_prompt = mock_generate.call_args.kwargs["system_prompt"]
        assert "[1]" in system_prompt
        assert "[2]" in system_prompt
        assert "[3]" in system_prompt

    @patch("app.services.prompting.generate_answer")
    def test_return_structure(self, mock_generate):
        mock_generate.return_value = "answer"
        result = build_prompt(query="q", chunks=[])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"answer", "user_message", "sources"}
        assert isinstance(result["answer"], str)
        assert isinstance(result["user_message"], str)
        assert isinstance(result["sources"], list)

    @patch("app.services.prompting.generate_answer")
    def test_generate_answer_receives_correct_args(self, mock_generate):
        """Verify generate_answer is called with the assembled system_prompt and query."""
        mock_generate.return_value = "synthesized answer"
        build_prompt(query="my question", chunks=[])

        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["user_message"] == "my question"
        assert "Context:" in call_kwargs["system_prompt"]
