"""Unit tests for app.services.generation module.

Tests cover:
- generate_answer(): LLM answer generation via HuggingFace Inference API (mocked)
- get_llm_client(): lazy singleton initialization
"""
import pytest
from unittest.mock import patch, MagicMock

from app.services.generation import generate_answer, get_llm_client


class TestGenerateAnswer:
    """Tests for the generate_answer function with mocked InferenceClient."""

    @patch("app.services.generation.get_llm_client")
    def test_generates_answer(self, mock_client_fn):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Revenue grew by 20% last quarter."
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = generate_answer(
            system_prompt="You are a helpful assistant.\n\nContext:\nRevenue grew 20%.",
            user_message="What is the revenue growth?",
        )

        assert result == "Revenue grew by 20% last quarter."
        mock_client.chat.completions.create.assert_called_once()

    @patch("app.services.generation.get_llm_client")
    def test_passes_correct_messages(self, mock_client_fn):
        """Verify the system and user messages are passed correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        generate_answer(system_prompt="sys prompt", user_message="user msg")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "sys prompt"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "user msg"

    @patch("app.services.generation.get_llm_client")
    def test_uses_config_settings(self, mock_client_fn):
        """Verify max_tokens and temperature are passed from settings."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        generate_answer(system_prompt="sys", user_message="msg")

        call_kwargs = mock_client.chat.completions.create.call_args
        from app.core.config import settings
        assert call_kwargs.kwargs.get("max_tokens") == settings.llm_max_tokens
        assert call_kwargs.kwargs.get("temperature") == settings.llm_temperature

    @patch("app.services.generation.get_llm_client")
    def test_empty_response(self, mock_client_fn):
        """Handle empty LLM responses gracefully."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = generate_answer(system_prompt="sys", user_message="msg")
        assert result == ""


class TestGetLlmClient:
    """Tests for the lazy singleton client loader."""

    @patch("app.services.generation.InferenceClient")
    def test_creates_client_with_settings(self, mock_inference_cls):
        """Verify client is created with correct model and token."""
        import app.services.generation as gen_module
        gen_module._client = None  # Reset singleton

        mock_inference_cls.return_value = MagicMock()
        client = get_llm_client()

        mock_inference_cls.assert_called_once()
        call_kwargs = mock_inference_cls.call_args
        from app.core.config import settings
        assert call_kwargs.kwargs.get("model") == settings.llm_model

        # Clean up singleton
        gen_module._client = None

    @patch("app.services.generation.InferenceClient")
    def test_singleton_pattern(self, mock_inference_cls):
        """Client should only be created once (singleton)."""
        import app.services.generation as gen_module
        gen_module._client = None  # Reset singleton

        mock_inference_cls.return_value = MagicMock()
        client1 = get_llm_client()
        client2 = get_llm_client()

        assert client1 is client2
        mock_inference_cls.assert_called_once()

        # Clean up singleton
        gen_module._client = None
