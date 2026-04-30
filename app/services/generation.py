"""LLM answer generation service.

Uses HuggingFace Inference API to generate answers from a system prompt
and user message via chat completion.
"""
from huggingface_hub import InferenceClient
from app.core.config import settings


# Lazy singleton for the inference client
_client = None


def get_llm_client() -> InferenceClient:
    """Lazy loader for HuggingFace InferenceClient."""
    global _client
    if _client is None:
        _client = InferenceClient(
            model=settings.llm_model,
            token=settings.hf_token,
        )
    return _client


def generate_answer(system_prompt: str, user_message: str) -> str:
    """Generate an answer using the configured LLM via HuggingFace Inference API.

    Args:
        system_prompt: The system prompt containing context from retrieved chunks.
        user_message: The user's original query.

    Returns:
        The generated answer text from the LLM.
    """
    client = get_llm_client()

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )

    return response.choices[0].message.content
