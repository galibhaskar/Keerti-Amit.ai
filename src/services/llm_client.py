"""LLM client management service."""

import streamlit as st
from groq import Groq
from typing import Optional


@st.cache_resource
def get_groq_client(api_key: Optional[str] = None) -> Groq:
    """
    Get or create a cached Groq client.

    Args:
        api_key: Optional API key. If not provided, uses st.secrets["GROQ_API_KEY"]

    Returns:
        Groq client instance
    """
    if api_key is None:
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in secrets")
    return Groq(api_key=api_key)


def get_default_model() -> str:
    """Get the default model name from config or secrets."""
    return st.secrets.get("DEFAULT_MODEL", "llama-3.1-8b-instant")


def generate_groq_response(
    client: Groq,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
) -> str:
    """
    Generate a response using Groq API.

    Args:
        client: Groq client instance
        prompt: User prompt
        system_prompt: System prompt
        model: Model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        response_format: Optional response format (e.g., {"type": "json_object"})

    Returns:
        Generated response text
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if response_format:
        kwargs["response_format"] = response_format

    try:
        chat_completion = client.chat.completions.create(**kwargs)
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return "ERROR_API_FAILED"

