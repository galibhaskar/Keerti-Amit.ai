"""LLM client management service - refactored to use core/llm/provider."""

import streamlit as st
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm_model, LLMProviderNotAvailable, ProviderKey
from config.models import DEFAULT_PROVIDER, DEFAULT_MODEL


def get_default_model() -> str:
    """
    Get the default model name from config.
    
    Returns:
        Default model name from config.models.DEFAULT_MODEL
    """
    return DEFAULT_MODEL


def get_default_provider() -> ProviderKey:
    """
    Get the default provider from config.
    
    Returns:
        Default provider from config.models.DEFAULT_PROVIDER
        Falls back to "GROQ" if invalid provider specified
    """
    provider = DEFAULT_PROVIDER.upper()
    if provider not in ["GROQ", "OLLAMA"]:
        provider = "GROQ"
    return provider  # type: ignore


# Backward compatibility: Keep get_groq_client for existing code
# but it now returns the LangChain model instead
@st.cache_resource
def get_groq_client(api_key: Optional[str] = None):
    """
    Get or create a cached LLM model (backward compatibility wrapper).
    
    Note: This now returns a LangChain model, not a Groq client.
    For new code, use get_llm_model() directly.

    Args:
        api_key: Optional API key (kept for compatibility, not used)

    Returns:
        LangChain chat model instance
    """
    provider = get_default_provider()
    try:
        return get_llm_model(provider)
    except LLMProviderNotAvailable as e:
        st.error(f"LLM Provider Error: {e}")
        raise


def generate_groq_response(
    client,
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 1024,
    response_format: Optional[dict] = None,
) -> str:
    """
    Generate a response using LLM provider (refactored to use core/llm/provider).

    Args:
        client: LangChain chat model instance (from get_groq_client or get_llm_model)
        prompt: User prompt
        system_prompt: System prompt
        model: Model name (kept for compatibility, may be ignored if provider handles it)
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        response_format: Optional response format (e.g., {"type": "json_object"})

    Returns:
        Generated response text
    """
    try:
        # Build messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        
        # Bind parameters to the model (LangChain pattern)
        bound_client = client
        bind_kwargs = {}
        
        # Add temperature and max_tokens if supported
        if hasattr(bound_client, "bind"):
            bind_kwargs["temperature"] = temperature
            bind_kwargs["max_tokens"] = max_tokens
            
            try:
                bound_client = bound_client.bind(**bind_kwargs)
            except Exception:
                # If binding fails, continue with unbound client
                pass
        
        # Note: response_format is not supported via LangChain's bind() or invoke()
        # For Groq models, JSON format is enforced through the prompt/system message
        # The prompts in practice.py and battle.py already request JSON output
        # If strict JSON is required, consider using with_structured_output() or
        # accessing the underlying Groq client directly
        
        # Invoke the model
        response = bound_client.invoke(messages)
        
        # Extract content from response
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
            
    except Exception as e:
        st.error(f"LLM API Error: {e}")
        return "ERROR_API_FAILED"

