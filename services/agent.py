from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Literal, Sequence

from config.model_config import PROVIDERS, SYSTEM_PROMPTS

ProviderKey = Literal["OLLAMA", "GROQ"]
ChatHistory = Sequence[Dict[str, str]]


class LLMProviderNotAvailable(RuntimeError):
    """Raised when an LLM backend cannot be initialised."""


def _truncate(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return "Context not provided."
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _format_history(history: ChatHistory) -> str:
    if not history:
        return "No prior exchanges."

    formatted: List[str] = []
    for message in history:
        role = message.get("role", "")
        speaker = "Learner" if role == "user" else "Tutor"
        formatted.append(f"{speaker}: {message.get('content', '').strip()}")
    return "\n".join(formatted)


@lru_cache(maxsize=len(PROVIDERS))
def get_llm_model(selected_provider: ProviderKey = "OLLAMA"):
    provider_config = PROVIDERS.get(selected_provider)

    if not provider_config:
        raise ValueError(f"Unsupported provider: {selected_provider}")

    backend = provider_config.get("provider")
    model_name = provider_config.get("MODEL")

    try:
        if backend == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=model_name, temperature=0.3, repeat_penalty=1.1)

        if backend == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(model=model_name, temperature=0.3, repeat_penalty=1.1)
    except Exception as exc:  # pragma: no cover - depends on external backends
        raise LLMProviderNotAvailable(
            f"Failed to initialise provider '{selected_provider}': {exc}"
        ) from exc

    raise ValueError(f"Unsupported provider backend: {backend}")


def _call_llm(prompt: str, provider: ProviderKey = "OLLAMA") -> str:
    llm = get_llm_model(provider)
    response = llm.invoke(prompt)

    if isinstance(response, str):
        return response

    # For LangChain chat models the response exposes `.content`.
    return getattr(response, "content", str(response))


def generate_tutor_intro(
    topic: str,
    context: str | None,
    provider: ProviderKey = "OLLAMA",
) -> str:
    base_prompt = SYSTEM_PROMPTS["PRACTICE_TUTOR_PROMPT"].strip()
    crafted_prompt = f"""
{base_prompt}

Topic or learner goal:
{topic.strip()}

Reference material:
{_truncate(context)}

Task:
- Begin the study session with a short friendly greeting.
- Summarise what will be covered and ask one diagnostic question to gauge the learner's current understanding.
- Present the first flashcard (title + 2–3 bullet insights + 1 reflective prompt).
- Encourage the learner to respond before you continue.
"""
    return _call_llm(crafted_prompt, provider)


def generate_tutor_response(
    learner_message: str,
    topic: str,
    context: str | None,
    history: ChatHistory,
    provider: ProviderKey = "OLLAMA",
) -> str:
    base_prompt = SYSTEM_PROMPTS["PRACTICE_TUTOR_PROMPT"].strip()
    history_block = _format_history(history)

    crafted_prompt = f"""
{base_prompt}

Topic or learner goal:
{topic.strip()}

Reference material:
{_truncate(context)}

Conversation so far:
{history_block}

The learner just said:
{learner_message.strip()}

Respond as the tutor. Follow the teaching guidelines, build on prior steps,
and deliver the next flashcard (title, 2–3 concise insights, optional reflective prompt).
End with an invitation or question that keeps the learner engaged.
"""
    return _call_llm(crafted_prompt, provider)