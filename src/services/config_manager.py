"""Configuration management service for user settings."""

import streamlit as st
from typing import Dict, Optional, List
from config.models import PROVIDERS
from services.practice import load_flashcard_history
from services.battle import load_battle_history, format_history_option_label


def get_user_settings() -> Dict:
    """Get user settings from session state."""
    if "user_settings" not in st.session_state:
        st.session_state.user_settings = {
            "llm_provider": "GROQ",
            "model": "llama-3.1-8b-instant",
            "temperature": 0.5,
            "max_tokens": 1024,
        }
    return st.session_state.user_settings


def update_user_settings(settings: Dict):
    """Update user settings in session state."""
    st.session_state.user_settings = settings


def get_available_models() -> List[str]:
    """Get list of available models."""
    models = []
    for provider_name, provider_config in PROVIDERS.items():
        model_name = provider_config.get("MODEL", "")
        if model_name:
            models.append(f"{provider_name}: {model_name}")
    # Add Groq-specific models
    models.extend([
        "GROQ: llama-3.1-8b-instant",
        "GROQ: llama-3.1-70b-versatile",
        "GROQ: mixtral-8x7b-32768",
    ])
    return sorted(set(models))


def get_flashcard_statistics() -> Dict:
    """Get statistics about flashcard usage."""
    history = load_flashcard_history()
    if not history:
        return {
            "total_sessions": 0,
            "total_flashcards": 0,
            "topics_covered": 0,
            "modes_used": {},
            "average_flashcards_per_session": 0,
        }

    topics = set()
    modes = {}
    total_flashcards = 0
    
    # History now contains sessions, each with multiple flashcards
    for session in history:
        topic = session.get("topic", "Unknown")
        topics.add(topic)
        
        flashcards = session.get("flashcards", [])
        total_flashcards += len(flashcards)
        
        for flashcard in flashcards:
            mode = flashcard.get("mode", "unknown")
            modes[mode] = modes.get(mode, 0) + 1
    
    avg_flashcards = total_flashcards / len(history) if history else 0

    return {
        "total_sessions": len(history),
        "total_flashcards": total_flashcards,
        "topics_covered": len(topics),
        "modes_used": modes,
        "average_flashcards_per_session": round(avg_flashcards, 1),
    }


def get_battle_statistics() -> Dict:
    """Get statistics about battle mode usage."""
    history = load_battle_history()
    if not history:
        return {
            "total_battles": 0,
            "concepts_covered": 0,
            "average_questions": 0,
        }

    concepts = set()
    total_questions = 0
    for entry in history:
        concepts.add(entry.get("concept", "Unknown"))
        questions = entry.get("questions", [])
        total_questions += len(questions)

    avg_questions = total_questions / len(history) if history else 0

    return {
        "total_battles": len(history),
        "concepts_covered": len(concepts),
        "average_questions": round(avg_questions, 1),
    }

