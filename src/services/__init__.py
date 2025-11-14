"""Business service modules."""

from services.llm_client import get_groq_client, generate_groq_response, get_default_model
from services.retrieval import retrieve_context_langchain, retrieve_context_chroma
from services.practice import (
    generate_flashcard,
    save_flashcard_to_json,
    load_flashcard_history,
    save_flashcard_history,
    format_history_option_label as format_practice_history_option_label,
    get_structured_quiz_prompt,
)
from services.battle import (
    load_battle_history,
    save_battle_history,
    format_history_option_label,
    build_initial_question_prompt,
    build_followup_question_prompt,
    build_answer_assessment_prompt,
    build_interview_evaluation_prompt,
    prepare_question_content,
    generate_battle_llm_response,
)
from services.config_manager import (
    get_user_settings,
    update_user_settings,
    get_available_models,
    get_flashcard_statistics,
    get_battle_statistics,
)

__all__ = [
    # LLM Client
    "get_groq_client",
    "generate_groq_response",
    "get_default_model",
    # Retrieval
    "retrieve_context_langchain",
    "retrieve_context_chroma",
    # Practice
    "generate_flashcard",
    "save_flashcard_to_json",
    "load_flashcard_history",
    "save_flashcard_history",
    "format_practice_history_option_label",
    "get_structured_quiz_prompt",
    # Battle
    "load_battle_history",
    "save_battle_history",
    "format_history_option_label",
    "build_initial_question_prompt",
    "build_followup_question_prompt",
    "build_answer_assessment_prompt",
    "build_interview_evaluation_prompt",
    "prepare_question_content",
    "generate_battle_llm_response",
    # Config Manager
    "get_user_settings",
    "update_user_settings",
    "get_available_models",
    "get_flashcard_statistics",
    "get_battle_statistics",
]

