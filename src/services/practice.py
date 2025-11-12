"""Practice mode service for flashcard generation and logging."""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional
import streamlit as st

from config.settings import FLASHCARD_LOG_FILE
from services.llm_client import get_groq_client, generate_groq_response, get_default_model
from services.retrieval import retrieve_context_chroma


def save_flashcard_to_session(
    topic: str,
    mode: str,
    data: dict,
    session_id: str = None,
    log_file: str = None,
) -> str:
    """
    Save flashcard data to a practice session.
    If session_id is provided, adds flashcard to that session.
    Otherwise, creates a new session.

    Args:
        topic: Topic name
        mode: Difficulty mode
        data: Flashcard data
        session_id: Optional session ID to add to existing session
        log_file: Optional log file path

    Returns:
        Session ID (existing or newly created)
    """
    if log_file is None:
        log_file = FLASHCARD_LOG_FILE

    # Load existing sessions
    sessions = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    sessions_data = json.loads(content)
                    # Handle both old format (list of flashcards) and new format (list of sessions)
                    if sessions_data and isinstance(sessions_data[0], dict):
                        if "flashcards" in sessions_data[0]:
                            # New format: list of sessions
                            sessions = sessions_data
                        else:
                            # Old format: list of individual flashcards - convert to sessions
                            sessions = _convert_old_format_to_sessions(sessions_data)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Error reading {log_file}: {e}. Starting fresh.")
            sessions = []

    current_time = datetime.now(timezone.utc)
    flashcard_entry = {
        "mode": mode,
        "timestamp": current_time.isoformat(),
        "quiz_data": data,
    }

    # Find or create session
    if session_id:
        # Find existing session
        session_found = False
        for session in sessions:
            if session.get("session_id") == session_id:
                session["flashcards"].append(flashcard_entry)
                session["updated_at"] = current_time.isoformat()
                session_found = True
                break
        if not session_found:
            # Session not found, create new one
            session_id = None

    if not session_id:
        # Create new session
        session_id = f"session_{current_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        new_session = {
            "session_id": session_id,
            "topic": topic,
            "user_id": st.session_state.get("user_id", "local_user"),
            "started_at": current_time.isoformat(),
            "updated_at": current_time.isoformat(),
            "flashcards": [flashcard_entry],
        }
        sessions.append(new_session)

    # Save sessions
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, indent=4)
        st.toast(f"Flashcard saved to session", icon="💾")
    except IOError as e:
        st.error(f"Error writing to local file {log_file}: {e}")

    return session_id


def _convert_old_format_to_sessions(flashcards: List[Dict]) -> List[Dict]:
    """
    Convert old format (list of individual flashcards) to new format (sessions with flashcards).
    Groups flashcards by topic and time proximity (within 1 hour).

    Args:
        flashcards: List of individual flashcard entries

    Returns:
        List of session dictionaries
    """
    if not flashcards:
        return []

    sessions = []
    # Sort by timestamp
    sorted_flashcards = sorted(
        flashcards,
        key=lambda x: x.get("timestamp", ""),
    )

    current_session = None
    for flashcard in sorted_flashcards:
        topic = flashcard.get("topic", "Unknown")
        timestamp_str = flashcard.get("timestamp", "")
        
        try:
            if isinstance(timestamp_str, str):
                flashcard_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                flashcard_time = timestamp_str
        except (ValueError, AttributeError):
            flashcard_time = datetime.now(timezone.utc)

        # Check if we should start a new session
        if current_session is None:
            # Start new session
            current_session = {
                "session_id": f"session_{flashcard_time.strftime('%Y%m%d_%H%M%S')}_migrated",
                "topic": topic,
                "user_id": flashcard.get("user_id", "local_user"),
                "started_at": flashcard_time.isoformat(),
                "updated_at": flashcard_time.isoformat(),
                "flashcards": [],
            }
        else:
            # Check if this flashcard belongs to current session
            # (same topic and within 1 hour of last flashcard in session)
            last_flashcard_time_str = current_session["flashcards"][-1].get("timestamp", "")
            try:
                if isinstance(last_flashcard_time_str, str):
                    last_time = datetime.fromisoformat(last_flashcard_time_str.replace('Z', '+00:00'))
                else:
                    last_time = last_flashcard_time_str
            except (ValueError, AttributeError):
                last_time = flashcard_time

            time_diff = abs((flashcard_time - last_time).total_seconds())
            same_topic = current_session["topic"] == topic
            within_hour = time_diff < 3600  # 1 hour in seconds

            if not (same_topic and within_hour):
                # Save current session and start new one
                sessions.append(current_session)
                current_session = {
                    "session_id": f"session_{flashcard_time.strftime('%Y%m%d_%H%M%S')}_migrated",
                    "topic": topic,
                    "user_id": flashcard.get("user_id", "local_user"),
                    "started_at": flashcard_time.isoformat(),
                    "updated_at": flashcard_time.isoformat(),
                    "flashcards": [],
                }

        # Add flashcard to current session
        # Handle both old format (quiz_data might be at top level) and new format
        quiz_data = flashcard.get("quiz_data")
        if not quiz_data:
            # Old format might have data at top level, extract relevant fields
            quiz_data = {
                "quiz_text": flashcard.get("quiz_text", ""),
                "answer": flashcard.get("answer", ""),
                "flashcard_concept": flashcard.get("flashcard_concept", ""),
                "flashcard_rationale": flashcard.get("flashcard_rationale", ""),
            }
        
        current_session["flashcards"].append({
            "mode": flashcard.get("mode", "concept"),
            "timestamp": flashcard.get("timestamp", flashcard_time.isoformat()),
            "quiz_data": quiz_data,
        })
        current_session["updated_at"] = flashcard_time.isoformat()

    # Don't forget the last session
    if current_session:
        sessions.append(current_session)

    return sessions


def save_flashcard_to_json(topic: str, mode: str, data: dict, session_id: str = None, log_file: str = None) -> str:
    """
    Save flashcard data to a practice session (wrapper for backward compatibility).

    Args:
        topic: Topic name
        mode: Difficulty mode
        data: Flashcard data
        session_id: Optional session ID to add to existing session
        log_file: Optional log file path

    Returns:
        Session ID
    """
    return save_flashcard_to_session(topic, mode, data, session_id, log_file)


def load_flashcard_history(log_file: str = None) -> List[Dict]:
    """
    Load practice session history from JSON file.

    Args:
        log_file: Optional log file path

    Returns:
        List of practice session entries (each containing multiple flashcards)
    """
    if log_file is None:
        log_file = FLASHCARD_LOG_FILE

    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                sessions_data = json.loads(content)
                # Handle both old format (list of flashcards) and new format (list of sessions)
                if sessions_data and isinstance(sessions_data[0], dict):
                    if "flashcards" in sessions_data[0]:
                        # New format: list of sessions
                        return sessions_data
                    else:
                        # Old format: list of individual flashcards - convert to sessions
                        return _convert_old_format_to_sessions(sessions_data)
                return []
    except (json.JSONDecodeError, IOError) as e:
        st.warning(f"Error reading {log_file}: {e}")
    return []


def save_flashcard_history(history: List[Dict], log_file: str = None) -> None:
    """
    Save practice session history to file.

    Args:
        history: List of practice session dictionaries
        log_file: Optional log file path
    """
    if log_file is None:
        log_file = FLASHCARD_LOG_FILE

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        st.warning(f"Unable to save flashcard history: {e}")


def format_history_option_label(index: int, entry: Dict) -> str:
    """
    Format a practice session entry for display in selectbox.

    Args:
        index: Index of the entry in the original list
        entry: Practice session dictionary

    Returns:
        Formatted label string
    """
    topic = entry.get("topic") or "Untitled Topic"
    saved_at = entry.get("started_at") or entry.get("updated_at") or entry.get("timestamp")
    flashcards_count = len(entry.get("flashcards", []))
    
    if saved_at:
        try:
            # Try parsing ISO format timestamp
            if isinstance(saved_at, str):
                # Handle both with and without timezone
                if 'Z' in saved_at:
                    stamp = datetime.fromisoformat(saved_at.replace('Z', '+00:00'))
                elif '+' in saved_at or saved_at.endswith('UTC'):
                    stamp = datetime.fromisoformat(saved_at)
                else:
                    # Try without timezone
                    stamp = datetime.fromisoformat(saved_at)
                    stamp = stamp.replace(tzinfo=timezone.utc)
            else:
                stamp = saved_at
            if hasattr(stamp, 'astimezone'):
                display = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            else:
                display = str(saved_at)
        except (ValueError, AttributeError):
            display = str(saved_at)
        return f"{topic} ({flashcards_count} flashcard{'s' if flashcards_count != 1 else ''}) - {display}"
    return f"{topic} ({flashcards_count} flashcard{'s' if flashcards_count != 1 else ''})"


def get_structured_quiz_prompt(
    topic: str,
    mode: str,
    context: str,
    previous_titles: List[str],
    current_concept_title: Optional[str] = None,
) -> str:
    """
    Generate a structured prompt for flashcard generation.

    Args:
        topic: Topic name
        mode: Difficulty mode (concept, easy, example)
        context: Retrieved context
        previous_titles: List of previous concept titles
        current_concept_title: Current concept title for easy/example modes

    Returns:
        Formatted prompt string
    """
    exclusion_instruction = ""
    current_concept = current_concept_title if current_concept_title else topic

    if mode == "concept" and previous_titles:
        title_list = "\n- ".join(previous_titles)
        exclusion_instruction = f"""
        **EXCLUSION RULE:** You MUST NOT generate a flashcard related to any of the following previous concepts/titles:
        - {title_list}
        """

    mode_instructions = {
        "concept": f"Your task is to generate one detailed flashcard focused on a **new, challenging concept** within: **{topic}**.",
        "easy": f"Your task is to generate a flashcard for the **EXACT SAME CONCEPT**: **'{current_concept}'**, but with a **significantly simpler explanation and analogy**.",
        "example": f"Your task is to generate a flashcard for the **EXACT SAME CONCEPT**: **'{current_concept}'**, focusing **exclusively on a practical code example or a real-world use case** to illustrate the principle."
    }

    prompt = f"""
    You are an expert in the given topic. 
    
    {exclusion_instruction}
    
    You MUST use ONLY the following **CONTEXT** from the course material to formulate the flashcard content. 
You are FORBIDDEN from using any external knowledge. 
If the CONTEXT is missing, off-topic, noisy, or insufficient, you MUST return the NO_CONTEXT JSON exactly as specified below.

    
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---

    **STRICT RAG ENFORCEMENT (must follow exactly):**
    Treat the text above as your only source of truth. Before generating anything, evaluate CONTEXT quality and topic match:

    **Insufficient/Invalid CONTEXT if ANY of these are true (case-insensitive):**
    - The exact topic word "{topic}" does NOT appear anywhere in the CONTEXT.
    - CONTEXT includes phrases like "No relevant context found" or "RAG Retrieval Error".
    - CONTEXT contains long runs of non-alphanumeric noise (e.g., 5+ repeated symbols such as @@@@, ####, or random encodings).
    - CONTEXT length has fewer than 100 characters of meaningful prose relevant to "{topic}".

    **If CONTEXT is insufficient by the rules above, return EXACTLY this JSON object (and nothing else):**
    {{
        "quiz_type": "Flashcard Content",
        "quiz_text": "No context available",
        "answer": "N/A - See Rationale",
        "flashcard_concept": "No context available",
        "flashcard_rationale": "Please upload relevant documents for this topic to your knowledge base and try again."
    }}

    **If CONTEXT is sufficient, proceed.**

    {mode_instructions.get(mode, mode_instructions['concept'])}

    **Provenance rule:** Use ONLY details present in CONTEXT. Do not invent, pad, or generalize beyond it. Avoid quoting any garbled/noisy substrings; summarize them in clean prose or omit them.

    The output must be a single JSON object.

    **CRITICAL FORMATTING INSTRUCTIONS (Read Carefully):**
    1. **JSON ESCAPING:** The content of 'flashcard_rationale' must be a single, valid JSON string. ALL content, including Markdown, code blocks, **newlines**, and **double quotes**, MUST be correctly JSON-escaped (e.g., newline becomes `\\n`, double quote becomes `\\"`).
    2. **STRICT FONT FIX:** DO NOT use any Markdown headings (`#`, `##`, `###`, etc.) within the 'flashcard_rationale'. Use **bold text** or *italics* instead for emphasis to avoid large fonts.
    3. **CODE BLOCKS:** If you use a code block (e.g., ```java\\n...\\n```), ensure every character, including the three backticks, the language name, and the code lines, are represented in a single, properly escaped JSON string field.

    **FINAL MANDATE:** The entire value of the `flashcard_rationale` key MUST be a valid JSON string that begins and ends with a double quote (") and has all internal double quotes and newlines escaped.

    Output the following JSON structure. Do not include any other text:

    {{
        "quiz_type": "Flashcard Content",
        "quiz_text": "A brief summary or header for the flashcard's core focus (e.g., 'The role of the DOM in JS').",
        "answer": "N/A - This field is not used for quiz answers, put 'See Rationale'",
        "flashcard_concept": "A concise, detailed title for the concept being reinforced.",
        "flashcard_rationale": "A detailed explanation of the concept/principle/example, formatted in Markdown for easy reading. (Remember: Use **bold** instead of # headings and ensure all internal characters like \\n and \\" are properly escaped)."
    }}

    """
    return prompt


def generate_flashcard(
    topic: str,
    mode: str,
    previous_titles: List[str],
    current_concept_title: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict:
    """
    Generate a flashcard using LLM.

    Args:
        topic: Topic name
        mode: Difficulty mode
        previous_titles: List of previous concept titles
        current_concept_title: Current concept title
        model: Optional model name

    Returns:
        Generated flashcard data
    """
    if model is None:
        model = get_default_model()

    # Use retrieve_context_langchain instead of retrieve_context_chroma
    # It uses the same vectorstore that's used for ingestion and is more reliable
    from services.retrieval import retrieve_context_langchain
    context = retrieve_context_langchain(topic, n_results=5)  # Get more results for better context
    
    prompt = get_structured_quiz_prompt(
        topic, mode, context, previous_titles, current_concept_title
    )

    system_prompt = (
        "You are a strict RAG assistant. Use ONLY the text inside the CONTEXT block. "
        "If the CONTEXT is missing, off-topic, or low quality, you MUST output the exact NO_CONTEXT JSON specified by the user message. "
        "Never use prior knowledge or add information not in CONTEXT. Respond with a single JSON object only."
    )

    client = get_groq_client()
    response = generate_groq_response(
        client=client,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.5,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    if response == "ERROR_API_FAILED":
        raise ValueError("API failed. Cannot generate flashcard.")

    return response

