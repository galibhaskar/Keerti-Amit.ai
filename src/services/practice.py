"""Practice mode service for flashcard generation and logging."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st

from config.settings import FLASHCARD_LOG_FILE
from services.llm_client import get_groq_client, generate_groq_response, get_default_model
from services.retrieval import retrieve_context_chroma


def save_flashcard_to_json(topic: str, mode: str, data: dict, log_file: str = None):
    """
    Save flashcard data to a JSON log file.

    Args:
        topic: Topic name
        mode: Difficulty mode
        data: Flashcard data
        log_file: Optional log file path
    """
    if log_file is None:
        log_file = FLASHCARD_LOG_FILE

    log_entry = {
        "user_id": st.session_state.get("user_id", "local_user"),
        "topic": topic,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "quiz_data": data,
    }

    all_logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_logs = json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Error reading {log_file}: {e}. Starting a fresh log.")
            all_logs = []

    all_logs.append(log_entry)

    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=4)
        st.toast(f"Flashcard saved to {log_file}", icon="💾")
    except IOError as e:
        st.error(f"Error writing to local file {log_file}: {e}")


def load_flashcard_history(log_file: str = None) -> List[Dict]:
    """
    Load flashcard history from JSON file.

    Args:
        log_file: Optional log file path

    Returns:
        List of flashcard entries
    """
    if log_file is None:
        log_file = FLASHCARD_LOG_FILE

    if not os.path.exists(log_file):
        return []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        st.warning(f"Error reading {log_file}: {e}")
    return []


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

    context = retrieve_context_chroma(topic)
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

