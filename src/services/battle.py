"""Battle mode service for battle history and prompt generation."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from config.settings import BATTLE_HISTORY_FILE
from services.llm_client import get_groq_client, generate_groq_response, get_default_model


# Constants
MAX_INTERVIEW_QUESTIONS = 3
QUESTION_AUDIO_MIME = "audio/mp3"
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
SOURCE_PATTERN = re.compile(r"^Source:\s*[^\n]+", re.MULTILINE)


def load_battle_history() -> List[Dict]:
    """Load battle history from file."""
    history_file = Path(BATTLE_HISTORY_FILE)
    if not history_file.exists():
        return []
    try:
        data = json.loads(history_file.read_text())
        if isinstance(data, list):
            return data
    except Exception as exc:
        st.warning(f"Unable to read saved battle history: {exc}")
    return []


def save_battle_history(history: List[Dict]) -> None:
    """Save battle history to file."""
    history_file = Path(BATTLE_HISTORY_FILE)
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = [_serialize_history_entry(entry) for entry in history]
        history_file.write_text(json.dumps(serializable, indent=2))
    except Exception as exc:
        st.warning(f"Unable to save battle history: {exc}")


def _strip_unserializable_turn(turn: Dict[str, str]) -> Dict[str, str]:
    """Remove unserializable fields from a turn."""
    return {k: v for k, v in turn.items() if k != "answer_audio"}


def _serialize_history_entry(entry: Dict) -> Dict:
    """Serialize a history entry for storage."""
    return {
        "concept": entry.get("concept"),
        "context": entry.get("context", ""),
        "questions": entry.get("questions", []),
        "transcript": [_strip_unserializable_turn(turn) for turn in entry.get("transcript", [])],
        "assessments": entry.get("assessments", []),
        "feedback": entry.get("feedback", {}),
        "saved_at": entry.get("saved_at"),
    }


def format_history_option_label(index: int, entry: Dict) -> str:
    """Format a history entry for display in selectbox."""
    concept = entry.get("concept") or "Untitled Battle"
    saved_at = entry.get("saved_at")
    label_idx = index + 1
    if saved_at:
        try:
            stamp = datetime.fromisoformat(saved_at)
            display = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            display = saved_at
        return f"Battle {label_idx}: {concept} ({display})"
    return f"Battle {label_idx}: {concept}"


def clean_context_for_battle(context: str) -> str:
    """
    Clean context by removing document source references and document name mentions.
    This ensures the LLM doesn't reference specific documents in questions.

    Args:
        context: Raw context string with source references

    Returns:
        Cleaned context without source references or document names
    """
    if not context:
        return context
    
    # Remove "Source: filename.pdf" lines (handles both "Source: filename" and "Source:filename")
    cleaned = SOURCE_PATTERN.sub("", context)
    
    # Remove "Retrieved Context:" header if present
    cleaned = re.sub(r"^Retrieved Context:\s*", "", cleaned, flags=re.MULTILINE)
    
    # Remove document file extensions with common patterns
    # Pattern 1: "as mentioned in doc1.pdf" or "according to document.pdf"
    cleaned = re.sub(
        r"\b(?:as\s+)?(?:mentioned|described|discussed|stated|explained|outlined|covered|presented|shown|indicated|noted|detailed|provided|given|found|listed|defined|referenced|written|documented|found)\s+(?:in|on|within|from|according\s+to|per|based\s+on)\s+[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\b",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Pattern 2: "see doc1.pdf" or "refer to document.pdf" or "check document.pdf"
    cleaned = re.sub(
        r"\b(?:see|refer\s+to|check|look\s+at|review|consult|examine|open|read|view|access)\s+(?:the\s+)?[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\b",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Pattern 3: "in doc1.pdf" or "from document.pdf"
    cleaned = re.sub(
        r"\b(?:in|from|of|at)\s+[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\b",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Pattern 4: Standalone document names in quotes, parentheses, or brackets
    # Match opening brackets/parentheses/quotes, then filename, then closing
    # Split into separate patterns to avoid SyntaxWarning with escape sequences
    # Match parentheses: (filename.pdf) or [filename.pdf] or {filename.pdf}
    cleaned = re.sub(r'[\(\[{]\s*[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\s*[\)\]}]', '', cleaned, flags=re.IGNORECASE)
    # Match quotes: "filename.pdf" or 'filename.pdf'
    cleaned = re.sub(r'["\']\s*[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\s*["\']', '', cleaned, flags=re.IGNORECASE)
    
    # Pattern 5: Document references at the start or end of lines
    cleaned = re.sub(
        r"^\s*[a-zA-Z0-9_\-]+\.(?:pdf|txt|docx?|md|json)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE
    )
    
    # Pattern 6: Remove standalone document names with file extensions (conservative)
    # Only matches if it appears after whitespace/punctuation and looks like a document reference
    # This catches remaining edge cases without removing legitimate technical terms
    cleaned = re.sub(
        r"(?<![a-zA-Z0-9])(?:doc\d+|document\d*|[a-zA-Z0-9_\-]{1,30}\.(?:pdf|txt|docx?|md|json))(?![a-zA-Z0-9])",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    
    # Clean up separators that had source info
    cleaned = re.sub(r"^\s*---\s*$", "", cleaned, flags=re.MULTILINE)
    
    # Clean up excessive whitespace and empty lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)  # Multiple spaces to single space
    cleaned = cleaned.strip()
    
    return cleaned


def prepare_question_content(raw_text: str) -> tuple[str, str]:
    """
    Prepare question content for display and audio.

    Args:
        raw_text: Raw question text

    Returns:
        Tuple of (display_text, audio_text)
    """
    display_text = raw_text.strip().replace("\"", "'")

    def block_replacer(match: re.Match) -> str:
        return " Refer to the provided code snippet. "

    audio_text = CODE_BLOCK_PATTERN.sub(block_replacer, raw_text)
    audio_text = INLINE_CODE_PATTERN.sub(" refer to the highlighted code segment ", audio_text)
    audio_text = audio_text.replace("\"", "'")
    audio_text = re.sub(r"\s+", " ", audio_text).strip()

    if not audio_text:
        audio_text = "Refer to the provided code snippet and share your thoughts."

    return display_text, audio_text


def build_initial_question_prompt(concept: str, context: str) -> str:
    """Build prompt for initial question generation."""
    # Clean context to remove document source references
    cleaned_context = clean_context_for_battle(context)
    
    return f"""
You are Amit, a senior technical interviewer hiring for a high-impact role. You are preparing to assess the candidate Keerti on the concept below.

Concept focus: {concept.strip()}

Reference material (may be noisy or empty):
{cleaned_context}

Craft the very first interview question Amit will ask Keerti. Stay grounded in the reference material—cite specific situations or facts from it when framing the prompt.

Guidelines:
1. Use a scenario anchored in the reference material, not a generic definition question.
2. Aim to surface both conceptual understanding and practical judgment.
3. Keep the scenario vivid, short, and practical—treat it like an on-the-spot hiring challenge.
4. Start with an accessible difficulty. Set `"difficulty": "foundation"` and design the scenario so a prepared candidate can ease in.
5. Use clear language but slip in a gentle twist or constraint that reveals whether Keerti is truly thinking.
6. Avoid quoting long noisy strings verbatim. If you must reference raw text, paraphrase it and replace any double quotes with single quotes.
7. **CRITICAL: Never mention any document names, file names, or source references (like "doc1.pdf", "document.pdf", etc.) in your question. Act as if the reference material is general knowledge, not from a specific document.**

Respond ONLY with valid JSON:
{{
  "question": "Thoughtful, scenario-based question text grounded in the reference material.",
  "focus_area": "Skill or competency you are testing.",
  "difficulty": "foundation|intermediate|advanced",
  "interviewer_goal": "What you expect to learn from the answer.",
  "evidence_anchor": "Short note on which parts of the reference material inspired the question."
}}
"""


def build_followup_question_prompt(
    concept: str,
    context: str,
    transcript: List[Dict[str, str]],
    latest_assessment: Optional[Dict[str, str]],
) -> str:
    """Build prompt for follow-up question generation."""
    # Clean context to remove document source references
    cleaned_context = clean_context_for_battle(context)
    
    transcript_lines = []
    for idx, turn in enumerate(transcript, start=1):
        transcript_lines.append(
            f"Question {idx}: {turn.get('question', '').strip()}\n"
            f"Learner answer {idx} (voice transcript): {turn.get('answer', '').strip()}"
        )
    transcript_block = "\n\n".join(transcript_lines)

    assessment_context = ""
    if latest_assessment:
        assessment_context = (
            f"\nLatest assessment:\n"
            f"- Understanding level: {latest_assessment.get('understanding_level', 'n/a')}\n"
            f"- Decision quality: {latest_assessment.get('decision_quality', 'n/a')}\n"
            f"- Suggested difficulty change: {latest_assessment.get('difficulty_adjustment', 'same')}\n"
            f"- Rationale: {latest_assessment.get('rationale', '')}\n"
            f"- Next focus: {latest_assessment.get('next_focus', '')}\n"
            f"- Interesting hook: {latest_assessment.get('interesting_hook', '')}\n"
        )

    return f"""
You are Amit continuing a technical interview for a high-stakes role with candidate Keerti.

Concept focus: {concept.strip()}

Reference material (may be noisy or empty):
{cleaned_context}

Conversation so far:
{transcript_block}

{assessment_context}

Craft the next follow-up question. It must:
1. React to the learner's most recent answer—press into gaps, assumptions, or opportunities you detect.
2. Keep the scenario short, vivid, and grounded in the reference material (cite specific details or cases).
3. Adjust the difficulty based on the latest assessment guidance (push harder if marked "harder", reinforce fundamentals if "easier").
4. Probe a different angle or deeper layer than previous questions (edge cases, integration, failure modes, stakeholder impact, time pressure, trade-offs).
5. Blend simple wording with a tricky constraint so Keerti must think aloud to navigate it.
6. Stay scenario-based; avoid rote definition or trivia.
7. Avoid copying raw noisy strings verbatim; paraphrase and swap any double quotes for single quotes.
8. **CRITICAL: Never mention any document names, file names, or source references (like "doc1.pdf", "document.pdf", etc.) in your question. Act as if the reference material is general knowledge, not from a specific document.**

Respond ONLY with valid JSON:
{{
  "question": "Next scenario-driven question tied to the learner's last response and the reference material.",
  "focus_area": "Skill or competency you are targeting now.",
  "difficulty": "foundation|intermediate|advanced",
  "interviewer_goal": "What you expect to learn from the answer.",
  "evidence_anchor": "Short note on which parts of the reference material or learner response prompted this follow-up."
}}
"""


def build_answer_assessment_prompt(
    concept: str,
    question: Dict[str, str],
    learner_transcript: str,
    context: str,
) -> str:
    """Build prompt for answer assessment."""
    # Clean context to remove document source references
    cleaned_context = clean_context_for_battle(context)
    
    return f"""
You are Amit benchmarking Keerti's spoken answer during a live technical interview.

Concept focus: {concept.strip()}

Reference material (may be noisy or empty):
{cleaned_context}

Question asked:
{question.get("question", "").strip()}

Keerti's transcript:
{learner_transcript.strip()}

Judge how well the candidate navigated the scenario.

Respond ONLY with valid JSON:
{{
  "understanding_level": "strong|solid|partial|unclear",
  "decision_quality": "excellent|good|mixed|risky",
  "difficulty_adjustment": "harder|same|easier",
  "rationale": "One sentence referencing specific parts of the answer.",
  "next_focus": "Which scenario angle to probe next.",
  "interesting_hook": "A short, vivid situation seed to keep the interview engaging."
}}
"""


def build_interview_evaluation_prompt(
    concept: str,
    questions: List[Dict[str, str]],
    transcript: List[Dict[str, str]],
    context: str,
    assessments: List[Dict[str, str]],
) -> str:
    """Build prompt for final interview evaluation."""
    # Clean context to remove document source references
    cleaned_context = clean_context_for_battle(context)
    
    question_list = "\n".join(
        f"- Q{item.get('id', idx + 1)} ({item.get('difficulty', 'n/a')}) "
        f"[{item.get('focus_area', 'unspecified')}]: {item.get('question', '')} "
        f"(context hook: {item.get('evidence_anchor', 'n/a')})"
        for idx, item in enumerate(questions)
    )

    transcript_lines = []
    for idx, turn in enumerate(transcript, start=1):
        transcript_lines.append(
            f"Question {idx}: {turn.get('question', '').strip()}\n"
            f"Learner answer {idx} (voice transcript): {turn.get('answer', '').strip()}"
        )
    transcript_block = "\n\n".join(transcript_lines)

    assessment_lines = []
    for idx, review in enumerate(assessments, start=1):
        assessment_lines.append(
            f"Q{review.get('question_id', idx)} assessment: "
            f"understanding={review.get('understanding_level', 'n/a')}, "
            f"decision={review.get('decision_quality', 'n/a')}, "
            f"difficulty_shift={review.get('difficulty_adjustment', 'n/a')} -> "
            f"{review.get('rationale', '')}"
        )
    assessments_block = "\n".join(assessment_lines) or "No per-question assessments captured."

    return f"""
You are Rival-Ally wrapping up an interview. Assess the learner's answers to determine conceptual mastery and practical readiness.

Concept defended: {concept.strip()}

Interview blueprint:
{question_list}

Reference material (may be noisy or empty):
{cleaned_context}

Interview transcript:
{transcript_block}

Per-question assessments:
{assessments_block}

Deliver a coaching-oriented debrief.

Respond ONLY with valid JSON:
{{
  "score": "excellent|good|fair|needs-work",
  "verdict": "One-sentence summary of how the learner performed overall.",
  "strengths": ["List crisp evidence-backed strengths. Use [] if none."],
  "improvement_opportunities": ["List targeted improvement areas. Use [] if none."],
  "improved_explanation": "Provide an upgraded mental model tying together the concept and scenarios.",
  "practice_prompt": "A short reflective drill or scenario to practice next.",
  "question_feedback": [
    {{
      "id": 1,
      "question": "Restate the question asked.",
      "assessment": "Discuss how well the learner handled it.",
      "improvements": ["Specific things to polish for this question."],
      "follow_up": "Optional follow-up practice prompt for this topic."
    }}
  ]
}}
"""


def generate_battle_llm_response(prompt: str, model: Optional[str] = None) -> str:
    """
    Generate LLM response for battle mode.

    Args:
        prompt: User prompt
        model: Optional model name

    Returns:
        Generated response
    """
    if model is None:
        model = get_default_model()

    system_prompt = (
        "You are Amit, the Rival-Ally interviewer. "
        "Guide Keerti through scenario-based battles by asking probing, real-world questions, "
        "evaluating her spoken responses, and coaching improvement. "
        "Always answer strictly as a JSON object. "
        "Never mention document names, file names, or source references in your questions or responses. "
        "Treat all reference material as general knowledge, not from specific documents."
    )

    client = get_groq_client()
    return generate_groq_response(
        client=client,
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=0.4,
        max_tokens=900,
        response_format={"type": "json_object"},
    )

