import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from groq import Groq
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from config.app_config import (
    COLLECTION_NAME,
    SEARCH_KWARGS,
    SENTENCE_TRANSFORMERS_MODEL,
    VECTOR_DB_PATH,
)
from services.audio import synthesize_speech, transcribe_audio


# --- 1. CONFIGURATION & CLIENT INITIALISATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    MODEL_NAME = "llama-3.1-8b-instant"

    @st.cache_resource
    def get_groq_client() -> Groq:
        return Groq(api_key=GROQ_API_KEY)

    client = get_groq_client()
except KeyError:
    st.error("🚨 Configuration Error: GROQ_API_KEY missing from .streamlit/secrets.toml.")
    st.stop()
except Exception as exc:  # pragma: no cover - Streamlit runtime feedback
    st.error(f"Initialization Error: {exc}")
    st.stop()


# --- 2. SESSION STATE MANAGEMENT ---
MAX_INTERVIEW_QUESTIONS = 3
QUESTION_AUDIO_MIME = "audio/mp3"
CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

BATTLE_HISTORY_FILE = Path("./.cache/battle_history.json")



def _load_persisted_battle_history() -> List[Dict]:
    if not BATTLE_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(BATTLE_HISTORY_FILE.read_text())
        if isinstance(data, list):
            return data
    except Exception as exc:
        st.warning(f"Unable to read saved battle history: {exc}")
    return []


def _strip_unserializable_turn(turn: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in turn.items() if k != "answer_audio"}


def _serialize_history_entry(entry: Dict) -> Dict:
    return {
        "concept": entry.get("concept"),
        "context": entry.get("context", ""),
        "questions": entry.get("questions", []),
        "transcript": [_strip_unserializable_turn(turn) for turn in entry.get("transcript", [])],
        "assessments": entry.get("assessments", []),
        "feedback": entry.get("feedback", {}),
        "saved_at": entry.get("saved_at"),
    }


def persist_battle_history(history: List[Dict]) -> None:
    try:
        BATTLE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = [_serialize_history_entry(entry) for entry in history]
        BATTLE_HISTORY_FILE.write_text(json.dumps(serializable, indent=2))
    except Exception as exc:
        st.warning(f"Unable to save battle history: {exc}")


def render_chat_history(history_entry: Dict) -> None:
    if not history_entry:
        return

    turns = history_entry.get("transcript") or []
    if not turns:
        st.caption("No exchanges recorded for this battle yet.")
        return

    for idx, turn in enumerate(turns, start=1):
        question = turn.get("question", "")
        answer = turn.get("answer", "")
        with st.chat_message("assistant"):
            st.markdown(question or f"_Question {idx} missing._")
        with st.chat_message("user"):
            st.markdown(answer or "_No answer captured._")


def _format_history_option_label(index: int, entry: Dict) -> str:
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

if "battle_topic" not in st.session_state:
    st.session_state.battle_topic = ""

if "battle_history" not in st.session_state:
    st.session_state.battle_history: List[Dict[str, str]] = []

if not st.session_state.battle_history:
    persisted_history = _load_persisted_battle_history()
    if persisted_history:
        st.session_state.battle_history = persisted_history

if "battle_context" not in st.session_state:
    st.session_state.battle_context = ""

if "battle_questions" not in st.session_state:
    st.session_state.battle_questions: List[Dict[str, str]] = []

if "battle_current_idx" not in st.session_state:
    st.session_state.battle_current_idx = 0

if "battle_transcript" not in st.session_state:
    st.session_state.battle_transcript: List[Dict[str, str]] = []

if "battle_feedback" not in st.session_state:
    st.session_state.battle_feedback: Optional[Dict[str, str]] = None

if "battle_question_audio" not in st.session_state:
    st.session_state.battle_question_audio: Dict[int, Optional[bytes]] = {}

if "battle_pending_audio" not in st.session_state:
    st.session_state.battle_pending_audio: Dict[int, bytes] = {}

if "battle_live_transcript" not in st.session_state:
    st.session_state.battle_live_transcript: Dict[int, str] = {}

if "battle_assessments" not in st.session_state:
    st.session_state.battle_assessments: List[Dict[str, str]] = []

if "battle_feedback_audio" not in st.session_state:
    st.session_state.battle_feedback_audio: Dict[str, Optional[bytes]] = {}

if "battle_waiting_next" not in st.session_state:
    st.session_state.battle_waiting_next = False

if "battle_waiting_question_id" not in st.session_state:
    st.session_state.battle_waiting_question_id = None

if "battle_disabled_questions" not in st.session_state:
    st.session_state.battle_disabled_questions: Dict[int, bool] = {}

if "selected_battle_idx" not in st.session_state:
    st.session_state.selected_battle_idx = 0

if "selected_battle_label" not in st.session_state:
    st.session_state.selected_battle_label = ""

if "battle_active" not in st.session_state:
    st.session_state.battle_active = False

# --- 3. RAG/RETRIEVAL HELPERS ---
@st.cache_resource
def get_vectorstore() -> Chroma:
    embeddings = SentenceTransformerEmbeddings(
        model_name=SENTENCE_TRANSFORMERS_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
    )


def _clean_text(text: str) -> str:
    text = re.sub(r"[^\x20-\x7E\s]+", "", text or "").strip()
    return text.replace('"', "'")


def _format_source(metadata: Optional[Dict]) -> str:
    metadata = metadata or {}
    return (
        metadata.get("source")
        or metadata.get("path")
        or metadata.get("file_path")
        or metadata.get("doc_id")
        or "Knowledge Base"
    )


def _dedupe_by_text(docs: List[Document]) -> List[Document]:
    seen, deduped = set(), []
    for doc in docs:
        content = getattr(doc, "page_content", "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        deduped.append(doc)
    return deduped


def retrieve_context(query: str, n_results: int = 3) -> str:
    try:
        vectorstore = get_vectorstore()
        k = max(1, n_results or 1)
        base_kwargs = dict(SEARCH_KWARGS)
        base_kwargs_k = base_kwargs.get("k")
        if base_kwargs_k is None or base_kwargs_k < k:
            base_kwargs["k"] = k

        documents: List[Document] = []
        try:
            documents = vectorstore.similarity_search(query, k=base_kwargs.get("k", k))
        except Exception:
            retriever = vectorstore.as_retriever(search_kwargs=base_kwargs)
            documents = retriever.get_relevant_documents(query)

        if not documents:
            return "No relevant context found in the knowledge base."

        documents = _dedupe_by_text(documents)[:k]

        context_blocks: List[str] = []
        for doc in documents:
            raw_text = getattr(doc, "page_content", "")
            cleaned_text = _clean_text(raw_text)
            if not cleaned_text or not any(ch.isalpha() for ch in cleaned_text):
                continue
            source = _format_source(getattr(doc, "metadata", None))
            context_blocks.append(f"Source: {source}\n{cleaned_text}")

        if not context_blocks:
            return "No relevant context found in the knowledge base."

        context = "\n---\n".join(context_blocks)
        return f"Retrieved Context:\n{context}"
    except Exception as exc:  # pragma: no cover - retrieval resilience
        return f"RAG Retrieval Error: Could not load context. {exc}"

# --- 4A. AUDIO HELPERS ---
def render_audio_clip(audio_bytes: Optional[bytes], mime: str = QUESTION_AUDIO_MIME, autoplay: bool = False):
    if not audio_bytes:
        return
    if autoplay:
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        autoplay_tag = f"""
            <audio src="data:{mime};base64,{b64_audio}" autoplay controls style="width: 100%; margin-top: 0.5rem;">
                Your browser does not support the audio element.
            </audio>
        """
        st.markdown(autoplay_tag, unsafe_allow_html=True)
    else:
        st.audio(audio_bytes, format=mime)


def prepare_question_content(raw_text: str) -> tuple[str, str]:
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


# --- 4. LLM PROMPT HELPERS ---
def generate_llm_response(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Amit, the Rival-Ally interviewer. "
                "Guide Keerti through scenario-based battles by asking probing, real-world questions, "
                "evaluating her spoken responses, and coaching improvement. "
                "Always answer strictly as a JSON object."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.4,
            response_format={"type": "json_object"},
            max_tokens=900,
        )
        return chat_completion.choices[0].message.content
    except Exception as exc:
        st.error(f"Groq API Error: {exc}")
        return "ERROR_API_FAILED"


def clean_and_parse_json(text: str) -> Dict:
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```", "", text).strip()
    return json.loads(text)


def get_question_audio(question_id: int, prompt_text: str) -> Optional[bytes]:
    cached = st.session_state.battle_question_audio.get(question_id)
    if cached:
        return cached

    verbal_prompt = prompt_text
    audio_bytes = synthesize_speech(verbal_prompt)
    if audio_bytes:
        st.session_state.battle_question_audio[question_id] = audio_bytes
    return audio_bytes


def build_initial_question_prompt(concept: str, context: str) -> str:
    return f"""
You are Amit, a senior technical interviewer hiring for a high-impact role. You are preparing to assess the candidate Keerti on the concept below.

Concept focus: {concept.strip()}

Reference material (may be noisy or empty):
{context}

Craft the very first interview question Amit will ask Keerti. Stay grounded in the reference material—cite specific situations or facts from it when framing the prompt.

Guidelines:
1. Use a scenario anchored in the reference material, not a generic definition question.
2. Aim to surface both conceptual understanding and practical judgment.
3. Keep the scenario vivid, short, and practical—treat it like an on-the-spot hiring challenge.
4. Start with an accessible difficulty. Set `"difficulty": "foundation"` and design the scenario so a prepared candidate can ease in.
5. Use clear language but slip in a gentle twist or constraint that reveals whether Keerti is truly thinking.
6. Avoid quoting long noisy strings verbatim. If you must reference raw text, paraphrase it and replace any double quotes with single quotes.

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
{context}

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

Respond ONLY with valid JSON:
{{
  "question": "Next scenario-driven question tied to the learner's last response and the reference material.",
  "focus_area": "Skill or competency you are targeting now.",
  "difficulty": "foundation|intermediate|advanced",
  "interviewer_goal": "What you expect to learn from the answer.",
  "evidence_anchor": "Short note on which parts of the reference material or learner response prompted this follow-up."
}}
"""


def get_feedback_audio(label: str, text: str) -> Optional[bytes]:
    cache = st.session_state.battle_feedback_audio
    cache_key = f"{label}:{text}"
    if cache_key not in cache:
        spoken = f"Amit reflects: {text}"
        audio_bytes = synthesize_speech(spoken, language="en")
        cache[cache_key] = audio_bytes
    return cache.get(cache_key)


def build_answer_assessment_prompt(
    concept: str,
    question: Dict[str, str],
    learner_transcript: str,
    context: str,
) -> str:
    return f"""
You are Amit benchmarking Keerti's spoken answer during a live technical interview.

Concept focus: {concept.strip()}

Reference material (may be noisy or empty):
{context}

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
{context}

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


# --- 5. PAGE HEADER & INTRO ---
st.markdown("## Critical Thinking Battle")
st.write(
    "Amit, the interviewer, probes Keerti through rapid-fire scenarios. Everything happens in voice: "
    "Amit asks, Keerti responds, and the debrief keeps pace with their conversation."
)

# --- 6. BATTLE FORM ---
if not st.session_state.battle_active:
    with st.form("battle_arena", clear_on_submit=False):
        concept = st.text_input(
            "Concept you want to defend",
            value=st.session_state.battle_topic,
            placeholder="e.g. Backpropagation in neural networks",
        )
        submit_request = st.form_submit_button("Start Challenge", type="primary")

    if submit_request:
        if not concept.strip():
            st.warning("Please enter the concept you want to defend.")
        else:
            st.session_state.battle_topic = concept.strip()

            with st.spinner("Amit is designing the battle plan..."):
                context = retrieve_context(concept.strip())
                st.session_state.battle_context = context
                st.session_state.battle_question_audio = {}
                st.session_state.battle_pending_audio = {}
                st.session_state.battle_live_transcript = {}
                st.session_state.battle_assessments = []
                st.session_state.battle_waiting_next = False
                st.session_state.battle_waiting_question_id = None
                st.session_state.battle_disabled_questions = {}

                prompt = build_initial_question_prompt(concept, context)
                json_response_text = generate_llm_response(prompt)

                if json_response_text == "ERROR_API_FAILED":
                    st.stop()

                try:
                    question_payload = clean_and_parse_json(json_response_text)
                    first_question = question_payload.get("question", "").strip()

                    if not first_question:
                        st.error("The interviewer could not generate a starting question. Please try again.")
                    else:
                        display_question, audio_prompt = prepare_question_content(first_question)
                        st.session_state.battle_questions = [
                            {
                                "id": 1,
                                "question": display_question,
                                "focus_area": question_payload.get("focus_area", ""),
                                "difficulty": question_payload.get("difficulty", ""),
                                "interviewer_goal": question_payload.get("interviewer_goal", ""),
                                "evidence_anchor": question_payload.get("evidence_anchor", ""),
                                "audio_prompt": audio_prompt,
                            }
                        ]
                        get_question_audio(1, audio_prompt)
                        st.session_state.battle_current_idx = 0
                        st.session_state.battle_transcript = []
                        st.session_state.battle_feedback = None
                        st.session_state.battle_active = True
                        st.rerun()
                except json.JSONDecodeError:
                    st.error("Failed to parse the interviewer plan. Please try again.")

if st.session_state.battle_active:
    if st.button("End Challenge", type="secondary"):
        st.session_state.battle_active = False
        st.session_state.battle_current_idx = len(st.session_state.battle_questions)
        answered_count = len(st.session_state.battle_transcript)
        answered_questions = st.session_state.battle_questions[:answered_count]
        should_reset = True
        if st.session_state.battle_transcript:
            with st.spinner("Amit is compiling your battle report..."):
                evaluation_prompt = build_interview_evaluation_prompt(
                    st.session_state.battle_topic,
                    answered_questions,
                    st.session_state.battle_transcript,
                    st.session_state.battle_context,
                    st.session_state.battle_assessments,
                )
                eval_response_text = generate_llm_response(evaluation_prompt)

                if eval_response_text == "ERROR_API_FAILED":
                    st.warning("Amit couldn't produce the final report. Please try again later.")
                    should_reset = False
                else:
                    try:
                        battle_feedback = clean_and_parse_json(eval_response_text)
                        st.session_state.battle_feedback = battle_feedback
                        record = {
                            "concept": st.session_state.battle_topic,
                            "transcript": list(st.session_state.battle_transcript),
                            "questions": list(answered_questions),
                            "context": st.session_state.battle_context,
                            "assessments": list(st.session_state.battle_assessments),
                            "feedback": battle_feedback,
                            "saved_at": datetime.now(timezone.utc).isoformat(),
                        }
                        st.session_state.battle_history.insert(0, record)
                        st.session_state.selected_battle_idx = 0
                        persist_battle_history(st.session_state.battle_history)
                        st.success("Challenge ended. Review the transcript and feedback below.")
                    except json.JSONDecodeError:
                        st.warning("Amit's debrief response was invalid. Please try ending again.")
                        should_reset = False
        else:
            st.info("Challenge ended without any responses recorded.")

        if should_reset:
            st.session_state.battle_questions = []
            st.session_state.battle_transcript = []
            st.session_state.battle_assessments = []
            st.session_state.battle_pending_audio = {}
            st.session_state.battle_live_transcript = {}
            st.session_state.battle_question_audio = {}
            st.session_state.battle_feedback = None
            st.session_state.battle_waiting_next = False
            st.session_state.battle_waiting_question_id = None
            st.session_state.battle_disabled_questions = {}
            st.session_state.battle_topic = ""
            st.rerun()


# --- 7. INTERVIEW FLOW ---
if st.session_state.battle_questions:
    total_questions = len(st.session_state.battle_questions)
    current_idx = st.session_state.battle_current_idx

    if current_idx < total_questions:
        st.markdown("---")
        st.subheader("🎤 Battle In Progress")
        st.caption(f"Concept under fire: {st.session_state.battle_topic}")

        # Replay completed exchange transcripts
        for past_turn in st.session_state.battle_transcript:
            past_question_audio = get_question_audio(
                past_turn.get("id", 0), past_turn.get("audio_prompt", past_turn.get("question", ""))
            )
            with st.chat_message("assistant"):
                render_audio_clip(past_question_audio, autoplay=False)
                st.markdown(past_turn.get("question", ""))
            with st.chat_message("user"):
                if past_turn.get("answer_audio"):
                    render_audio_clip(past_turn["answer_audio"], mime="audio/wav", autoplay=False)
                st.caption(past_turn.get("answer", ""))

        current_question = st.session_state.battle_questions[current_idx]
        question_text = current_question.get("question", "")
        audio_prompt = current_question.get("audio_prompt")
        if not audio_prompt:
            _, audio_prompt = prepare_question_content(question_text)
            current_question["audio_prompt"] = audio_prompt
        focus_area = current_question.get("focus_area", "")
        difficulty = current_question.get("difficulty", "")
        interviewer_goal = current_question.get("interviewer_goal", "")
        evidence_anchor = current_question.get("evidence_anchor", "")
        question_id = current_question.get("id", current_idx + 1)

        question_audio_bytes = get_question_audio(question_id, audio_prompt)

        with st.chat_message("assistant"):
            render_audio_clip(question_audio_bytes)
            transcript_bits = []
            if evidence_anchor:
                transcript_bits.append(f"(Context: {evidence_anchor})")
            if interviewer_goal:
                transcript_bits.append(f"(Goal: {interviewer_goal})")
            transcript_header = " ".join(transcript_bits)
            st.markdown(question_text)
            if transcript_header:
                st.caption(transcript_header)

        record_key = f"battle_voice_{current_idx}"
        waiting_state = st.session_state.battle_waiting_next
        waiting_question_id = st.session_state.battle_waiting_question_id

        if waiting_state and waiting_question_id is not None and waiting_question_id != question_id:
            st.session_state.battle_waiting_next = False
            st.session_state.battle_waiting_question_id = None
            waiting_state = False
            waiting_question_id = None

        waiting_for_followup = waiting_state and waiting_question_id == question_id

        pending_audio = st.session_state.battle_pending_audio.get(current_idx)
        live_preview = st.session_state.battle_live_transcript.get(current_idx, "")
        disable_map = st.session_state.battle_disabled_questions
        form_disabled = disable_map.get(question_id, False)

        if waiting_for_followup or form_disabled:
            st.info("Amit is reviewing your response. Hang tight while the next scenario loads...")
            submit_answer = False
            audio_capture = None
        else:
            if form_disabled:
                st.warning("You have already answered this question. Please wait for the next scenario to load.")
                submit_answer = False
                audio_capture = None
            else:
                with st.form(f"battle_response_{current_idx}", clear_on_submit=True):
                    audio_capture = st.audio_input(
                        "Keerti, record your answer",
                        key=record_key,
                        help="Press the microphone, speak with structure, and press stop when finished.",
                    )
                    submit_answer = st.form_submit_button("Submit answer", type="primary")

        if not waiting_for_followup and not form_disabled and audio_capture is not None:
            captured_bytes = audio_capture.getvalue()
            if captured_bytes:
                st.session_state.battle_pending_audio[current_idx] = captured_bytes
                preview_text = transcribe_audio(captured_bytes)
                if preview_text.strip():
                    st.session_state.battle_live_transcript[current_idx] = preview_text.strip()

        live_preview = st.session_state.battle_live_transcript.get(current_idx, "")
        if live_preview:
            st.caption(f"Keerti (live transcript): {live_preview}")
        else:
            st.caption("Keerti, speak clearly about your reasoning. A transcript preview will appear once detected.")

        pending_audio = st.session_state.battle_pending_audio.get(current_idx)
        if pending_audio:
            render_audio_clip(pending_audio, mime="audio/wav", autoplay=False)

        if submit_answer and not waiting_for_followup:
            answer_audio = st.session_state.battle_pending_audio.get(current_idx)
            transcript_text = st.session_state.battle_live_transcript.get(current_idx, "")

            if not answer_audio:
                st.warning("Please record an answer before continuing.")
            else:
                if not transcript_text.strip():
                    transcript_text = transcribe_audio(answer_audio)

                if not transcript_text.strip():
                    st.warning(
                        "We captured the audio but couldn't understand it clearly. "
                        "Please try again with a clearer recording."
                    )
                else:
                    st.success(f"Transcribed summary: {transcript_text.strip()}")
                    st.session_state.battle_transcript.append(
                        {
                            "id": current_question.get("id", current_idx + 1),
                            "question": question_text,
                            "audio_prompt": audio_prompt,
                            "answer": transcript_text.strip(),
                            "focus_area": focus_area,
                            "difficulty": difficulty,
                            "interviewer_goal": interviewer_goal,
                            "evidence_anchor": current_question.get("evidence_anchor", ""),
                            "answer_audio": answer_audio,
                            "answer_origin": "audio",
                        }
                    )

                    st.session_state.battle_disabled_questions[question_id] = True
                    st.session_state.battle_waiting_next = True
                    st.session_state.battle_waiting_question_id = current_question.get("id", current_idx + 1)

                    assessment_prompt = build_answer_assessment_prompt(
                        st.session_state.battle_topic,
                        current_question,
                        transcript_text,
                        st.session_state.battle_context,
                    )
                    assessment_raw = generate_llm_response(assessment_prompt)
                    latest_assessment: Dict[str, str] = {}
                    if assessment_raw != "ERROR_API_FAILED":
                        try:
                            latest_assessment = clean_and_parse_json(assessment_raw)
                            latest_assessment["question_id"] = current_question.get("id", current_idx + 1)
                            st.session_state.battle_assessments.append(latest_assessment)
                            st.caption(
                                f"Assessment snapshot -> understanding: {latest_assessment.get('understanding_level', 'n/a')}, "
                                f"decision: {latest_assessment.get('decision_quality', 'n/a')}, "
                                f"next focus: {latest_assessment.get('next_focus', 'n/a')}."
                            )
                        except json.JSONDecodeError:
                            st.warning("Could not parse the interviewer's quick assessment. Continuing anyway.")
                    else:
                        st.warning("Assessment service unavailable right now. Continuing without adaptive feedback.")

                    st.session_state.battle_current_idx += 1
                    st.session_state.battle_feedback = None
                    st.session_state.battle_pending_audio.pop(current_idx, None)
                    st.session_state.battle_live_transcript.pop(current_idx, None)
                    st.session_state.pop(record_key, None)

                    answers_count = len(st.session_state.battle_transcript)

                    if len(st.session_state.battle_questions) <= answers_count:
                        with st.spinner("Amit is lining up the next scenario..."):
                            followup_prompt = build_followup_question_prompt(
                                st.session_state.battle_topic,
                                st.session_state.battle_context,
                                st.session_state.battle_transcript,
                                st.session_state.battle_assessments[-1]
                                if st.session_state.battle_assessments
                                else None,
                            )
                            followup_response_text = generate_llm_response(followup_prompt)

                            if followup_response_text == "ERROR_API_FAILED":
                                st.warning("Could not fetch the next challenge. You can end the battle when ready.")
                            else:
                                try:
                                    followup_payload = clean_and_parse_json(followup_response_text)
                                    followup_question = followup_payload.get("question", "").strip()
                                    if not followup_question:
                                        st.warning("Amit couldn't craft the next challenge. You may choose to end the battle.")
                                    else:
                                        display_followup, followup_audio = prepare_question_content(followup_question)
                                        st.session_state.battle_questions.append(
                                            {
                                                "id": len(st.session_state.battle_questions) + 1,
                                                "question": display_followup,
                                                "focus_area": followup_payload.get("focus_area", ""),
                                                "difficulty": followup_payload.get("difficulty", ""),
                                                "interviewer_goal": followup_payload.get("interviewer_goal", ""),
                                                "evidence_anchor": followup_payload.get("evidence_anchor", ""),
                                                "audio_prompt": followup_audio,
                                            }
                                        )
                                        get_question_audio(
                                            st.session_state.battle_questions[-1]["id"],
                                            followup_audio,
                                        )
                                        st.info("Next scenario loaded.")
                                except json.JSONDecodeError:
                                    st.warning("Amit's follow-up was garbled. You can request a new challenge by ending and restarting.")

                    st.rerun()


# --- 8. DISPLAY RESULTS ---
if not st.session_state.battle_active and st.session_state.battle_history:
    st.markdown("---")
    st.subheader("📊 Battle Report")

    history_indices = list(range(len(st.session_state.battle_history)))
    selected_idx = st.selectbox(
        "Review a completed battle",
        history_indices,
        index=min(st.session_state.selected_battle_idx, len(history_indices) - 1),
        format_func=lambda idx: _format_history_option_label(idx, st.session_state.battle_history[idx]),
        key="battle_report_selector",
    )
    st.session_state.selected_battle_idx = selected_idx
    selected_label = _format_history_option_label(selected_idx, st.session_state.battle_history[selected_idx])
    st.session_state.selected_battle_label = selected_label

    exchange = st.session_state.battle_history[selected_idx]
    feedback = exchange.get("feedback", {})
    score = feedback.get("score", "needs-work").lower()

    if score == "excellent":
        st.success(f"Round {selected_idx + 1}: {feedback.get('verdict', 'Excellent work!')}")
    elif score == "good":
        st.info(f"Round {selected_idx + 1}: {feedback.get('verdict', 'Strong showing with room to polish.')}")
    elif score == "fair":
        st.warning(f"Round {selected_idx + 1}: {feedback.get('verdict', 'Some gaps to address.')}")
    else:
        st.error(f"Round {selected_idx + 1}: {feedback.get('verdict', 'Needs more work before the next battle.')}")

    verdict_audio = get_feedback_audio(f"verdict_{selected_idx}", feedback.get("verdict", "").strip())
    if verdict_audio:
        render_audio_clip(verdict_audio, autoplay=False)

    with st.expander(f"See detailed critique for '{exchange.get('concept', 'Battle')}'", expanded=True):
        assessment_map = {
            item.get("question_id"): item for item in exchange.get("assessments", [])
        }

        st.markdown("**Interview transcript**")
        for turn in exchange.get("transcript", []):
            st.markdown("**Q:**")
            st.markdown(turn.get("question", "Unknown question"))
            meta_bits = []
            if turn.get("focus_area"):
                meta_bits.append(f"Focus: {turn['focus_area']}")
            if turn.get("difficulty"):
                meta_bits.append(f"Level: {turn['difficulty']}")
            if turn.get("evidence_anchor"):
                meta_bits.append(f"Context hook: {turn['evidence_anchor']}")
            if meta_bits:
                st.caption(" · ".join(meta_bits))
            question_audio = get_question_audio(
                turn.get("id", 0), turn.get("audio_prompt", turn.get("question", ""))
            )
            if question_audio:
                render_audio_clip(question_audio, autoplay=False)
            st.markdown("**Your answer**")
            st.markdown(turn.get("answer", "_No answer recorded._"))
            if turn.get("answer_audio"):
                render_audio_clip(turn["answer_audio"], mime="audio/wav", autoplay=False)
            if turn.get("answer_origin") == "audio":
                st.caption("Transcribed automatically from your recording.")
            assessment = assessment_map.get(turn.get("id"))
            if assessment:
                st.caption(
                    f"Interviewer read: understanding={assessment.get('understanding_level', 'n/a')}, "
                    f"decision={assessment.get('decision_quality', 'n/a')} "
                    f"-> next focus: {assessment.get('next_focus', 'n/a')} "
                    f"(difficulty cue: {assessment.get('difficulty_adjustment', 'n/a')})."
                )

        question_feedback = feedback.get("question_feedback") or []
        if question_feedback:
            st.markdown("**Question-by-question coaching**")
            for qf in question_feedback:
                st.markdown(f"**Q{qf.get('id', '?')} recap**")
                st.markdown(qf.get("assessment", ""))
                improvements = qf.get("improvements") or []
                if improvements:
                    st.markdown("Improvements to practice:")
                    for item in improvements:
                        st.markdown(f"- {item}")
                follow_up = qf.get("follow_up", "").strip()
                if follow_up:
                    st.caption(f"Follow-up drill: {follow_up}")

        strengths = feedback.get("strengths") or []
        improvements = feedback.get("improvement_opportunities") or []

        if strengths:
            st.markdown("**What landed well**")
            for item in strengths:
                st.markdown(f"- {item}")

        if improvements:
            st.markdown("**What to sharpen**")
            for item in improvements:
                st.markdown(f"- {item}")

        improved_expl = feedback.get("improved_explanation", "").strip()
        if improved_expl:
            st.markdown("**Mentor's upgraded explanation**")
            st.info(improved_expl)
            improved_audio = get_feedback_audio(f"improved_{selected_idx}", improved_expl)
            if improved_audio:
                render_audio_clip(improved_audio, autoplay=False)

        practice_prompt = feedback.get("practice_prompt", "").strip()
        if practice_prompt:
            st.markdown("**Next rep**")
            st.markdown(f"> {practice_prompt}")

        context = exchange.get("context", "")
        if context and not context.startswith("No relevant context"):
            with st.expander("Reference context used (from knowledge base)"):
                st.markdown(context)

