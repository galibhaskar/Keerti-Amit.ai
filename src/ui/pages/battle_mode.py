"""Battle Mode page - Voice-based interview challenges."""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import streamlit as st

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
from services.retrieval import retrieve_context_langchain
from services.audio import synthesize_speech, transcribe_audio
from services.config_manager import get_user_settings
from ui.utils import render_audio_clip
from utils.json_parser import clean_and_parse_json

# Initialize session state
if "battle_topic" not in st.session_state:
    st.session_state.battle_topic = ""

if "battle_history" not in st.session_state:
    st.session_state.battle_history: List[Dict[str, str]] = []

if not st.session_state.battle_history:
    persisted_history = load_battle_history()
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


def get_question_audio(question_id: int, prompt_text: str) -> Optional[bytes]:
    """Get or generate question audio."""
    cached = st.session_state.battle_question_audio.get(question_id)
    if cached:
        return cached

    audio_bytes = synthesize_speech(prompt_text)
    if audio_bytes:
        st.session_state.battle_question_audio[question_id] = audio_bytes
    return audio_bytes


def get_feedback_audio(label: str, text: str) -> Optional[bytes]:
    """Get or generate feedback audio."""
    cache = st.session_state.battle_feedback_audio
    cache_key = f"{label}:{text}"
    if cache_key not in cache:
        spoken = f"Amit reflects: {text}"
        audio_bytes = synthesize_speech(spoken, language="en")
        cache[cache_key] = audio_bytes
    return cache.get(cache_key)


# Page header
st.markdown("## Critical Thinking Battle")
st.write(
    "Amit, the interviewer, probes Keerti through rapid-fire scenarios. Everything happens in voice: "
    "Amit asks, Keerti responds, and the debrief keeps pace with their conversation."
)

# Battle form
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
                context = retrieve_context_langchain(concept.strip())
                st.session_state.battle_context = context
                st.session_state.battle_question_audio = {}
                st.session_state.battle_pending_audio = {}
                st.session_state.battle_live_transcript = {}
                st.session_state.battle_assessments = []
                st.session_state.battle_waiting_next = False
                st.session_state.battle_waiting_question_id = None
                st.session_state.battle_disabled_questions = {}

                prompt = build_initial_question_prompt(concept, context)
                settings = get_user_settings()
                model = settings.get("model", "llama-3.1-8b-instant")
                json_response_text = generate_battle_llm_response(prompt, model=model)

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

# End battle
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
                settings = get_user_settings()
                model = settings.get("model", "llama-3.1-8b-instant")
                eval_response_text = generate_battle_llm_response(evaluation_prompt, model=model)

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
                        save_battle_history(st.session_state.battle_history)
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

# Interview flow
if st.session_state.battle_questions:
    total_questions = len(st.session_state.battle_questions)
    current_idx = st.session_state.battle_current_idx

    if current_idx < total_questions:
        st.markdown("---")
        st.subheader("🎤 Battle In Progress")
        st.caption(f"Concept under fire: {st.session_state.battle_topic}")

        # Replay completed exchanges
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
                    settings = get_user_settings()
                    model = settings.get("model", "llama-3.1-8b-instant")
                    assessment_raw = generate_battle_llm_response(assessment_prompt, model=model)
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
                            followup_response_text = generate_battle_llm_response(followup_prompt, model=model)

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

# Display results
if not st.session_state.battle_active and st.session_state.battle_history:
    st.markdown("---")
    st.subheader("📊 Battle Report")

    history_indices = list(range(len(st.session_state.battle_history)))
    selected_idx = st.selectbox(
        "Review a completed battle",
        history_indices,
        index=min(st.session_state.selected_battle_idx, len(history_indices) - 1),
        format_func=lambda idx: format_history_option_label(idx, st.session_state.battle_history[idx]),
        key="battle_report_selector",
    )
    st.session_state.selected_battle_idx = selected_idx
    selected_label = format_history_option_label(selected_idx, st.session_state.battle_history[selected_idx])
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
