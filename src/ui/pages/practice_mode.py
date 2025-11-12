"""Practice Mode page - Interactive flashcards and quizzes."""

import streamlit as st
from services.practice import (
    generate_flashcard,
    save_flashcard_to_json,
    load_flashcard_history,
    format_history_option_label,
)
from services.config_manager import get_user_settings
from utils.json_parser import clean_and_parse_json

# Initialize session state
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "concept"

if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None

if "current_concept_title" not in st.session_state:
    st.session_state.current_concept_title = None

if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

if "previous_quiz_titles" not in st.session_state:
    st.session_state.previous_quiz_titles = []

if "user_id" not in st.session_state:
    st.session_state.user_id = "local_user"

if "practice_session_id" not in st.session_state:
    st.session_state.practice_session_id = None

if "practice_history" not in st.session_state:
    st.session_state.practice_history = load_flashcard_history()

if "selected_practice_idx" not in st.session_state:
    st.session_state.selected_practice_idx = 0

if "selected_practice_label" not in st.session_state:
    st.session_state.selected_practice_label = ""


def trigger_flashcard_generation():
    """Trigger a new flashcard generation."""
    if st.session_state.current_quiz and st.session_state.difficulty in ["easy", "example"]:
        st.session_state.current_concept_title = st.session_state.current_quiz.get('flashcard_concept')
    st.session_state.current_quiz = None
    st.rerun()


# Page header
st.markdown("## Practice Mode")
st.write("Practice your knowledge with Quizzes and Flashcards :)")

# Starting form
if not st.session_state.quiz_started:
    with st.form(key='topic_form'):
        topic_input = st.text_input("Enter a Concept/Topic to Study:", key="new_topic_input")
        submit_button = st.form_submit_button(label='Start Flashcards')

        if submit_button and topic_input:
            st.session_state.last_topic = topic_input
            st.session_state.quiz_started = True
            st.session_state.current_quiz = None
            st.session_state.difficulty = "concept"
            st.session_state.previous_quiz_titles = []
            st.session_state.current_concept_title = None
            st.session_state.practice_session_id = None
            st.rerun()

# Main flashcard generation/display flow
if st.session_state.quiz_started:
    topic = st.session_state.last_topic

    # Generate/regenerate flashcard
    if st.session_state.current_quiz is None:
        with st.spinner(f"Generating {st.session_state.difficulty} flashcard on '{topic}'..."):
            try:
                settings = get_user_settings()
                model = settings.get("model", "llama-3.1-8b-instant")
                
                response_text = generate_flashcard(
                    topic=topic,
                    mode=st.session_state.difficulty,
                    previous_titles=st.session_state.previous_quiz_titles,
                    current_concept_title=st.session_state.current_concept_title,
                    model=model,
                )

                quiz_data = clean_and_parse_json(response_text)
                
                # Check if we got the "no context" response
                if quiz_data.get("flashcard_concept") == "No context available" or \
                   quiz_data.get("quiz_text") == "No context available":
                    st.warning("⚠️ **No context found for this topic.**")
                    st.info(
                        "💡 **To generate flashcards, you need to:**\n\n"
                        "1. Go to the **Data Ingestion** page\n"
                        "2. Upload documents related to your topic\n"
                        "3. Wait for the documents to be processed (check the queue status)\n"
                        "4. Try generating flashcards again"
                    )
                    st.session_state.quiz_started = False
                    st.stop()
                
                st.session_state.current_quiz = quiz_data

                # Save to session (will create new session if session_id is None)
                session_id = save_flashcard_to_json(
                    topic=topic,
                    mode=st.session_state.difficulty,
                    data=quiz_data,
                    session_id=st.session_state.practice_session_id,
                )
                
                # Store session ID for subsequent flashcards in this session
                st.session_state.practice_session_id = session_id
                
                # Reload history after saving
                st.session_state.practice_history = load_flashcard_history()

                # Store concept title for subsequent generations
                if st.session_state.difficulty == "concept":
                    concept_title = quiz_data.get('flashcard_concept', 'A general concept')
                    st.session_state.previous_quiz_titles.append(concept_title)
                    st.session_state.current_concept_title = concept_title

                st.rerun()

            except ValueError as e:
                st.error(f"API failed. Cannot generate flashcard: {e}")
                st.session_state.quiz_started = False
            except Exception as e:
                st.error(f"Failed to generate flashcard: {e}")
                st.session_state.current_quiz = None
                st.session_state.quiz_started = False

    # Display flashcard interface
    if st.session_state.current_quiz:
        quiz_data = st.session_state.current_quiz

        st.markdown("---")
        st.subheader(f"📖 Flashcard on: *{topic}*")

        # Render card content
        st.success(f"**Focus:** {quiz_data.get('quiz_text', 'N/A')}")
        st.markdown(f"**Concept Title:** *{quiz_data['flashcard_concept']}*")
        st.info(quiz_data['flashcard_rationale'])

        st.markdown("---")
        st.subheader("What do you want to do next?")

        # Navigation buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("⏪ Let's Try Again (Easier)", help="Generate a new flashcard for the SAME concept with a simpler explanation."):
                st.session_state.difficulty = "easy"
                trigger_flashcard_generation()

        with col2:
            if st.button("💻 Example Please", help="Generate a new flashcard for the SAME concept focusing on a code or real-world example."):
                st.session_state.difficulty = "example"
                trigger_flashcard_generation()

        with col3:
            if st.button("➡️ Proceed to Next One", help="Generate a flashcard for a NEW concept under the same topic."):
                st.session_state.difficulty = "concept"
                st.session_state.current_concept_title = None
                trigger_flashcard_generation()

        with col4:
            if st.button("🔄 Start Over", help="Go back to the main page to enter a new topic."):
                st.session_state.quiz_started = False
                st.session_state.current_quiz = None
                st.session_state.last_topic = None
                st.session_state.previous_quiz_titles = []
                st.session_state.current_concept_title = None
                # End current session
                st.session_state.practice_session_id = None
                # Reload history when starting over
                st.session_state.practice_history = load_flashcard_history()
                st.rerun()

# Display history
if not st.session_state.quiz_started and st.session_state.practice_history:
    st.markdown("---")
    st.subheader("📚 Practice History")
    st.write("Review your previously generated practice sessions below. Each session contains all flashcards from one practice session.")

    # Reverse history to show most recent first (like battle mode)
    reversed_history = list(reversed(st.session_state.practice_history))
    history_indices = list(range(len(reversed_history)))
    
    # Map display indices to original indices (reversed)
    def get_original_idx(display_idx: int) -> int:
        return len(st.session_state.practice_history) - 1 - display_idx
    
    selected_display_idx = st.selectbox(
        "Select a practice session to review",
        history_indices,
        index=min(st.session_state.selected_practice_idx, len(history_indices) - 1) if history_indices else 0,
        format_func=lambda display_idx: format_history_option_label(
            get_original_idx(display_idx), 
            reversed_history[display_idx]
        ),
        key="practice_history_selector",
    )
    st.session_state.selected_practice_idx = selected_display_idx
    
    if selected_display_idx is not None and selected_display_idx < len(reversed_history):
        selected_session = reversed_history[selected_display_idx]
        original_idx = get_original_idx(selected_display_idx)
        selected_label = format_history_option_label(original_idx, selected_session)
        st.session_state.selected_practice_label = selected_label

        session = selected_session
        topic = session.get("topic", "Unknown Topic")
        flashcards = session.get("flashcards", [])
        started_at = session.get("started_at", "")
        updated_at = session.get("updated_at", "")
        user_id = session.get("user_id", "")

        st.markdown(f"### 📖 Practice Session: {topic}")
        
        # Display session metadata
        col1, col2 = st.columns(2)
        with col1:
            if started_at:
                try:
                    from datetime import datetime, timezone
                    if isinstance(started_at, str):
                        stamp = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    else:
                        stamp = started_at
                    display_time = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    st.caption(f"**Started:** {display_time}")
                except (ValueError, AttributeError):
                    st.caption(f"**Started:** {started_at}")
        
        with col2:
            if updated_at:
                try:
                    from datetime import datetime, timezone
                    if isinstance(updated_at, str):
                        stamp = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    else:
                        stamp = updated_at
                    display_time = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    st.caption(f"**Last Updated:** {display_time}")
                except (ValueError, AttributeError):
                    st.caption(f"**Last Updated:** {updated_at}")

        st.info(f"**Total Flashcards:** {len(flashcards)}")

        # Display all flashcards in the session
        if flashcards:
            st.markdown("#### Flashcards in this session:")
            for idx, flashcard in enumerate(flashcards, start=1):
                mode = flashcard.get("mode", "concept")
                quiz_data = flashcard.get("quiz_data", {})
                timestamp = flashcard.get("timestamp", "")
                
                # Format mode display
                mode_display = {
                    "concept": "Concept",
                    "easy": "Easy",
                    "example": "Example"
                }.get(mode, mode.capitalize())

                with st.expander(f"Flashcard {idx}: {quiz_data.get('flashcard_concept', 'N/A')} ({mode_display})", expanded=False):
                    if timestamp:
                        try:
                            from datetime import datetime, timezone
                            if isinstance(timestamp, str):
                                stamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            else:
                                stamp = timestamp
                            display_time = stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                            st.caption(f"Generated: {display_time}")
                        except (ValueError, AttributeError):
                            st.caption(f"Generated: {timestamp}")

                    st.markdown("**Focus:**")
                    st.success(quiz_data.get('quiz_text', 'N/A'))

                    st.markdown("**Concept Title:**")
                    st.markdown(f"*{quiz_data.get('flashcard_concept', 'N/A')}*")

                    st.markdown("**Mode:**")
                    st.info(mode_display)

                    st.markdown("**Rationale:**")
                    st.info(quiz_data.get('flashcard_rationale', 'N/A'))
        else:
            st.warning("No flashcards found in this session.")

        if user_id:
            st.caption(f"User ID: {user_id}")

elif not st.session_state.quiz_started and not st.session_state.practice_history:
    st.markdown("---")
    st.info("📚 No practice history available yet. Start practicing to see your flashcards here!")
