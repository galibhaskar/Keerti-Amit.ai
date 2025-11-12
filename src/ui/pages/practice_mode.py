"""Practice Mode page - Interactive flashcards and quizzes."""

import streamlit as st
from services.practice import (
    generate_flashcard,
    save_flashcard_to_json,
)
from services.config_manager import get_user_settings
from utils.json_parser import clean_and_parse_json

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { margin-top: 0rem; margin-bottom: 0rem; font-size: 3.5rem; }
    h2 { font-size: 2.5rem; }
    </style>
    """, unsafe_allow_html=True)

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
                st.session_state.current_quiz = quiz_data

                # Save to log
                save_flashcard_to_json(
                    topic=topic,
                    mode=st.session_state.difficulty,
                    data=quiz_data,
                )

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
                st.rerun()
