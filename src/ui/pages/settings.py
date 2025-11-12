"""Settings page for application configuration."""

import streamlit as st
from services.config_manager import (
    get_user_settings,
    update_user_settings,
    get_available_models,
    get_flashcard_statistics,
    get_battle_statistics,
)
from services.practice import load_flashcard_history
from services.battle import load_battle_history, format_history_option_label, save_battle_history
from config.models import PROVIDERS
import json


st.header("Settings")
st.write("Configure your application settings and view statistics.")

# Initialize settings
settings = get_user_settings()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Model Configuration",
    "📊 Statistics",
    "📚 Flashcard History",
    "⚔️ Battle History"
])

# Tab 1: Model Configuration
with tab1:
    st.subheader("LLM Model Configuration")
    
    # Provider selection
    provider_options = list(PROVIDERS.keys())
    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=provider_options.index(settings.get("llm_provider", "GROQ")) if settings.get("llm_provider") in provider_options else 0,
        help="Select the LLM provider to use"
    )
    
    # Model selection
    available_models = get_available_models()
    current_model = settings.get("model", "llama-3.1-8b-instant")
    model_display = f"GROQ: {current_model}" if not any(current_model in m for m in available_models) else next((m for m in available_models if current_model in m), available_models[0])
    
    selected_model_display = st.selectbox(
        "Model",
        options=available_models,
        index=available_models.index(model_display) if model_display in available_models else 0,
        help="Select the specific model to use"
    )
    
    # Extract model name from display string
    selected_model = selected_model_display.split(": ")[-1] if ": " in selected_model_display else selected_model_display
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=float(settings.get("temperature", 0.5)),
        step=0.1,
        help="Controls randomness in generation. Lower = more deterministic."
    )
    
    # Max tokens
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4096,
        value=int(settings.get("max_tokens", 1024)),
        step=100,
        help="Maximum number of tokens to generate"
    )
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        new_settings = {
            "llm_provider": selected_provider,
            "model": selected_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        update_user_settings(new_settings)
        st.success("Settings saved successfully!")
        st.rerun()

# Tab 2: Statistics
with tab2:
    st.subheader("Usage Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Practice Mode")
        flashcard_stats = get_flashcard_statistics()
        st.metric("Total Flashcards", flashcard_stats["total_flashcards"])
        st.metric("Topics Covered", flashcard_stats["topics_covered"])
        
        if flashcard_stats["modes_used"]:
            st.markdown("**Modes Used:**")
            for mode, count in flashcard_stats["modes_used"].items():
                st.write(f"- {mode.capitalize()}: {count}")
    
    with col2:
        st.markdown("### ⚔️ Battle Mode")
        battle_stats = get_battle_statistics()
        st.metric("Total Battles", battle_stats["total_battles"])
        st.metric("Concepts Covered", battle_stats["concepts_covered"])
        st.metric("Avg Questions per Battle", battle_stats["average_questions"])

# Tab 3: Flashcard History
with tab3:
    st.subheader("Flashcard History")
    
    history = load_flashcard_history()
    
    if not history:
        st.info("No flashcard history found.")
    else:
        st.write(f"Total flashcards: {len(history)}")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            topics = sorted(set(entry.get("topic", "Unknown") for entry in history))
            selected_topic = st.selectbox("Filter by Topic", ["All"] + topics)
        
        with col2:
            modes = sorted(set(entry.get("mode", "unknown") for entry in history))
            selected_mode = st.selectbox("Filter by Mode", ["All"] + modes)
        
        # Filter history
        filtered_history = history
        if selected_topic != "All":
            filtered_history = [h for h in filtered_history if h.get("topic") == selected_topic]
        if selected_mode != "All":
            filtered_history = [h for h in filtered_history if h.get("mode") == selected_mode]
        
        # Display history
        for idx, entry in enumerate(reversed(filtered_history[-20:]), 1):  # Show last 20
            with st.expander(f"Flashcard {len(filtered_history) - idx + 1}: {entry.get('topic', 'Unknown')} ({entry.get('mode', 'unknown')})"):
                quiz_data = entry.get("quiz_data", {})
                st.write(f"**Topic:** {entry.get('topic', 'Unknown')}")
                st.write(f"**Mode:** {entry.get('mode', 'unknown')}")
                st.write(f"**Timestamp:** {entry.get('timestamp', 'Unknown')}")
                st.write(f"**Concept:** {quiz_data.get('flashcard_concept', 'N/A')}")
                st.write(f"**Focus:** {quiz_data.get('quiz_text', 'N/A')}")
                st.info(quiz_data.get('flashcard_rationale', 'N/A'))
        
        # Export option
        if st.button("📥 Export History (JSON)"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(history, indent=2),
                file_name="flashcard_history.json",
                mime="application/json"
            )

# Tab 4: Battle History
with tab4:
    st.subheader("Battle History")
    
    history = load_battle_history()
    
    if not history:
        st.info("No battle history found.")
    else:
        st.write(f"Total battles: {len(history)}")
        
        # Battle selector
        battle_indices = list(range(len(history)))
        selected_idx = st.selectbox(
            "Select Battle to View",
            battle_indices,
            format_func=lambda idx: format_history_option_label(idx, history[idx]),
            index=0
        )
        
        if selected_idx is not None:
            battle = history[selected_idx]
            
            st.markdown("### Battle Details")
            st.write(f"**Concept:** {battle.get('concept', 'Unknown')}")
            st.write(f"**Saved At:** {battle.get('saved_at', 'Unknown')}")
            
            # Questions
            questions = battle.get("questions", [])
            if questions:
                st.markdown("### Questions Asked")
                for q in questions:
                    st.write(f"**Q{q.get('id', '?')}** ({q.get('difficulty', 'N/A')}): {q.get('question', 'N/A')}")
            
            # Feedback
            feedback = battle.get("feedback", {})
            if feedback:
                st.markdown("### Feedback")
                score = feedback.get("score", "needs-work").lower()
                if score == "excellent":
                    st.success(f"**Score:** {feedback.get('verdict', 'Excellent work!')}")
                elif score == "good":
                    st.info(f"**Score:** {feedback.get('verdict', 'Strong showing.')}")
                elif score == "fair":
                    st.warning(f"**Score:** {feedback.get('verdict', 'Some gaps to address.')}")
                else:
                    st.error(f"**Score:** {feedback.get('verdict', 'Needs more work.')}")
                
                strengths = feedback.get("strengths", [])
                if strengths:
                    st.markdown("**Strengths:**")
                    for strength in strengths:
                        st.write(f"- {strength}")
                
                improvements = feedback.get("improvement_opportunities", [])
                if improvements:
                    st.markdown("**Improvement Opportunities:**")
                    for improvement in improvements:
                        st.write(f"- {improvement}")
            
            # Delete button
            if st.button("🗑️ Delete This Battle", type="secondary"):
                history.pop(selected_idx)
                save_battle_history(history)
                st.success("Battle deleted!")
                st.rerun()
        
        # Export option
        if st.button("📥 Export All Battles (JSON)"):
            st.download_button(
                label="Download JSON",
                data=json.dumps(history, indent=2),
                file_name="battle_history.json",
                mime="application/json"
            )
