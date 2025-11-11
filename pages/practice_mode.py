import streamlit as st
import json
import re
from groq import Groq
import chromadb
from datetime import datetime
import os 


try:
    from database.vector_db import get_chroma_collection
    from config.app_config import VECTOR_DB_PATH, COLLECTION_NAME
except ImportError:
    st.warning("Could not import vector_db or app_config. RAG functionality may be impaired.")
    class DummyRetriever:
        def query(self, query_texts, n_results, include): return {'documents': [[]]}
    def get_chroma_collection(path, collection_name): return DummyRetriever()
    VECTOR_DB_PATH, COLLECTION_NAME = "", ""


# --- CONFIGURATION & CLIENT INITIALIZATION ---

LOG_FILE = "flashcard_log.json" 

try:
    # Use the key from the secrets file provided in context
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    MODEL_NAME = "llama-3.1-8b-instant" 
    
    @st.cache_resource
    def get_groq_client():
        return Groq(api_key=GROQ_API_KEY)

    client = get_groq_client()
    
except KeyError:
    st.error("🚨 Configuration Error: GROQ_API_KEY not found in .streamlit/secrets.toml.")
    st.stop()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()


# --- 2. SESSION STATE MANAGEMENT ---
    
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False 

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "concept" 
    
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None

# NEW STATE: Stores the title of the current concept for "easy" and "example" modes
if "current_concept_title" not in st.session_state:
    st.session_state.current_concept_title = None
    

if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

if "previous_quiz_titles" not in st.session_state:
    st.session_state.previous_quiz_titles = []

if "user_id" not in st.session_state:
     st.session_state.user_id = "local_user" 


# --- 3. LOCAL FILE LOGGING FUNCTION ---

def save_flashcard_to_json(topic: str, mode: str, data: dict):
    """Saves the generated flashcard data to a local JSON file."""
    
    # 1. Create the new log entry
    log_entry = {
        "user_id": st.session_state.user_id,
        "topic": topic,
        "mode": mode,
        "timestamp": datetime.now().isoformat(), # Use ISO format for JSON compatibility
        "quiz_data": data,
    }
    
    # 2. Read existing data (or initialize if file doesn't exist)
    all_logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    all_logs = json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Error reading {LOG_FILE}: {e}. Starting a fresh log.")
            all_logs = []

    # 3. Append new entry
    all_logs.append(log_entry)

    # 4. Write all data back to the file
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=4)
        st.toast(f"Flashcard saved to {LOG_FILE}", icon="💾")
    except IOError as e:
        st.error(f"Error writing to local file {LOG_FILE}: {e}")


# --- 4. RAG/RETRIEVAL HELPER FUNCTIONS ---

@st.cache_resource 
def get_retriever_collection():
    """Initializes and returns the Chroma collection."""
    return get_chroma_collection(
        path=VECTOR_DB_PATH, 
        collection_name=COLLECTION_NAME
    )

def retrieve_context(query: str, n_results: int = 1) -> str:
    """
    Queries the Chroma DB collection for context and cleans the output.
    """
    try:
        collection = get_retriever_collection()
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results, 
            include=['documents'] 
        )
        
        if results and results.get('documents') and results['documents'][0]:
            cleaned_documents = []
            for doc_text in results['documents'][0]:
                cleaned_text = re.sub(r'[^\x20-\x7E\s]+', '', doc_text)
                cleaned_text = cleaned_text.strip()
                
                if cleaned_text:
                    cleaned_documents.append(cleaned_text)

            if cleaned_documents:
                context = "\n---\n".join(cleaned_documents)
                return f"Retrieved Context:\n{context}"
        
        return "No relevant context found in the knowledge base."
        
    except Exception as e:
        return f"RAG Retrieval Error: Could not load context. {e}"


# --- 5. LLM & PROMPT HELPER FUNCTIONS ---

def generate_llm_response(prompt: str) -> str:
    """
    Sends a structured prompt to the Groq API using the SDK.
    """
    # System prompt focused on JSON output, allowing flexible expertise
    messages = [
        {"role": "system", "content": "You are a strict RAG assistant. Use ONLY the text inside the CONTEXT block. "
  "If the CONTEXT is missing, off-topic, or low quality, you MUST output the exact NO_CONTEXT JSON specified by the user message. "
  "Never use prior knowledge or add information not in CONTEXT. Respond with a single JSON object only."
},
        {"role": "user", "content": prompt}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.5, 
            response_format={"type": "json_object"}, 
            max_tokens=1024 
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return "ERROR_API_FAILED"


def get_structured_quiz_prompt(topic: str, mode: str) -> str:
    """
    Generates a structured prompt template, tailored for flashcard output.
    Uses the current_concept_title for 'easy' and 'example' modes.
    """
    
    context = retrieve_context(topic) 
    previous_titles = st.session_state.get('previous_quiz_titles', [])
    exclusion_instruction = ""
    
    current_concept = st.session_state.current_concept_title if st.session_state.current_concept_title else topic
    
    if mode == "concept" and previous_titles:
        title_list = "\n- ".join(previous_titles)
        exclusion_instruction = f"""
        **EXCLUSION RULE:** You MUST NOT generate a flashcard related to any of the following previous concepts/titles:
        - {title_list}
        """
        
    # --- DYNAMIC INSTRUCTIONS BASED ON MODE ---
    mode_instructions = {
        "concept": "Your task is to generate one detailed flashcard focused on a **new, challenging concept** within: **{topic}**.",
        "easy": f"Your task is to generate a flashcard for the **EXACT SAME CONCEPT**: **'{current_concept}'**, but with a **significantly simpler explanation and analogy**.",
        "example": f"Your task is to generate a flashcard for the **EXACT SAME CONCEPT**: **'{current_concept}'**, focusing **exclusively on a practical code example or a real-world use case** to illustrate the principle."
    }
    
    # Final Augmented Prompt
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

    {mode_instructions.get(mode, mode_instructions['concept']).format(topic=topic)}

    **Provenance rule:** Use ONLY details present in CONTEXT. Do not invent, pad, or generalize beyond it. Avoid quoting any garbled/noisy substrings; summarize them in clean prose or omit them.

    The output must be a single JSON object.

    **CRITICAL FORMATTING INSTRUCTIONS (Read Carefully):**
    1. **JSON ESCAPING:** The content of 'flashcard_rationale' must be a single, valid JSON string. ALL content, including Markdown, code blocks, **newlines**, and **double quotes**, MUST be correctly JSON-escaped (e.g., newline becomes `\\n`, double quote becomes `\\"`).
    2. **STRICT FONT FIX:** DO NOT use any Markdown headings (`#`, `##`, `###`, etc.) within the 'flashcard_rationale'. Use **bold text** or *italics* instead for emphasis to avoid large fonts.
    3. **CODE BLOCKS:** If you use a code block (e.g., \`\`\`java\\n...\\n\`\`\`), ensure every character, including the three backticks, the language name, and the code lines, are represented in a single, properly escaped JSON string field.

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


def clean_and_parse_json(text: str):
    """Removes common LLM formatting (like ```json) and attempts to parse the JSON string."""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*```', '', text).strip()
    return json.loads(text)


def trigger_flashcard_generation():
    """Triggers a new flashcard generation using the stored last topic and difficulty mode."""
    # This stores the concept title before clearing the quiz object for regeneration
    if st.session_state.current_quiz and st.session_state.difficulty in ["easy", "example"]:
        st.session_state.current_concept_title = st.session_state.current_quiz.get('flashcard_concept')
        
    st.session_state.current_quiz = None
    st.rerun()


# --- 6. CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { margin-top: 0rem; margin-bottom: 0rem; font-size: 3.5rem; }
    h2 { font-size: 2.5rem; }
    </style>
    """, unsafe_allow_html=True)


# --- 7. APPLICATION LAYOUT & FLOW ---
st.markdown("## Focused Learning Mode")
st.write(f"Enter a topic to begin learning. You can generate flashcards on challenging concepts, request simpler explanations, or see practical examples.")

# --- Starting Form (If a quiz hasn't started) ---
if not st.session_state.quiz_started:
    with st.form(key='topic_form'):
        topic_input = st.text_input("Enter a Concept/Topic to Study:", key="new_topic_input")
        submit_button = st.form_submit_button(label='Start Flashcards')

        if submit_button and topic_input:
            # Initialize for the new topic
            st.session_state.last_topic = topic_input
            st.session_state.quiz_started = True
            st.session_state.current_quiz = None
            st.session_state.difficulty = "concept" 
            st.session_state.previous_quiz_titles = []
            st.session_state.current_concept_title = None # Reset concept title
            st.rerun()

# --- Main Flashcard Generation/Display Flow ---

if st.session_state.quiz_started:
    topic = st.session_state.last_topic
    
    # 1. GENERATE/REGENERATE FLASHCARD
    if st.session_state.current_quiz is None:
        
        with st.spinner(f"Generating {st.session_state.difficulty} flashcard on '{topic}'..."):
            
            mode = st.session_state.difficulty
            llm_prompt = get_structured_quiz_prompt(topic, mode)
            json_response_text = generate_llm_response(llm_prompt)
            
            if json_response_text == "ERROR_API_FAILED":
                st.error("API failed. Cannot generate flashcard.")
                st.session_state.quiz_started = False
            
            else:
                try:
                    quiz_data = clean_and_parse_json(json_response_text)
                    st.session_state.current_quiz = quiz_data
                    
                    # SAVE TO LOCAL JSON LOG HERE
                    save_flashcard_to_json(topic, mode, quiz_data)

                    # When generating a 'concept', store its title for subsequent 'easy'/'example' generations
                    if st.session_state.difficulty == "concept":
                         concept_title = quiz_data.get('flashcard_concept', 'A general concept')
                         st.session_state.previous_quiz_titles.append(concept_title)
                         st.session_state.current_concept_title = concept_title
                    
                    st.rerun() 

                except Exception as e:
                     st.error(f"Failed to generate the next flashcard. LLM output was not valid JSON. Error: {e}. Raw LLM text was: {json_response_text}")
                     st.session_state.current_quiz = None
                     st.session_state.quiz_started = False 

    # 2. DISPLAY FLASHCARD INTERFACE
    if st.session_state.current_quiz:
        quiz_data = st.session_state.current_quiz
        
        st.markdown("---")
        st.subheader(f"📖 Flashcard on: *{topic}*")

        # RENDER THE CARD CONTENT (all visible immediately)
        st.success(f"**Focus:** {quiz_data.get('quiz_text', 'N/A')}")
        
        st.markdown(f"**Concept Title:** *{quiz_data['flashcard_concept']}*")
        st.info(quiz_data['flashcard_rationale'])
        
        st.markdown("---")
        st.subheader("What do you want to do next?")
        
        # 3. NAVIGATION BUTTONS
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
                st.session_state.current_concept_title = None # Clear title to force a new concept query
                trigger_flashcard_generation()

        with col4:
            if st.button("🔄 Start Over", help="Go back to the main page to enter a new topic."):
                st.session_state.quiz_started = False
                st.session_state.current_quiz = None
                st.session_state.last_topic = None
                st.session_state.previous_quiz_titles = []
                st.session_state.current_concept_title = None
                st.rerun()