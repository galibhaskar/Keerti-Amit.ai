import streamlit as st
import json
import re
from groq import Groq
import chromadb
from database.vector_db import get_chroma_collection
from config.app_config import VECTOR_DB_PATH, COLLECTION_NAME

# --- 1. CONFIGURATION & CLIENT INITIALIZATION ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    # Using a fast, supported model for quizzes
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
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Welcome! Enter a concept to generate a quiz."})
    
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
    
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False

if "last_topic" not in st.session_state:
    st.session_state.last_topic = None

if "flashcard_added_to_chat" not in st.session_state:
    st.session_state.flashcard_added_to_chat = False
    
# State to track previously asked questions for the current topic
if "previous_quiz_titles" not in st.session_state:
    st.session_state.previous_quiz_titles = []


# --- 3. RAG/RETRIEVAL HELPER FUNCTIONS ---

@st.cache_resource 
def get_retriever_collection():
    """Initializes and returns the Chroma collection."""
    # NOTE: This assumes your config and database files exist and are correctly set up.
    return get_chroma_collection(
        path=VECTOR_DB_PATH, 
        collection_name=COLLECTION_NAME
    )

def retrieve_context(query: str, n_results: int = 1) -> str:
    """
    Queries the ChromaDB collection for context and cleans the output.
    Using n_results=1 to conserve prompt tokens.
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
                # Remove non-printable and control characters
                cleaned_text = re.sub(r'[^\x20-\x7E\s]+', '', doc_text)
                cleaned_text = cleaned_text.strip()
                
                if cleaned_text:
                    cleaned_documents.append(cleaned_text)

            if cleaned_documents:
                context = "\n---\n".join(cleaned_documents)
                return f"Retrieved Context:\n{context}"
        
        return "No relevant context found in the knowledge base."
        
    except Exception as e:
        # NOTE: Pass is used here to avoid crashing the app if RAG fails, 
        # allowing Groq to use general knowledge.
        return f"RAG Retrieval Error: Could not load context. {e}"


# --- 4. LLM & PROMPT HELPER FUNCTIONS ---

def generate_llm_response(prompt: str) -> str:
    """
    Sends a structured prompt to the Groq API using the SDK.
    """
    messages = [
        # System prompt focused on JSON output, allowing flexible expertise
        {"role": "system", "content": "You are an expert in every concept the user asks. Always respond ONLY with a JSON object as requested by the user."},
        {"role": "user", "content": prompt}
    ]

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.5, # Increased temperature for question variety
            response_format={"type": "json_object"}, 
            max_tokens=1024 # Increased for complete JSON output
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return "ERROR_API_FAILED"


def get_structured_quiz_prompt(topic: str) -> str:
    """
    Generates a structured prompt template, including RAG context and exclusion rules.
    """
    
    # RAG Retrieval Step (n_results=1 is assumed in the function definition)
    context = retrieve_context(topic) 
    
    # Exclusion Rule based on session history
    previous_titles = st.session_state.get('previous_quiz_titles', [])
    exclusion_instruction = ""
    if previous_titles:
        title_list = "\n- ".join(previous_titles)
        exclusion_instruction = f"""
        **EXCLUSION RULE:** You MUST NOT generate a question related to any of the following previous concepts/titles:
        - {title_list}
        """

    # Final Augmented Prompt
    prompt = f"""
    You are an expert in the given topic. 
    
    {exclusion_instruction}
    
    You MUST use the following **CONTEXT** from the course material to formulate the quiz question and flashcard explanation. If the CONTEXT is generic, you may use external knowledge, but always prioritize the context provided.
    
    --- START CONTEXT ---
    {context}
    --- END CONTEXT ---
    
    Your task is to generate one challenging quiz question and its corresponding flashcard explanation focused *only* on the topic: **{topic}**.
    
    The quiz should be about the topics given the user, starting from basics to advanced.
    
    Output the following JSON structure. Do not include any other text:
    
    {{
        "quiz_type": "Code/Concept",
        "quiz_text": "Write the detailed quiz question here.",
        "answer": "The correct answer or value.",
        "flashcard_concept": "A concise title for the reinforcement concept.",
        "flashcard_rationale": "A detailed, technical explanation of why the answer is correct and the underlying  principle."
    }}
    """
    return prompt


def clean_and_parse_json(text: str):
    """Removes common LLM formatting (like ```json) and attempts to parse the JSON string."""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*```', '', text).strip()
    return json.loads(text)


def generate_next_quiz():
    """Triggers a new quiz generation using the stored last topic."""
    if st.session_state.last_topic:
        st.session_state.current_quiz = None
        st.session_state.show_answer = False
        st.session_state.flashcard_added_to_chat = False
        
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**Continuing quiz on:** *{st.session_state.last_topic}*"}
        )
        
        st.rerun()


# --- 5. CUSTOM CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
    h1 { margin-top: 0rem; margin-bottom: 0rem; font-size: 3.5rem; }
    h2 { font-size: 2.5rem; }
    </style>
    """, unsafe_allow_html=True)


# --- 6. APPLICATION LAYOUT ---
st.markdown("# Rival-Ally AI") 
st.markdown("## Practice Mode")
st.write("Practice your knowledge with Quizes and Flashcards :)")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handler (New Topic) ---
if prompt := st.chat_input("Enter a concept to quiz yourself on..."):
    # Clear old quiz state and history for a new topic
    st.session_state.current_quiz = None
    st.session_state.show_answer = False
    st.session_state.flashcard_added_to_chat = False
    st.session_state.previous_quiz_titles = [] # IMPORTANT: Clear titles for new topic
    
    st.session_state.last_topic = prompt
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Generating quiz on '{prompt}' via {MODEL_NAME} (Groq)..."):
            
            llm_prompt = get_structured_quiz_prompt(prompt)
            json_response_text = generate_llm_response(llm_prompt)
            
            if json_response_text == "ERROR_API_FAILED":
                st.session_state.messages.append({"role": "assistant", "content": "API failed. Check connection and API Key."})
            
            else:
                try:
                    quiz_data = clean_and_parse_json(json_response_text)
                    st.session_state.current_quiz = quiz_data
                    
                    # Store the concept title to prevent repetition
                    concept_title = quiz_data.get('flashcard_concept', 'A general quiz question')
                    st.session_state.previous_quiz_titles.append(concept_title)

                    quiz_message = f"""
                    **🧠 Quiz on: {prompt}**
                    
                    **Question:** {quiz_data.get('quiz_text', 'N/A')}
                    
                    *Click 'Show Answer' below the chat window to view the flashcard!*
                    """
                    st.markdown(quiz_message)
                    st.session_state.messages.append({"role": "assistant", "content": quiz_message})
                    
                except json.JSONDecodeError:
                    error_message = f"Error: Failed to parse the LLM's response. Raw output was: {json_response_text[:300]}..."
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": "Failed to generate a valid quiz. Try again."})
                    st.session_state.current_quiz = None

# --- Handler for Next Question Rerun ---
if st.session_state.current_quiz is None and st.session_state.last_topic:
    prompt = st.session_state.last_topic # Use the stored topic
    
    with st.chat_message("assistant"):
        with st.spinner(f"Generating next quiz question on '{prompt}'..."):
            
            llm_prompt = get_structured_quiz_prompt(prompt)
            json_response_text = generate_llm_response(llm_prompt)
            
            if json_response_text == "ERROR_API_FAILED":
                st.session_state.messages.append({"role": "assistant", "content": "API failed. Cannot generate next question."})
                st.session_state.last_topic = None 
            
            else:
                try:
                    quiz_data = clean_and_parse_json(json_response_text)
                    st.session_state.current_quiz = quiz_data
                    
                    # Store the concept title to prevent repetition
                    concept_title = quiz_data.get('flashcard_concept', 'A general quiz question')
                    st.session_state.previous_quiz_titles.append(concept_title)

                    quiz_message = f"""
                    **🧠 Quiz Continuation on:** {prompt}
                    
                    **Question:** {quiz_data.get('quiz_text', 'N/A')}
                    
                    *Click 'Show Answer' below the chat window to view the flashcard!*
                    """
                    st.markdown(quiz_message)
                    st.session_state.messages.append({"role": "assistant", "content": quiz_message})
                    
                except Exception:
                     st.error("Failed to generate the next question. Try a new topic.")
                     st.session_state.current_quiz = None
                     st.session_state.last_topic = None 
            

# --- 7. SEPARATE QUIZ/FLASHCARD DISPLAY AREA ---
st.markdown("---")

if st.session_state.current_quiz:
    quiz_data = st.session_state.current_quiz
    
    st.subheader("💡 Your Flashcard Reinforcement")

    # 1. Show Answer Button
    if not st.session_state.show_answer:
        if st.button("Show Answer"):
            st.session_state.show_answer = True
            st.rerun()
    
    # 2. Display Answer and Flashcard Rationale
    if st.session_state.show_answer:
        
        # Add Flashcard to Chat History (Only once per quiz)
        if not st.session_state.flashcard_added_to_chat:
            flashcard_message = f"""
            **✅ Answer:** {quiz_data['answer']}
            
            ---
            **Flashcard Concept:** *{quiz_data['flashcard_concept']}*
            
            **Rationale:** {quiz_data['flashcard_rationale']}
            """
            st.session_state.messages.append({"role": "assistant", "content": flashcard_message})
            st.session_state.flashcard_added_to_chat = True
            st.rerun() # Rerun to show the updated chat history instantly

        # RENDER THE FLASHCARD IN THE DEDICATED AREA
        st.success(f"**Answer:** {quiz_data['answer']}")
        st.markdown(f"**Concept:** *{quiz_data['flashcard_concept']}*")
        st.info(quiz_data['flashcard_rationale'])


    # 3. Next Question Button
    # Only available after the answer is shown (ensuring review)
    if st.session_state.show_answer: 
        if st.button("Next Question on This Topic"):
            generate_next_quiz()