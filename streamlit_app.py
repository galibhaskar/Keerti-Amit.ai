import streamlit as st
from pages import get_pages_for_role
from services.ingest_docs import EmbeddingQueue
from utilities.helpers import initialize_data_json

st.markdown("""
    <style>
    /* Target the main block container and reduce top padding */
    .block-container {
        padding-top: 1rem;       /* Reduced vertical space at the top */
        padding-bottom: 0rem;    /* Reduced vertical space at the bottom */
        padding-left: 2rem;      /* Optional: Adjust left padding */
        padding-right: 2rem;     /* Optional: Adjust right padding */
    }

    /* Adjust the specific font size for st.title, which uses h1 */
    h1 {
        margin-top: 0rem;       /* Remove default margin above the title */
        font-size: 3.5rem;      /* Initial Font Size for Rival-Ally AI */
    }
    
    /* Adjust the font size for the Practice Mode header, which uses h2/h3 */
    h2 {
        font-size: 2.5rem;      /* Initial Font Size for Practice Mode */
    }
    
    </style>
    """, unsafe_allow_html=True)

# --- Initialize session state variables ---
if "role" not in st.session_state:
    st.session_state.role = None

# --- Ensure a single shared background queue is running
if "embed_queue" not in st.session_state:
    st.session_state.embed_queue = EmbeddingQueue()
    st.session_state.embed_queue.start()

initialize_data_json()  # Ensure resource config file exists

# --- Page Layout ---
# st.title("Rival-Ally AI")
st.markdown("# Rival-Ally AI")

# Sidebar User Info
with st.sidebar:
    if st.session_state.role is not None:
        st.write(f"Logged in as : {st.session_state.get('username', 'Guest')}")

# Page Navigation Setup
pages = get_pages_for_role(st.session_state.get("role"))
st.navigation(pages).run()