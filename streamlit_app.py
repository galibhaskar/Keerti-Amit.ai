import streamlit as st
from pages import get_pages_for_role
from services.ingest_docs import EmbeddingQueue
from utilities.helpers import initialize_data_json

# --- Initialize session state variables ---
if "role" not in st.session_state:
    st.session_state.role = None

# --- Ensure a single shared background queue is running
if "embed_queue" not in st.session_state:
    st.session_state.embed_queue = EmbeddingQueue()
    st.session_state.embed_queue.start()

initialize_data_json()  # Ensure resource config file exists

# --- Page Layout ---
st.title("Second Brain Application")

# Sidebar User Info
with st.sidebar:
    if st.session_state.role is not None:
        st.write(f"Logged in as : {st.session_state.get('username', 'Guest')}")

# Page Navigation Setup
pages = get_pages_for_role(st.session_state.get("role"))
st.navigation(pages).run()