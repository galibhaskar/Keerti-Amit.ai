import streamlit as st
from pages import get_pages_for_role

# --- Initialize session state variables ---
if "role" not in st.session_state:
    st.session_state.role = None

# --- Page Layout ---
st.title("Second Brain Application")

# Sidebar User Info
with st.sidebar:
    if st.session_state.role is not None:
        st.write(f"Logged in as : {st.session_state.get('username', 'Guest')}")

# Page Navigation Setup
pages = get_pages_for_role(st.session_state.get("role"))
st.navigation(pages).run()