"""Main Streamlit application entry point."""
import streamlit as st
from ui.navigation import get_pages_for_role
from services.ingestion.queue import EmbeddingQueue
from utils.helpers import initialize_data_json

# st.markdown("""
#     <style>
#     /* Target the main block container and reduce top padding */
#     .block-container {
#         padding-top: 1rem;       /* Reduced vertical space at the top */
#         padding-bottom: 0rem;    /* Reduced vertical space at the bottom */
#         padding-left: 2rem;      /* Optional: Adjust left padding */
#         padding-right: 2rem;     /* Optional: Adjust right padding */
#     }

#     /* Adjust the specific font size for st.title, which uses h1 */
#     h1 {
#         margin-top: 0rem;       /* Remove default margin above the title */
#         font-size: 3.5rem;      /* Initial Font Size for Rival-Ally AI */
#     }
    
#     /* Adjust the font size for the Practice Mode header, which uses h2/h3 */
#     h2 {
#         font-size: 2.5rem;      /* Initial Font Size for Practice Mode */
#     }
    
#     </style>
#     """, unsafe_allow_html=True)

# --- Initialize session state variables ---
if "role" not in st.session_state:
    st.session_state.role = None

# Track if user just logged in to handle navigation transition
just_logged_in = st.session_state.get("just_logged_in", False)
if just_logged_in:
    # Clear the flag
    st.session_state.just_logged_in = False

# --- Ensure a single shared background queue is running
if "embed_queue" not in st.session_state:
    st.session_state.embed_queue = EmbeddingQueue()

    st.session_state.embed_queue.start()

initialize_data_json()  # Ensure resource config file exists

# --- Page Layout ---
st.markdown("# Keerti-Amit AI")

# Sidebar User Info
with st.sidebar:
    if st.session_state.role is not None:
        st.write(f"Logged in as : {st.session_state.get('username', 'Guest')}")

# Page Navigation Setup
role = st.session_state.get("role")

try:
    pages = get_pages_for_role(role)
except Exception as e:
    st.error(f"Error getting pages for role '{role}': {e}")
    import traceback
    st.code(traceback.format_exc())
    pages = None

# Only render navigation if pages are available
if pages:
    try:
        # Create navigation - Streamlit will automatically show the first page
        # when navigation structure changes (e.g., after login)
        # When user just logged in, navigation will show the first page in the
        # first category, which is now "Data Ingestion" (default landing page)
        nav = st.navigation(pages)
        nav.run()
    except Exception as e:
        st.error(f"Error running navigation: {e}")
        import traceback
        with st.expander("Error Details", expanded=True):
            st.code(traceback.format_exc())
        st.write("**Debug Info:**")
        st.write("- Role:", role)
        st.write("- Just logged in:", just_logged_in)
        st.write("- Pages type:", type(pages))
        if isinstance(pages, dict):
            st.write("- Page categories:", list(pages.keys()))
            for key, value in pages.items():
                st.write(f"  - {key}: {len(value) if isinstance(value, list) else 'N/A'} pages")
else:
    # If no pages available (shouldn't happen, but handle gracefully)
    st.warning("No pages available. Please log in.")

