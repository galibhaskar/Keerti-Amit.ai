"""Page navigation and routing configuration."""
import streamlit as st
from ui.pages.logout import logout

settings_page = st.Page("./ui/pages/settings.py", title="Settings", icon=":material/settings:")

battle_mode_page = st.Page("./ui/pages/battle_mode.py", title="Battle Mode", icon=":material/swords:")

data_ingestion_page = st.Page("./ui/pages/data_ingestion.py", title="Data Ingestion", icon=":material/database:")

practice_mode_page = st.Page("./ui/pages/practice_mode.py", title="Practice Mode", icon=":material/school:")

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

login_page = st.Page("./ui/pages/login.py", title="Log in", icon=":material/login:")


def get_pages_for_role(role: str | None):
    """
    Get pages available for a given user role.

    Args:
        role: User role ("User", "Guest", or None)

    Returns:
        Dictionary mapping page categories to page lists
    """

    if role in ["User", "Guest"]:
        account_pages = [settings_page, logout_page]
        
        # Order matters: Streamlit shows the first page in the first category
        # Put Data Ingestion first as the default landing page after login
        pages = {
            "Data Ingestion": [data_ingestion_page],
            "Practice Mode": [practice_mode_page],
            "Battle Mode": [battle_mode_page],
            "Account": account_pages,
        }
        return pages
    else:
        print("Returning login page")
        return {"Account": [login_page]}

