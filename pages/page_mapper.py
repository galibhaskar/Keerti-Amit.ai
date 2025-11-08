import streamlit as st
from pages.logout import logout

settings_page = st.Page("./pages/settings.py", title="Settings", icon=":material/settings:")

battle_mode_page = st.Page("./pages/battle_mode.py", title="Battle Mode", icon=":material/swords:")

data_ingestion_page = st.Page("./pages/data_ingestion.py", title="Data Ingestion", icon=":material/database:")

practice_mode_page = st.Page("./pages/practice_mode.py", title="Practice Mode", icon=":material/school:")

logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

login_page = st.Page("./pages/login.py", title="Log in", icon=":material/login:")