"""Logout page - Log out of your account."""

import streamlit as st

def logout():
    print("Logging out...")
    
    st.session_state.role = None

    st.rerun()