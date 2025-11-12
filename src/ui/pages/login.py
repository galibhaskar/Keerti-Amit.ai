"""Login page - Log in to your account."""

import streamlit as st

def login():
    st.header("Log in")
    
    ROLES = ["User", "Guest"]  

    role = st.selectbox("Choose your role", ROLES, key="login_role_select")

    if role == "Guest":
        st.info("You are logging in as a Guest. Limited access will be provided.")
        username = "Guest"
    else:
        username = st.text_input("Username", key="login_username_input")

    if st.button("Log in", type="primary"):
        # Validate
        if role is None:
            st.error("Please select a role.")
            return
        
        if role == "User" and (not username or not username.strip()):
            st.error("Please enter a username.")
            return
        
        # Set session state
        st.session_state.role = role
        st.session_state.username = username.strip() if username else "Guest"
        # Mark that we just logged in to ensure navigation resets
        st.session_state.just_logged_in = True
        
        st.success(f"Logged in successfully as {st.session_state.username}!")
        st.rerun()

# Only call login() if this file is run directly or through navigation
if __name__ == "__main__" or True:
    login()