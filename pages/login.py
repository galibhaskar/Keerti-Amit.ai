import streamlit as st

def login():
    print("Logging in...")
    
    ROLES = ["User", "Guest"]  

    st.header("Log in")

    role = st.selectbox("Choose your role", ROLES)

    if role == "Guest":
        st.info("You are logging in as a Guest. Limited access will be provided.")
        
        username = "Guest"
    
    else:
        username = st.text_input("Username")

    if st.button("Log in"):
        st.session_state.role = role
        st.session_state.username = username
        st.rerun()

login()