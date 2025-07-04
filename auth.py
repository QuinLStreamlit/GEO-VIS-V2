"""
Simple password authentication for the application.
"""
import streamlit as st
import hashlib

# Password configuration
PASSWORD = "123456"
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_password():
    """Returns True if password is correct."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == PASSWORD_HASH:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Return True if password is validated
    if st.session_state.get("password_correct", False):
        return True

    # Show password input
    st.markdown(
        "<h1 style='text-align: center; margin: 3rem 0;'>Geotechnical Data Visualisation</h1>", 
        unsafe_allow_html=True
    )
    
    # Center the password input
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password",
            placeholder="Enter password"
        )
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("Password incorrect. Please try again.")
    
    return False