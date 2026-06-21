import streamlit as st
import os

def load_custom_styles():
    """Load the external CSS stylesheet for the ChroniCare Streamlit app.
    The CSS file is located in the `assets` directory as `style.css`.
    This function reads the file contents and injects it using `st.markdown`
    with `unsafe_allow_html=True`.
    """
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load custom styles: {e}")
