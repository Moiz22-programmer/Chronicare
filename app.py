import streamlit as st
from styles import apply_custom_styles

# Page configuration
st.set_page_config(
    page_title="ChroniCare - Advanced Diagnostics System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme in session state to prevent widget key garbage collection
if "active_theme" not in st.session_state:
    st.session_state["active_theme"] = "Dark"

# Sidebar branding: logo and theme selector
st.sidebar.image("assets/logo.png", width=120)
st.sidebar.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
theme_index = 0 if st.session_state["active_theme"] == "Dark" else 1
theme = st.sidebar.radio("Theme Mode", options=["Dark", "Light"], index=theme_index, key="theme_selector_widget")
st.session_state["active_theme"] = theme

# Load and apply custom UI styles (will dynamically check active_theme)
apply_custom_styles()

# Define the exact 5 pages in navigation
home_page = st.Page("pages/1_home.py", title="Home", icon="🏠", default=True)
heart_page = st.Page("pages/3_heart.py", title="Heart Disease", icon="❤️")
diabetes_page = st.Page("pages/2_diabetes.py", title="Diabetes", icon="🩸")
kidney_page = st.Page("pages/4_kidney.py", title="Kidney Disease", icon="🫘")
liver_page = st.Page("pages/5_liver.py", title="Liver Disease", icon="🟡")

# Group and initialize the navigation structure
pg = st.navigation([home_page, heart_page, diabetes_page, kidney_page, liver_page])
pg.run()
