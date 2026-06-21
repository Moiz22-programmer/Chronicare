import streamlit as st
import os

def apply_custom_styles():
    """Load custom CSS from assets/style.css and inject it, supporting dark/light mode dynamically."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )
    
    # Load stylesheet from assets
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load stylesheet: {e}")
        
    # Apply theme override based on session_state
    if st.session_state.get("active_theme") == "Light":
        st.markdown(
            """
            <style>
            :root {
                --bg-color: #F8FAFC !important;
                --text-color: #0F172A !important;
                --bg-image: radial-gradient(circle at 80% 20%, rgba(0, 188, 212, 0.08) 0%, transparent 50%),
                            radial-gradient(circle at 20% 80%, rgba(0, 150, 136, 0.05) 0%, transparent 50%) !important;
                
                --sidebar-bg: rgba(255, 255, 255, 0.95) !important;
                --sidebar-border: rgba(0, 188, 212, 0.2) !important;
                --sidebar-text: #1E293B !important;
                
                --card-bg: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.98) 100%) !important;
                --card-border: rgba(0, 188, 212, 0.2) !important;
                --card-text: #0F172A !important;
                --card-text-muted: #475569 !important;
                
                --input-bg: rgba(255, 255, 255, 0.95) !important;
                --input-border: rgba(0, 188, 212, 0.25) !important;
                --input-text: #0F172A !important;
                
                --button-bg: linear-gradient(135deg, #0097A7 0%, #006064 100%) !important;
                --button-shadow: rgba(0, 151, 167, 0.2) !important;
                --accent-color: #0097A7 !important;
                --border-hover: rgba(0, 151, 167, 0.5) !important;
                
                --tabs-bg: rgba(241, 245, 249, 0.7) !important;
                --tabs-border: rgba(0, 188, 212, 0.2) !important;
                --tab-active-text: #0097A7 !important;
            }
            .disease-card-3d, .cta-box {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(241, 245, 249, 0.95) 100%) !important;
                border-color: rgba(0, 188, 212, 0.2) !important;
                color: #0F172A !important;
            }
            .disease-info h4 {
                color: #0097A7 !important;
            }
            .disease-info p {
                color: #475569 !important;
            }
            .cta-box h3 {
                color: #0097A7 !important;
            }
            .cta-box p {
                color: #475569 !important;
            }
            
            /* High-contrast Label colors for all fields in Light Mode */
            label[data-testid="stWidgetLabel"],
            label[data-testid="stWidgetLabel"] p,
            label[data-testid="stWidgetLabel"] span,
            .stWidgetLabel,
            .stWidgetLabel p,
            .stWidgetLabel span {
                color: #1E293B !important;
                font-weight: 600 !important;
            }

            /* Selectbox container & inner selector background fixes */
            div[data-baseweb="select"] {
                background-color: #FFFFFF !important;
                border: 1px solid rgba(0, 188, 212, 0.3) !important;
                border-radius: 10px !important;
            }
            div[data-baseweb="select"] > div {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
                border: none !important;
            }
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] div[role="button"],
            div[data-baseweb="select"] div[data-testid="stMarkdownContainer"] p {
                color: #0F172A !important;
            }
            div[data-baseweb="select"] svg {
                fill: #0F172A !important;
                color: #0F172A !important;
            }

            /* Dropdown popover list items styling (when selectbox is clicked) */
            ul[role="listbox"],
            li[role="option"],
            div[data-baseweb="menu"] {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
            }
            li[role="option"]:hover,
            div[role="option"]:hover {
                background-color: #F1F5F9 !important;
                color: #0097A7 !important;
            }

            /* Text inputs and Number inputs */
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextInput"] input {
                color: #0F172A !important;
                background-color: #FFFFFF !important;
                border: 1px solid rgba(0, 188, 212, 0.3) !important;
                background-image: none !important;
            }
            /* Number input toggle buttons styling */
            div[data-testid="stNumberInput"] button {
                background-color: #F1F5F9 !important;
                color: #0F172A !important;
                border: 1px solid rgba(0, 188, 212, 0.2) !important;
            }
            div[data-testid="stNumberInput"] button:hover {
                background-color: #E2E8F0 !important;
                color: #0097A7 !important;
            }

            /* Slider label and value displays */
            div[data-testid="stSlider"] p, 
            div[data-testid="stSlider"] span,
            div[data-testid="stSlider"] div {
                color: #1E293B !important;
            }

            /* Form Card and Result Card text elements */
            .form-card, .result-card, .glass-card {
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.98) 100%) !important;
                border-color: rgba(0, 188, 212, 0.2) !important;
                color: #0F172A !important;
            }
            div[data-testid="stExpander"] {
                background-color: rgba(241, 245, 249, 0.85) !important;
                border-color: rgba(0, 188, 212, 0.2) !important;
            }
            button[data-baseweb="tab"] {
                color: #475569 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #0097A7 !important;
                border-color: #0097A7 !important;
            }
            .result-card p, .glass-card p, .form-card p {
                color: #475569 !important;
            }
            .result-card li, .glass-card li, .form-card li {
                color: #334155 !important;
            }

            /* File uploader styling */
            div[data-testid="stFileUploader"] section {
                background-color: #FFFFFF !important;
                border: 2px dashed rgba(0, 188, 212, 0.3) !important;
                color: #0F172A !important;
            }
            div[data-testid="stFileUploader"] section * {
                color: #475569 !important;
            }

            /* Info notification messages styling */
            div[data-testid="stNotification"] {
                color: #0F172A !important;
            }

            /* ═══ Dataframe / Table ─ full light-mode overrides ═══ */

            /* Outer wrapper: white background, light border */
            div[data-testid="stDataFrame"] {
                background-color: #FFFFFF !important;
                border: 1px solid rgba(0, 188, 212, 0.2) !important;
                border-radius: 12px !important;
                overflow: hidden !important;
            }

            /* All text inside the dataframe */
            div[data-testid="stDataFrame"] *,
            div[data-testid="stDataFrame"] th,
            div[data-testid="stDataFrame"] td,
            div[data-testid="stDataFrame"] span,
            div[data-testid="stDataFrame"] p {
                color: #0F172A !important;
            }

            /* Glide Data Grid (the actual rendered table canvas backdrop) */
            div[data-testid="stDataFrame"] > div,
            div[data-testid="stDataFrame"] > div > div,
            div[data-testid="stDataFrame"] [class*="glideDataEditor"],
            div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
                background-color: #FFFFFF !important;
                color: #0F172A !important;
            }

            /* Classic HTML table fallback (if Streamlit renders <table>) */
            div[data-testid="stDataFrame"] table {
                background-color: #FFFFFF !important;
                border-collapse: collapse !important;
            }
            div[data-testid="stDataFrame"] table thead th {
                background-color: #F1F5F9 !important;
                color: #0F172A !important;
                font-weight: 700 !important;
                border-bottom: 2px solid rgba(0, 188, 212, 0.25) !important;
                padding: 10px 14px !important;
            }
            div[data-testid="stDataFrame"] table tbody tr {
                background-color: #FFFFFF !important;
            }
            div[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
                background-color: #F8FAFC !important;
            }
            div[data-testid="stDataFrame"] table tbody tr:hover {
                background-color: #E0F7FA !important;
            }
            div[data-testid="stDataFrame"] table tbody td {
                color: #0F172A !important;
                padding: 10px 14px !important;
                border-bottom: 1px solid #E2E8F0 !important;
            }

            /* Scrollbar inside dataframe */
            div[data-testid="stDataFrame"] ::-webkit-scrollbar {
                width: 8px; height: 8px;
            }
            div[data-testid="stDataFrame"] ::-webkit-scrollbar-track {
                background: #F1F5F9;
            }
            div[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
                background: #CBD5E1; border-radius: 4px;
            }

            /* ═══ stTable fallback ═══ */
            div[data-testid="stTable"] table {
                background-color: #FFFFFF !important;
            }
            div[data-testid="stTable"] th {
                background-color: #F1F5F9 !important;
                color: #0F172A !important;
                font-weight: 700 !important;
            }
            div[data-testid="stTable"] td {
                color: #0F172A !important;
                background-color: #FFFFFF !important;
            }

            /* ═══ Download button light-mode ═══ */
            div[data-testid="stDownloadButton"] button {
                background: linear-gradient(135deg, #0097A7 0%, #006064 100%) !important;
                color: white !important;
                border: none !important;
            }

            /* ═══ "Clear History" & secondary buttons ═══ */
            button[kind="secondary"],
            button[data-testid="baseButton-secondary"] {
                background-color: #FFFFFF !important;
                color: #EF4444 !important;
                border: 2px solid #EF4444 !important;
            }
            button[kind="secondary"]:hover,
            button[data-testid="baseButton-secondary"]:hover {
                background-color: #FEF2F2 !important;
                color: #DC2626 !important;
                border-color: #DC2626 !important;
            }

            /* ═══ st.info / st.success / st.warning boxes ═══ */
            div[data-testid="stNotification"] {
                background-color: #F0F9FF !important;
                color: #0F172A !important;
                border-color: rgba(0, 188, 212, 0.3) !important;
            }
            div[data-testid="stNotification"] p,
            div[data-testid="stNotification"] span {
                color: #0F172A !important;
            }

            /* ═══ Markdown text inside cards ═══ */
            div[data-testid="stMarkdownContainer"] p,
            div[data-testid="stMarkdownContainer"] li,
            div[data-testid="stMarkdownContainer"] span {
                color: #334155 !important;
            }

            /* ═══ Tabs in light mode ═══ */
            div[data-baseweb="tab-list"] {
                background-color: rgba(241, 245, 249, 0.7) !important;
                border-bottom: 1px solid rgba(0, 188, 212, 0.2) !important;
            }
            div[data-baseweb="tab-panel"] {
                background-color: transparent !important;
                color: #334155 !important;
            }

            /* ═══ Radio button labels (Theme selector in sidebar) ═══ */
            div[data-testid="stRadio"] label span,
            div[data-testid="stRadio"] label p {
                color: #1E293B !important;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #0F172A !important;
            }
            .hero-title {
                background: linear-gradient(135deg, #0F172A 0%, #0097A7 50%, #006064 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }
            .section-header {
                color: #0097A7 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

def render_health_meter(label, value, is_high_risk=False, limit_label=""):
    color = "linear-gradient(90deg, #F87171 0%, #EF4444 100%)" if is_high_risk else "linear-gradient(90deg, #00BCD4 0%, #008080 100%)"
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; font-size: 1rem; font-weight: 700; color: #E2E8F0; letter-spacing:0.02em;">
                <span>{label}</span>
                <span>{limit_label} <span style="color: {'#F87171' if is_high_risk else '#00BCD4'}; font-size:1.1rem;">{value}%</span></span>
            </div>
            <div class="health-meter-container">
                <div class="health-meter-fill" style="width: {value}%; background: {color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
