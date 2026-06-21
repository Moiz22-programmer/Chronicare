import streamlit as st
from styles import apply_custom_styles

# Apply custom styling
apply_custom_styles()


# Hero Header Section
st.markdown("""
    <div style="text-align: center; padding: 25px 0px 40px 0px;">
        <span style="background: rgba(0, 188, 212, 0.15); color: #00E5FF; padding: 10px 24px; border-radius: 50px; font-size: 0.8rem; font-weight: 800; letter-spacing: 0.12em; border: 2px solid rgba(0, 188, 212, 0.3); display: inline-block; text-transform: uppercase; box-shadow: 0 0 20px rgba(0, 188, 212, 0.2);">
            🏥 Advanced Medical Diagnostics
        </span>
        <h1 class="hero-title" style="font-size: 4.8rem; font-weight: 900; margin-top: 24px; margin-bottom: 16px; letter-spacing: -0.03em; line-height: 1.1; background: linear-gradient(135deg, #F8FAFC 0%, #00E5FF 50%, #00BCD4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 40px rgba(0, 188, 212, 0.2);">
            ChroniCare
        </h1>
        <p class="hero-subtitle" style="font-size: 1.4rem; color: #00BCD4; font-weight: 600; max-width: 900px; margin: 0 auto; letter-spacing: 0.02em;">
            Intelligent Chronic Disease Prevention & Prediction
        </p>
        <p class="hero-description" style="font-size: 1.05rem; color: #94A3B8; max-width: 700px; margin: 16px auto 0 auto; line-height: 1.7; font-weight: 400;">
            Harness the power of machine learning to detect, analyze, and manage four major chronic conditions with clinical precision.
        </p>
    </div>
""", unsafe_allow_html=True)

# Animated divider
st.markdown("""
    <div style="height: 2px; background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.5), transparent); margin-bottom: 40px;"></div>
""", unsafe_allow_html=True)

# Grid Layout for 4 Diseases
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
        <div class="perspective-container">
            <div class="disease-card-3d card-heart">
                <div class="live-dot dot-heart"></div>
                <div class="disease-card-content">
                    <div class="disease-icon">❤️</div>
                    <div class="disease-info">
                        <h4>Cardiovascular Risk</h4>
                        <p>Analyzes blood pressure, cholesterol, and cardiac parameters to identify coronary artery disease risk with advanced ML algorithms.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Screen for Heart Disease", key="btn_heart", use_container_width=True):
        st.switch_page("pages/3_heart.py")
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="perspective-container">
            <div class="disease-card-3d card-kidney">
                <div class="live-dot dot-kidney"></div>
                <div class="disease-card-content">
                    <div class="disease-icon">🫘</div>
                    <div class="disease-info">
                        <h4>Renal Function</h4>
                        <p>Evaluates kidney health through serum chemistry, electrolytes, and urinalysis to detect Chronic Kidney Disease early.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Screen for Kidney Disease", key="btn_kidney", use_container_width=True):
        st.switch_page("pages/4_kidney.py")

with col_right:
    st.markdown("""
        <div class="perspective-container">
            <div class="disease-card-3d card-diabetes">
                <div class="live-dot dot-diabetes"></div>
                <div class="disease-card-content">
                    <div class="disease-icon">🩸</div>
                    <div class="disease-info">
                        <h4>Glycemic Control</h4>
                        <p>Monitors blood glucose, insulin levels, and metabolic markers to predict and manage diabetes progression effectively.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Screen for Diabetes", key="btn_diabetes", use_container_width=True):
        st.switch_page("pages/2_diabetes.py")
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="perspective-container">
            <div class="disease-card-3d card-liver">
                <div class="live-dot dot-liver"></div>
                <div class="disease-card-content">
                    <div class="disease-icon">🟡</div>
                    <div class="disease-info">
                        <h4>Hepatic Health</h4>
                        <p>Assesses liver enzymes, bilirubin levels, and protein synthesis to identify liver disease and hepatic dysfunction.</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Screen for Liver Disease", key="btn_liver", use_container_width=True):
        st.switch_page("pages/5_liver.py")

st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

# Prompt Section
st.markdown("""
    <div class="cta-box">
        <h3 style="margin-top:0; color:#00E5FF; font-weight:800; font-size:1.5rem; letter-spacing:-0.01em;">🚀 Ready to Begin?</h3>
        <p style="font-size: 1.05rem; line-height: 1.7; color: #CBD5E1; max-width: 600px; margin: 12px auto 0 auto; font-weight: 400;">
            Click on any disease card above to enter patient vitals and receive immediate AI-powered diagnostic assessments with comprehensive clinical insights.
        </p>
        <div style="margin-top: 20px; display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; font-size: 0.9rem; color: #94A3B8;">
            <span style="background: rgba(0, 188, 212, 0.1); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(0, 188, 212, 0.2);">⚡ Real-time Analysis</span>
            <span style="background: rgba(0, 188, 212, 0.1); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(0, 188, 212, 0.2);">📊 Dual-Model Validation</span>
            <span style="background: rgba(0, 188, 212, 0.1); padding: 6px 14px; border-radius: 20px; border: 1px solid rgba(0, 188, 212, 0.2);">📄 PDF Reports</span>
        </div>
    </div>
""", unsafe_allow_html=True)
