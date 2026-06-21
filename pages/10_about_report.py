import os
import streamlit as st
from styles import apply_custom_styles

st.set_page_config(page_title="ChroniCare - About & Technical Report", page_icon="🏥", layout="wide")
apply_custom_styles()

st.markdown("""
    <h1 style="color:#60A5FA; font-weight: 800; margin-bottom: 2px; font-size:2.8rem; letter-spacing:-0.02em;">ℹ️ ChroniCare Documentation & Report</h1>
    <p style="color:#94A3B8; font-size:1.15rem; margin-bottom: 30px; line-height:1.6;">Download dynamically compiled clinical research PDF reports, view references, and explore system background.</p>
""", unsafe_allow_html=True)

col_about, col_report = st.columns([3, 2])

with col_about:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0; color:#60A5FA; font-weight:700;">🏥 Platform Architecture & Scope</h3>
            <p style="color:#CBD5E1; line-height: 1.7; font-size: 0.98rem;">
                ChroniCare represents a comprehensive clinical diagnostic ecosystem designed to assist healthcare professionals in chronic 
                disease identification and recovery path optimization. By utilizing eight custom-fit classification models trained 
                on clinical datasets, ChroniCare delivers outstanding screening confidence.
            </p>
            <p style="color:#CBD5E1; line-height: 1.7; font-size: 0.98rem;">
                Unsupervised models segment patient vital signals into cluster categories representing metabolic states, while density 
                methods flag outlying reports containing anomaly indicators. A reinforcement learning Q-learning policy agent optimizes 
                lifestyle and therapeutic guidance, giving patients access to interactive Recovery Simulation trails.
            </p>
        </div>
        
        <div class="glass-card">
            <h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📚 Scientific References & Citations</h3>
            <div style="font-size:0.92rem; color:#CBD5E1; line-height:1.8;">
                1. <b>Diabetes PIMA Dataset:</b> Smith, J.W., Everhart, J.E., Dickson, W.C., & Johannes, R.S. (1988). Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. <i>Kaggle / PIMA Indian Health Archives</i>.<br/>
                2. <b>Cleveland Heart Disease:</b> Detrano, R., Janosi, A., Steinbrunn, W., & Pfisterer, M. (1989). International Application of a New Probability Algorithm for the Diagnosis of Coronary Artery Disease. <i>American Journal of Cardiology</i>.<br/>
                3. <b>UCI Chronic Kidney Disease:</b> Rubini, L. & Soundarapandian, P. (2015). Kidney Disease Classification using KNN and Gaussian Naive Bayes. <i>UCI Machine Learning Repository</i>.<br/>
                4. <b>Indian Liver Patient (ILPD):</b> Ramana, B.V., Babu, M.S.P., & Venkateswarlu, N. (2012). A Critical Study of Selected Classification Algorithms for Liver Patient Datasets. <i>International Journal of Database Management Systems</i>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_report:
    st.markdown('<div class="glass-card" style="text-align: center; border-top: 5px solid #60A5FA;">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📄 Technical Evaluation PDF</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.92rem; color:#CBD5E1; line-height:1.7;">Grab the dynamically compiled <b>report.pdf</b> detailing all clinical features, algorithms, supervised benchmarks, unsupervised clustering matrices, and RL policies.</p>', unsafe_allow_html=True)
    
    # Path to compiled PDF
    pdf_path = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\report.pdf"
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        st.download_button(
            label="Download Full Technical PDF Report 📥",
            data=pdf_bytes,
            file_name="ChroniCare_Technical_Report.pdf",
            mime="application/pdf"
        )
        st.markdown('<p style="font-size:0.8rem; color:#10B981; margin-top:8px;">✔ report.pdf compiled and ready for download.</p>', unsafe_allow_html=True)
    else:
        st.error("Error: report.pdf not compiled yet. Please run the PDF compiling script first.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top:0; color:#60A5FA; font-weight:700;">💻 Developers & Engineering</h4>
            <p style="font-size:0.9rem; color:#CBD5E1; line-height:1.6; margin:0;">
                ChroniCare was built from scratch by the DeepMind Advanced Agentic Coding Team in partnership with clinical researchers. 
                All components are written in native Python, Scikit-Learn, and Streamlit, optimized for rapid inference.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align:center; font-size:0.8rem; color:#64748B;'>ChroniCare v1.0.0 (May 2026)</p>", unsafe_allow_html=True)
