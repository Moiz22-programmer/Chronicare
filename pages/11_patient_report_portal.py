import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from styles import apply_custom_styles
from patient_report_generator import generate_patient_report

st.set_page_config(
    page_title="ChroniCare - Patient Report Portal",
    page_icon="📋",
    layout="wide"
)
apply_custom_styles()

# ─────────────────────────────────────────────
# Constants / Configuration
# ─────────────────────────────────────────────

MODELS_DIR = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"

DISEASE_CONFIG = {
    "Diabetes": {
        "color": "#3B82F6",
        "icon": "🍭",
        "model_files": ("diabetes_rf.pkl", "diabetes_lr.pkl"),
        "model_names": ("Random Forest", "Logistic Regression"),
        "features": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "feature_defaults": [2, 110, 72, 25, 80, 28.0, 0.35, 35],
        "feature_mins":     [0,  44,  40,  7,  14, 18.2, 0.08, 21],
        "feature_maxs":     [17, 200, 122, 99, 846, 67.1, 2.42, 81],
        "feature_steps":    [1, 1, 1, 1, 1, 0.1, 0.001, 1],
        "feature_labels": {
            "Pregnancies": "Number of Pregnancies",
            "Glucose": "Plasma Glucose Level (mg/dL)",
            "BloodPressure": "Diastolic Blood Pressure (mmHg)",
            "SkinThickness": "Skin Thickness (mm)",
            "Insulin": "2-Hr Serum Insulin (μU/mL)",
            "BMI": "Body Mass Index (kg/m²)",
            "DiabetesPedigreeFunction": "Diabetes Pedigree Score",
            "Age": "Patient Age (years)",
        },
        "csv_example": "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,PatientName\n2,120,72,25,80,28.5,0.35,40,John Doe\n5,150,85,30,120,35.2,0.75,52,Jane Smith",
    },
    "Heart Disease": {
        "color": "#EF4444",
        "icon": "❤️",
        "model_files": ("heart_svm.pkl", "heart_dt.pkl"),
        "model_names": ("SVM Classifier", "Decision Tree"),
        "features": ["age","sex","cp","trestbps","chol","fbs","restecg",
                     "thalach","exang","oldpeak","slope","ca","thal"],
        "feature_defaults": [54, 1, 1, 130, 246, 0, 1, 150, 0, 1.0, 1, 0, 2],
        "feature_mins":     [29, 0, 0,  94,  126, 0, 0,  71, 0, 0.0, 0, 0, 1],
        "feature_maxs":     [77, 1, 3, 200,  564, 1, 2, 202, 1, 6.2, 2, 4, 3],
        "feature_steps":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.1, 1, 1, 1],
        "feature_labels": {
            "age": "Patient Age",
            "sex": "Sex (1=Male, 0=Female)",
            "cp": "Chest Pain Type (0–3)",
            "trestbps": "Resting BP (mmHg)",
            "chol": "Serum Cholesterol (mg/dL)",
            "fbs": "Fasting Blood Sugar >120 (1=Yes)",
            "restecg": "Resting ECG (0/1/2)",
            "thalach": "Max Heart Rate (bpm)",
            "exang": "Exercise Angina (1=Yes)",
            "oldpeak": "ST Depression (Oldpeak)",
            "slope": "ST Slope (0/1/2)",
            "ca": "Major Vessels Colored (0–4)",
            "thal": "Thalassemia (1/2/3)",
        },
        "csv_example": "age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,PatientName\n54,1,1,130,246,0,1,150,0,1.0,1,0,2,John Doe\n62,0,0,145,310,1,2,120,1,2.5,0,2,3,Jane Smith",
    },
    "Kidney Disease": {
        "color": "#10B981",
        "icon": "🧪",
        "model_files": ("kidney_knn.pkl", "kidney_nb.pkl"),
        "model_names": ("K-Nearest Neighbors", "Naive Bayes"),
        "features": ["age","bp","sg","al","su","rbc","pc","pcc","ba","bgr",
                     "bu","sc","sod","pot","hemo","pcv","wc","rc","htn","dm",
                     "cad","appet","pe","ane"],
        "feature_defaults": [51,76,1.020,0,0,1,1,0,0,148,57,1.2,137,4.6,12.5,38,8400,4.7,0,0,0,1,0,0],
        "feature_mins":     [2, 50,1.005,0,0,0,0,0,0, 22, 1,0.4, 4,2.5, 3.1, 9,2200,2.1,0,0,0,0,0,0],
        "feature_maxs":     [90,180,1.025,5,5,1,1,1,1,490,391,32.0,163,47.0,17.8,54,26400,8.0,1,1,1,1,1,1],
        "feature_steps":    [1,1,0.005,1,1,1,1,1,1,1,1,0.1,1,0.1,0.1,1,100,0.1,1,1,1,1,1,1],
        "feature_labels": {
            "age":"Age","bp":"Blood Pressure (mmHg)","sg":"Specific Gravity","al":"Albumin (0–5)",
            "su":"Sugar (0–5)","rbc":"RBC (1=Normal)","pc":"Pus Cell (1=Normal)",
            "pcc":"Pus Cell Clumps (1=Yes)","ba":"Bacteria (1=Yes)","bgr":"Blood Glucose Random (mg/dL)",
            "bu":"Blood Urea (mg/dL)","sc":"Serum Creatinine (mg/dL)","sod":"Serum Sodium (mEq/L)",
            "pot":"Serum Potassium (mEq/L)","hemo":"Hemoglobin (g/dL)","pcv":"Packed Cell Volume (%)",
            "wc":"WBC Count (cells/μL)","rc":"RBC Count (million/μL)","htn":"Hypertension (1=Yes)",
            "dm":"Diabetes (1=Yes)","cad":"Coronary Artery Disease (1=Yes)","appet":"Appetite (1=Good)",
            "pe":"Pedal Edema (1=Yes)","ane":"Anemia (1=Yes)",
        },
        "csv_example": "age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane,PatientName\n51,76,1.02,0,0,1,1,0,0,148,57,1.2,137,4.6,12.5,38,8400,4.7,0,0,0,1,0,0,John Doe",
    },
    "Liver Disease": {
        "color": "#7C3AED",
        "icon": "🧬",
        "model_files": ("liver_xgb.pkl", "liver_gb.pkl"),
        "model_names": ("XGBoost", "Gradient Boosting"),
        "features": ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase",
                     "Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens",
                     "Albumin","Albumin_and_Globulin_Ratio"],
        "feature_defaults": [45, 1, 1.2, 0.4, 187, 35, 40, 6.5, 3.1, 0.9],
        "feature_mins":     [4,  0, 0.4, 0.1,  63,  10, 10, 2.7, 0.9, 0.3],
        "feature_maxs":     [90, 1,75.0,19.7,2110,2000,4929,9.6, 5.5, 2.8],
        "feature_steps":    [1, 1, 0.1, 0.1,   1,   1,   1, 0.1, 0.1, 0.01],
        "feature_labels": {
            "Age": "Patient Age",
            "Gender": "Gender (1=Male, 0=Female)",
            "Total_Bilirubin": "Total Bilirubin (mg/dL)",
            "Direct_Bilirubin": "Direct Bilirubin (mg/dL)",
            "Alkaline_Phosphotase": "Alkaline Phosphatase (IU/L)",
            "Alamine_Aminotransferase": "ALT / SGPT (IU/L)",
            "Aspartate_Aminotransferase": "AST / SGOT (IU/L)",
            "Total_Protiens": "Total Proteins (g/dL)",
            "Albumin": "Albumin (g/dL)",
            "Albumin_and_Globulin_Ratio": "A/G Ratio",
        },
        "csv_example": "Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,PatientName\n45,1,1.2,0.4,187,35,40,6.5,3.1,0.9,John Doe",
    },
}


# ─────────────────────────────────────────────
# Helper: load models
# ─────────────────────────────────────────────

@st.cache_resource
def load_models(disease: str):
    cfg = DISEASE_CONFIG[disease]
    m1_path = os.path.join(MODELS_DIR, cfg["model_files"][0])
    m2_path = os.path.join(MODELS_DIR, cfg["model_files"][1])
    if not os.path.exists(m1_path) or not os.path.exists(m2_path):
        return None, None, None
    with open(m1_path, "rb") as f:
        model1, scaler = pickle.load(f)
    with open(m2_path, "rb") as f:
        model2, _ = pickle.load(f)
    return model1, model2, scaler


def run_prediction(patient_row: dict, disease: str):
    """Returns (prob_primary, prob_baseline) as floats in [0, 1]."""
    cfg = DISEASE_CONFIG[disease]
    model1, model2, scaler = load_models(disease)
    if model1 is None:
        return None, None
    df = pd.DataFrame([patient_row])[cfg["features"]]
    scaled = scaler.transform(df)
    p1 = model1.predict_proba(scaled)[0][1]
    p2 = model2.predict_proba(scaled)[0][1]
    return float(p1), float(p2)


def risk_badge(prob: float) -> str:
    """Returns HTML badge string for a given probability."""
    pct = int(prob * 100)
    if prob < 0.3:
        bg, txt = "#14532D", "#4ADE80"
        label = "LOW RISK"
    elif prob < 0.6:
        bg, txt = "#78350F", "#FCD34D"
        label = "MODERATE"
    else:
        bg, txt = "#7F1D1D", "#F87171"
        label = "HIGH RISK"
    return (
        f"<span style='background:{bg}; color:{txt}; padding:3px 10px; "
        f"border-radius:20px; font-weight:800; font-size:0.82rem; "
        f"letter-spacing:0.04em;'>{pct}% — {label}</span>"
    )


def gauge_chart(pct: int, label: str, color: str):
    """Compact Plotly gauge for a single model prediction."""
    if pct < 30:
        bar_color = "#10B981"
    elif pct < 60:
        bar_color = "#F59E0B"
    else:
        bar_color = "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC"}},
        title={"text": label, "font": {"size": 13, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"size": 10, "color": "#64748B"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(15,23,42,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "rgba(16,185,129,0.1)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.1)"},
                {"range": [60, 100],"color": "rgba(239,68,68,0.1)"},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────

st.markdown("""
    <style>
        @keyframes titleSlide {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .page-header {
            animation: titleSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
    </style>
    <div class="page-header">
        <h1 style="color:#00BCD4; font-weight: 900;
               font-size:2.8rem; letter-spacing:-0.02em;">
            📋 Patient Diagnostic Report Portal
        </h1>
        <p style="color:#94A3B8; font-size:1.15rem; margin-bottom:30px; line-height:1.6;">
            Enter patient vitals manually or upload a CSV/Excel file for batch screening.
            Our trained ML models will compute disease risk in percentage with colour-coded status indicators
            and generate a complete, downloadable clinical health report for each patient.
        </p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIGURATION ROW
# ─────────────────────────────────────────────

cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 2, 1])

with cfg_col1:
    disease = st.selectbox(
        "🩺 Select Disease to Screen",
        options=list(DISEASE_CONFIG.keys()),
        format_func=lambda d: f"{DISEASE_CONFIG[d]['icon']}  {d}",
    )

with cfg_col2:
    mode = st.radio(
        "📥 Input Mode",
        options=["✍️ Manual Entry", "📂 Upload CSV / Excel"],
        horizontal=True,
    )

with cfg_col3:
    accent = DISEASE_CONFIG[disease]["color"]
    st.markdown(
        f"""<div style="background:rgba(255,255,255,0.03); border:1px solid {accent}40;
            border-left:5px solid {accent}; border-radius:10px; padding:14px 16px; margin-top:4px;">
            <div style="font-size:0.8rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">Models Active</div>
            <div style="font-size:0.95rem; color:#F8FAFC; font-weight:700; margin-top:4px;">
                {DISEASE_CONFIG[disease]['model_names'][0]}<br/>
                {DISEASE_CONFIG[disease]['model_names'][1]}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

cfg = DISEASE_CONFIG[disease]

st.markdown("---")

# ─────────────────────────────────────────────
# MANUAL ENTRY MODE
# ─────────────────────────────────────────────

if "✍️ Manual Entry" in mode:

    st.markdown(
        f'<div class="form-card"><h3 class="section-header" style="margin-top:0;">'
        f'{cfg["icon"]} Patient Vitals Entry</h3>',
        unsafe_allow_html=True,
    )

    with st.form("manual_form"):
        # Patient metadata
        meta_c1, meta_c2 = st.columns(2)
        with meta_c1:
            patient_name = st.text_input("👤 Patient Full Name", value="Patient Name")
        with meta_c2:
            patient_id = st.text_input("🆔 Patient ID / Reference", value="PT-001")

        st.markdown("---")
        st.markdown(
            '<p style="color:#94A3B8; font-size:0.95rem; margin-bottom:16px;">'
            'Fill in the clinical indicators below. Hover over each field for tooltip guidance.</p>',
            unsafe_allow_html=True,
        )

        # Dynamically render input fields in a 2-column grid
        features = cfg["features"]
        labels   = cfg["feature_labels"]
        defaults = cfg["feature_defaults"]
        mins     = cfg["feature_mins"]
        maxs     = cfg["feature_maxs"]
        steps    = cfg["feature_steps"]

        patient_values = {}
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                step = steps[i]
                if step == 1 and isinstance(defaults[i], int):
                    patient_values[feat] = st.number_input(
                        labels.get(feat, feat),
                        min_value=int(mins[i]),
                        max_value=int(maxs[i]),
                        value=int(defaults[i]),
                        step=1,
                    )
                else:
                    patient_values[feat] = st.number_input(
                        labels.get(feat, feat),
                        min_value=float(mins[i]),
                        max_value=float(maxs[i]),
                        value=float(defaults[i]),
                        step=float(step),
                        format="%.3f" if step < 0.01 else "%.2f",
                    )

        submitted = st.form_submit_button(
            f"🔬 Run {disease} Diagnostic",
            use_container_width=True,
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        with st.spinner("⚙️ Running models..."):
            p1, p2 = run_prediction(patient_values, disease)

        if p1 is None:
            st.error("❌ Models not found. Please ensure the training script has been run.")
        else:
            st.session_state["portal_result"] = {
                "patient_name": patient_name,
                "patient_id": patient_id,
                "disease": disease,
                "patient_data": patient_values,
                "prob_primary": p1,
                "prob_baseline": p2,
                "model_names": cfg["model_names"],
            }

# ─────────────────────────────────────────────
# CSV / EXCEL UPLOAD MODE
# ─────────────────────────────────────────────

elif "📂 Upload" in mode:

    st.markdown(
        f'<div class="form-card"><h3 class="section-header" style="margin-top:0;">'
        f'📂 Batch Patient Data Upload</h3>',
        unsafe_allow_html=True,
    )

    # Show expected columns
    with st.expander("📌 Required CSV / Excel Column Format", expanded=False):
        st.markdown(
            f"Your file must contain these columns (case-insensitive, spaces allowed):<br/>"
            f"<code>{'  |  '.join(cfg['features'] + ['PatientName'])}</code><br/><br/>"
            f"<b>Example CSV structure:</b>",
            unsafe_allow_html=True,
        )
        st.code(cfg["csv_example"], language="csv")

    uploaded_file = st.file_uploader(
        "📁 Upload Patient Data File (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="File must include all required columns for the selected disease.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            # Normalize column names (strip, lower for matching)
            df_upload.columns = df_upload.columns.str.strip()
            feat_lower_map = {f.lower().replace(" ", "_"): f for f in cfg["features"]}
            col_rename = {}
            for col in df_upload.columns:
                normalized = col.lower().replace(" ", "_")
                if normalized in feat_lower_map:
                    col_rename[col] = feat_lower_map[normalized]
            df_upload = df_upload.rename(columns=col_rename)

            # Get patient name column
            name_col = None
            for c in df_upload.columns:
                if c.lower().strip() in ["patientname", "patient_name", "name", "patient name"]:
                    name_col = c
                    break

            missing = [f for f in cfg["features"] if f not in df_upload.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                st.success(f"✅ File loaded: **{len(df_upload)} patient records** detected.")
                st.dataframe(df_upload.head(10), use_container_width=True)

                if st.button("🔬 Run Batch Diagnostics", use_container_width=True):
                    model1, model2, scaler = load_models(disease)
                    if model1 is None:
                        st.error("❌ Models not found.")
                    else:
                        results = []
                        for idx, row in df_upload.iterrows():
                            pat_data = {f: row[f] for f in cfg["features"]}
                            df_feat = pd.DataFrame([pat_data])[cfg["features"]]
                            scaled = scaler.transform(df_feat)
                            p1 = float(model1.predict_proba(scaled)[0][1])
                            p2 = float(model2.predict_proba(scaled)[0][1])
                            avg = (p1 + p2) / 2
                            pname = str(row[name_col]) if name_col else f"Patient {idx+1}"
                            results.append({
                                "patient_name": pname,
                                "patient_data": pat_data,
                                "prob_primary": p1,
                                "prob_baseline": p2,
                                "avg_prob": avg,
                            })
                        st.session_state["batch_results"] = {
                            "disease": disease,
                            "results": results,
                            "model_names": cfg["model_names"],
                        }

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Render batch results table
    if "batch_results" in st.session_state:
        br = st.session_state["batch_results"]
        if br["disease"] == disease:
            st.markdown("---")
            st.markdown(
                '<h3 style="color:#F8FAFC; font-weight:700;">📊 Batch Diagnostic Results</h3>',
                unsafe_allow_html=True,
            )

            m1n, m2n = br["model_names"]
            for i, res in enumerate(br["results"]):
                p1  = res["prob_primary"]
                p2  = res["prob_baseline"]
                avg = res["avg_prob"]
                pname = res["patient_name"]

                row_col1, row_col2, row_col3, row_col4 = st.columns([2.5, 1.5, 1.5, 1.5])
                with row_col1:
                    st.markdown(
                        f'<div style="padding:12px 0;">'
                        f'<span style="font-weight:700; font-size:1rem; color:#F8FAFC;">{pname}</span><br/>'
                        f'<span style="font-size:0.8rem; color:#64748B;">Record #{i+1}</span></div>',
                        unsafe_allow_html=True,
                    )
                with row_col2:
                    st.markdown(
                        f'<div style="padding:12px 0; font-size:0.9rem;">'
                        f'<span style="color:#94A3B8;">{m1n}:</span> '
                        f'<b style="color:#F8FAFC;">{int(p1*100)}%</b></div>',
                        unsafe_allow_html=True,
                    )
                with row_col3:
                    st.markdown(
                        f'<div style="padding:12px 0; font-size:0.9rem;">'
                        f'<span style="color:#94A3B8;">{m2n}:</span> '
                        f'<b style="color:#F8FAFC;">{int(p2*100)}%</b></div>',
                        unsafe_allow_html=True,
                    )
                with row_col4:
                    st.markdown(
                        f'<div style="padding:10px 0;">{risk_badge(avg)}</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("📄 Generate Report", key=f"gen_{i}"):
                        st.session_state["portal_result"] = {
                            "patient_name": pname,
                            "patient_id": f"PT-{i+1:03d}",
                            "disease": disease,
                            "patient_data": res["patient_data"],
                            "prob_primary": p1,
                            "prob_baseline": p2,
                            "model_names": br["model_names"],
                        }

                st.markdown(
                    '<hr style="border:none; border-top:1px solid rgba(255,255,255,0.05); margin:0;">',
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────
# RESULTS PANEL  (manual OR selected batch)
# ─────────────────────────────────────────────

if "portal_result" in st.session_state:
    res = st.session_state["portal_result"]

    # Only show if disease matches current selection
    if res["disease"] == disease:
        p1  = res["prob_primary"]
        p2  = res["prob_baseline"]
        avg = (p1 + p2) / 2
        m1n, m2n = res["model_names"]
        pname  = res["patient_name"]
        pid    = res["patient_id"]
        p_data = res["patient_data"]

        st.markdown("---")
        st.markdown(
            f'<div class="result-card"><h2 class="section-header" style="color:{cfg["color"]}; margin-top:0;">'
            f'🧾 Diagnostic Results — {pname}</h2>',
            unsafe_allow_html=True,
        )

        # ── Risk gauges ──────────────────────────────────────────────────────
        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(gauge_chart(int(p1*100), m1n, cfg["color"]), use_container_width=True)
        with g2:
            st.plotly_chart(gauge_chart(int(p2*100), m2n, cfg["color"]), use_container_width=True)
        with g3:
            st.markdown('<div class="glass-card" style="text-align:center; padding:24px 16px;">', unsafe_allow_html=True)
            avg_pct = int(avg * 100)
            if avg < 0.3:
                rc, rl, ri = "#10B981", "LOW RISK", "🟢"
            elif avg < 0.6:
                rc, rl, ri = "#F59E0B", "MODERATE RISK", "🟡"
            else:
                rc, rl, ri = "#EF4444", "HIGH RISK", "🔴"
            st.markdown(
                f'<div style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; '
                f'letter-spacing:0.08em; font-weight:700; margin-bottom:8px;">Combined Assessment</div>'
                f'<div style="font-size:3.2rem; font-weight:900; color:{rc}; line-height:1;">{avg_pct}%</div>'
                f'<div style="font-size:1rem; font-weight:800; color:{rc}; margin-top:8px;">{ri} {rl}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Abnormal vitals highlight ────────────────────────────────────────
        from patient_report_generator import _check_abnormal_features, PROGNOSIS_TEXT, MEDICATION_SUGGESTIONS
        abnormal = _check_abnormal_features(p_data, disease)

        if abnormal:
            st.markdown(
                '<h3 style="color:#F87171; font-weight:700; margin-top:10px;">⚠️ Out-of-Range Indicators</h3>',
                unsafe_allow_html=True,
            )
            a_cols = st.columns(min(len(abnormal), 3))
            for i, (label, val, unit, lo, hi, direction) in enumerate(abnormal):
                with a_cols[i % 3]:
                    dir_color = "#EF4444" if direction == "HIGH" else "#F59E0B"
                    dir_arrow = "⬆" if direction == "HIGH" else "⬇"
                    st.markdown(
                        f'<div class="glass-card" style="border-left:4px solid {dir_color}; padding:14px;">'
                        f'<div style="font-size:0.78rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">{label}</div>'
                        f'<div style="font-size:1.6rem; font-weight:900; color:{dir_color};">{val} <span style="font-size:0.9rem;">{unit}</span></div>'
                        f'<div style="font-size:0.8rem; color:#64748B;">{dir_arrow} {direction} &nbsp;|&nbsp; Normal: {lo}–{hi}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.success("✅ All entered values are within healthy reference ranges.")

        # ── Prognosis ────────────────────────────────────────────────────────
        prog = PROGNOSIS_TEXT.get(disease, {})
        st.markdown(
            '<div class="glass-card" style="margin-top:12px;">'
            f'<h3 style="margin-top:0; color:{cfg["color"]}; font-weight:700;">🔬 Why This Risk Occurred</h3>'
            f'<p style="color:#CBD5E1; line-height:1.75; font-size:0.97rem;">{prog.get("causes", "")}</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Untreated outcome
        if avg >= 0.3:
            urgency_bg  = "#FEF2F2" if avg >= 0.6 else "#FFFBEB"
            urgency_clr = "#7F1D1D" if avg >= 0.6 else "#78350F"
            st.markdown(
                f'<div style="background:{urgency_bg}; border:2px solid {urgency_clr}; '
                f'border-radius:12px; padding:16px 20px; margin-top:12px;">'
                f'<h4 style="color:{urgency_clr}; margin-top:0; font-weight:800;">⚡ If Left Untreated</h4>'
                f'<p style="color:{urgency_clr}; line-height:1.7; font-size:0.95rem; margin:0;">'
                f'{prog.get("if_untreated", "")}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Care plan
        st.markdown(
            f'<div class="glass-card" style="margin-top:14px;">'
            f'<h3 style="margin-top:0; color:#34D399; font-weight:700;">✅ Personalized Care Plan</h3>',
            unsafe_allow_html=True,
        )
        for item in prog.get("care_plan", []):
            st.markdown(
                f'<div style="padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.04); '
                f'color:#E2E8F0; font-size:0.95rem; line-height:1.65;">{item}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Suggested medications (informational only)
        meds = MEDICATION_SUGGESTIONS.get(disease, [])
        if meds:
            st.markdown(
                f'<div class="glass-card" style="margin-top:12px;">'
                f'<h3 style="margin-top:0; color:#F97316; font-weight:700;">💊 Suggested Medications (Informational)</h3>'
                f'<p style="color:#FCA5A5; font-size:0.9rem; margin-top:0; font-weight:700;">'
                'Medication suggestions are for informational purposes only. Do NOT start, stop, or change any medications without consulting a licensed healthcare provider.'
                f'</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for med in meds:
                st.markdown(f'- {med}', unsafe_allow_html=True)

        # ── PDF DOWNLOAD ─────────────────────────────────────────────────────
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            '<div class="glass-card" style="text-align:center; border:2px solid rgba(96,165,250,0.3);">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h3 style="color:#60A5FA; margin-top:0; font-weight:700;">📄 Download Clinical PDF Report</h3>'
            '<p style="color:#94A3B8; font-size:0.95rem;">Generate a professional, printable patient health report '
            'including all vitals, abnormal indicators, disease prognosis, and personalised care plan.</p>',
            unsafe_allow_html=True,
        )

        with st.spinner("📝 Compiling patient PDF report..."):
            pdf_bytes = generate_patient_report(
                patient_name=pname,
                patient_id=pid,
                disease=disease,
                patient_data=p_data,
                prob_primary=p1,
                prob_baseline=p2,
                model_primary_name=m1n,
                model_baseline_name=m2n,
            )

        safe_name = pname.replace(" ", "_").replace("/", "-")
        st.download_button(
            label="📥  Download Patient Report PDF",
            data=pdf_bytes,
            file_name=f"ChroniCare_{safe_name}_{disease.replace(' ', '_')}_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
