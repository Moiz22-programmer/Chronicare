import streamlit as st
import pandas as pd
import numpy as np
from styles import apply_custom_styles
from utils import load_ml_model, save_prediction, render_history_section
from patient_report_generator import generate_patient_report

# Apply custom styles
apply_custom_styles()

st.markdown("""
    <style>
    .page-header { animation: titleSlide 0.8s cubic-bezier(0.16, 1, 0.3, 1) both; }
    </style>
    <div class="page-header" style="margin-bottom: 32px;">
        <h1 style="color:#EF4444; font-weight: 900; margin-bottom: 8px; font-size:3rem; letter-spacing:-0.02em;">❤️ Cardiovascular Risk Assessment</h1>
        <p style="color:#94A3B8; font-size:1.15rem; margin:0; line-height:1.7; font-weight:400;">Advanced screening for coronary artery disease using blood pressure, cholesterol, and cardiac parameters.</p>
    </div>
""", unsafe_allow_html=True)

# Load primary and baseline models
model, scaler = load_ml_model("heart_svm.pkl")
model_base, _ = load_ml_model("heart_dt.pkl")

if model is None or scaler is None:
    st.warning("Model not loaded. Please add the trained model file.")
    st.stop()

# Layout: Form and File Upload
st.markdown('<div class="form-card"><h3 class="section-header" style="margin-top:0;">👤 Patient Clinical Data Input</h3><p class="section-subtitle">Enter patient vitals and cardiac parameters below:</p>', unsafe_allow_html=True)

# Two-column layout for the manual input form
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name", value="Jane Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex_str = st.selectbox("Sex", options=["Female", "Male"])
    sex = 1 if sex_str == "Male" else 0
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    # BMI Auto-Calculation
    bmi = 0.0
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**Computed Body Mass Index (BMI):** <span style='color:#00BCD4; font-weight:700;'>{bmi:.1f}</span>", unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours per day", min_value=1.0, max_value=24.0, value=7.5, step=0.5)
    activity = st.selectbox("Physical Activity Level", options=["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"])

with col2:
    cp_desc = st.selectbox(
        "Chest Pain Type",
        options=[
            "Asymptomatic (Silent chest issues)",
            "Typical Angina (Squeezing/burning pressure)",
            "Atypical Angina (Brief/sharp pain)",
            "Non-Anginal (Sharp/unrelated chest irritation)"
        ]
    )
    # Map to chest pain type indices used in model (Asymptomatic=0, Typical=1, Atypical=2, Non-Anginal=3)
    cp = {
        "Asymptomatic (Silent chest issues)": 0,
        "Typical Angina (Squeezing/burning pressure)": 1,
        "Atypical Angina (Brief/sharp pain)": 2,
        "Non-Anginal (Sharp/unrelated chest irritation)": 3
    }[cp_desc]
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol Level (mg/dL)", min_value=50, max_value=600, value=200)
    fbs_str = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", options=["No", "Yes"])
    fbs = 1 if fbs_str == "Yes" else 0
    
    restecg_desc = st.selectbox(
        "Resting ECG Results",
        options=[
            "Normal",
            "ST-T Wave Abnormality (T wave inversions / ST elevation)",
            "Left Ventricular Hypertrophy (LVH - voltage criteria)"
        ]
    )
    restecg = {
        "Normal": 0,
        "ST-T Wave Abnormality (T wave inversions / ST elevation)": 1,
        "Left Ventricular Hypertrophy (LVH - voltage criteria)": 2
    }[restecg_desc]
    
    thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang_str = st.selectbox("Exercise-Induced Angina?", options=["No", "Yes"])
    exang = 1 if exang_str == "Yes" else 0
    
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ca = st.slider("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0)

st.markdown('</div>', unsafe_allow_html=True)

# Evaluation logic
analyze_clicked = st.button("🔬 Analyze Patient Vitals", use_container_width=True)

if analyze_clicked:
    # Prepare input for prediction (model expects 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    # Defaulting slope = 1 and thal = 2
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': 1,
        'ca': ca,
        'thal': 2
    }])
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prob_base = model_base.predict_proba(input_scaled)[0][1] if model_base is not None else prob
    
    prediction = "⚠️ Disease Detected" if prob > 0.5 else "✅ No Disease Detected"
    
    # Save to history
    save_prediction(name, age, sex_str, "Heart Disease", prediction)
    
    # Display Result
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header" style="margin-top:0;">📊 Diagnostic Assessment Results</h3>', unsafe_allow_html=True)
    
    if prediction == "⚠️ Disease Detected":
        st.markdown(f'<h4 style="color:#EF4444 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        
        # Why this happened explanation
        reasons = []
        if chol > 200:
            reasons.append(f"your cholesterol level of {chol} mg/dL is higher than the recommended limit of 200 mg/dL, which can lead to fat building up in your arteries")
        if trestbps > 120:
            reasons.append(f"your resting blood pressure of {trestbps} mm Hg is elevated, which forces your heart to work much harder to pump blood")
        if oldpeak > 1.0:
            reasons.append(f"your exercise test showed a significant ST depression value of {oldpeak}, indicating potential blood flow reduction to your heart muscle during activity")
        if cp > 0:
            reasons.append("you reported active chest discomfort or pressure, which is a classic warning sign of underlying arterial narrowing")
            
        if not reasons:
            reasons.append("the model identified elevated cardiovascular stress based on the combined profile of your blood pressure, ECG, and vessel counts")
            
        why_text = f"This warning is displayed because " + ", and ".join(reasons[:2]) + ". These indicators suggest that your cardiovascular system may be facing extra strain."
        st.markdown(f"<p style='color:#CBD5E1; font-size:1.05rem; line-height:1.6; margin-bottom:20px;'><strong>Why this happened:</strong> {why_text}</p>", unsafe_allow_html=True)
        
        # Recovery Plan Tabs
        tab_food, tab_meds, tab_exercises = st.tabs(["🥗 Foods to Eat", "💊 Medicines / Supplements", "🏃 Exercises"])
        with tab_food:
            st.markdown("""
            - **Oats & Whole Grains:** Rich in soluble fiber, which helps lower LDL cholesterol levels.
            - **Salmon & Fatty Fish:** Packed with Omega-3 fatty acids, which reduce inflammation and lower heart disease risk.
            - **Blueberries & Strawberries:** High in antioxidants that protect blood vessels and reduce vascular oxidative stress.
            - **Extra Virgin Olive Oil:** Packed with monounsaturated fats that support healthy cholesterol balances.
            - **Leafy Green Vegetables:** High in nitrates and vitamin K, which help dilate arteries and improve vascular tone.
            - **Walnuts & Almonds:** Provide healthy fats, fiber, and plant sterols that actively protect heart tissue.
            - **Beans & Lentils:** High in fiber and magnesium, supporting blood pressure regulation.
            - **Avocados:** Excellent source of heart-healthy monounsaturated fats and potassium.
            """)
        with tab_meds:
            st.info("⚠️ **Disclaimer:** Consult a qualified medical practitioner or cardiologist before starting any medication or supplement regimen.")
            st.markdown("""
            - **Statins:** Common prescription medications used to lower blood LDL cholesterol levels and stabilize arterial plaque.
            - **Aspirin:** Low-dose aspirin is sometimes prescribed to reduce the risk of blood clots.
            - **Beta-Blockers:** Help reduce blood pressure and slow down heart rate, relieving workload on heart muscles.
            - **Coenzyme Q10 (CoQ10):** A natural cellular supplement that supports mitochondrial energy in cardiac cells.
            - **Omega-3 Supplements (Fish Oil):** High-quality capsules containing EPA/DHA to support healthy triglyceride levels.
            """)
        with tab_exercises:
            st.markdown("""
            - **Brisk Walking:** Low-impact, highly effective cardio that strengthens the heart muscle and regulates vascular pressure.
            - **Stationary Cycling:** Safe, controlled aerobic workout that improves cardiovascular endurance without straining joints.
            - **Water Aerobics or Light Swimming:** Excellent full-body conditioning that improves circulation and reduces arterial stiffness.
            - **Gentle Yoga & Stretching:** Reduces stress, lowers heart rate, and aids autonomic nervous system recovery.
            - **Light Bodyweight Squats:** Low-intensity resistance training that helps clear glucose and improves general circulatory return.
            """)
    else:
        st.markdown(f'<h4 style="color:#00BCD4 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        st.success("Your cardiovascular metrics and resting parameters indicate low risk of coronary artery disease. Keep maintaining your active lifestyle and check back regularly!")

    # PDF Report Generator Button
    try:
        pdf_data = {
            "age": age,
            "sex": sex_str,
            "cp": cp_desc,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs_str,
            "restecg": restecg_desc,
            "thalach": thalach,
            "exang": exang_str,
            "oldpeak": oldpeak,
            "ca": ca
        }
        p_id = f"CC-{np.random.randint(1000, 9999)}"
        pdf_bytes = generate_patient_report(
            patient_name=name,
            patient_id=p_id,
            disease="Heart Disease",
            patient_data=pdf_data,
            prob_primary=prob,
            prob_baseline=prob_base,
            model_primary_name="Support Vector Machine",
            model_baseline_name="Decision Tree"
        )
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Complete Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"ChroniCare_Heart_Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate report PDF: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-top:0; color:#00BCD4; font-weight:700;">📂 Section B — CSV / Excel File Upload</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV or Excel file for batch cardiovascular risk screening", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Batch Predict helper
        prepared_df = pd.DataFrame(index=df.index)
        
        # Helper to find column case-insensitively
        def get_col(names, default_val):
            for name in names:
                for col in df.columns:
                    if col.strip().lower() == name.lower():
                        return df[col]
            return pd.Series(default_val, index=df.index)
            
        prepared_df['age'] = get_col(['age', 'Age'], 50)
        
        # Map sex
        raw_sex = get_col(['sex', 'Sex', 'gender', 'Gender'], 'Male')
        prepared_df['sex'] = raw_sex.apply(lambda x: 1 if str(x).strip().lower().startswith('m') or str(x) == '1' else 0)
        
        prepared_df['cp'] = get_col(['cp', 'chest pain type', 'Chest Pain Type', 'Chest Pain'], 0).astype(int)
        prepared_df['trestbps'] = get_col(['trestbps', 'resting blood pressure', 'blood pressure', 'Resting BP', 'Resting Blood Pressure'], 120)
        prepared_df['chol'] = get_col(['chol', 'cholesterol', 'Cholesterol Level', 'Cholesterol'], 200)
        
        # Map fbs
        raw_fbs = get_col(['fbs', 'fasting blood sugar', 'Fasting Blood Sugar > 120 mg/dl', 'Fasting Blood Sugar'], 0)
        prepared_df['fbs'] = raw_fbs.apply(lambda x: 1 if str(x).strip().lower().startswith('y') or str(x) == '1' or str(x) == 'true' else 0)
        
        prepared_df['restecg'] = get_col(['restecg', 'resting ecg results', 'Resting ECG'], 0).astype(int)
        prepared_df['thalach'] = get_col(['thalach', 'max heart rate', 'max heart rate achieved', 'Max HR'], 150)
        
        # Map exang
        raw_exang = get_col(['exang', 'exercise-induced angina', 'exercise induced angina', 'Angina'], 0)
        prepared_df['exang'] = raw_exang.apply(lambda x: 1 if str(x).strip().lower().startswith('y') or str(x) == '1' or str(x) == 'true' else 0)
        
        prepared_df['oldpeak'] = get_col(['oldpeak', 'st depression', 'ST depression'], 1.0)
        prepared_df['slope'] = get_col(['slope', 'slope of peak exercise'], 1).astype(int)
        prepared_df['ca'] = get_col(['ca', 'number of major vessels', 'vessels', 'Major Vessels'], 0).astype(int)
        prepared_df['thal'] = get_col(['thal', 'thalassemia', 'Thalassemia'], 2).astype(int)
        
        # Scale and predict
        input_scaled = scaler.transform(prepared_df)
        probs = model.predict_proba(input_scaled)[:, 1]
        
        predictions = ["⚠️ Disease Detected" if p > 0.5 else "✅ No Disease Detected" for p in probs]
        
        # Add prediction column
        df['Prediction'] = predictions
        
        # Show in table
        st.write("### Prediction Results")
        st.dataframe(df, use_container_width=True)
        
        # Save batch entries to history
        names_col = get_col(['name', 'Patient Name', 'PatientName'], 'Unknown Patient')
        ages_col = prepared_df['age']
        sexes_col = get_col(['sex', 'Sex', 'gender', 'Gender'], 'Female')
        
        for idx in df.index:
            p_name = names_col.loc[idx]
            p_age = ages_col.loc[idx]
            p_sex = str(sexes_col.loc[idx])
            p_pred = predictions[idx]
            save_prediction(p_name, p_age, p_sex, "Heart Disease", p_pred)
            
        st.success("Batch predictions processed and logged to patient history successfully!")
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Persistent History Section
render_history_section()
