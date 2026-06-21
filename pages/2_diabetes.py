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
        <h1 style="color:#00BCD4; font-weight: 900; margin-bottom: 8px; font-size:3rem; letter-spacing:-0.02em;">🩸 Glycemic & Diabetes Screening</h1>
        <p style="color:#94A3B8; font-size:1.15rem; margin:0; line-height:1.7; font-weight:400;">Comprehensive assessment of blood glucose metabolism and diabetic risk indicators using advanced ML diagnostics.</p>
    </div>
""", unsafe_allow_html=True)

# Load primary and baseline models
model, scaler = load_ml_model("diabetes_rf.pkl")
model_base, _ = load_ml_model("diabetes_lr.pkl")

if model is None or scaler is None:
    st.warning("Model not loaded. Please add the trained model file.")
    st.stop()

# Layout: Form and File Upload
st.markdown('<div class="form-card"><h3 class="section-header" style="margin-top:0;">👤 Patient Clinical Data Input</h3><p class="section-subtitle">Enter patient vitals and health metrics below:</p>', unsafe_allow_html=True)

# Two-column layout for the manual input form
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name", value="John Doe")
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    sex_str = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_str == "Male" else 0
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=75.0)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=175.0)
    
    # BMI Auto-Calculation
    bmi = 0.0
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**Computed Body Mass Index (BMI):** <span style='color:#00BCD4; font-weight:700;'>{bmi:.1f}</span>", unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours per day", min_value=1.0, max_value=24.0, value=8.0, step=0.5)
    activity = st.selectbox("Physical Activity Level", options=["Medium", "Low", "High"])
    smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"])

with col2:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=10, max_value=500, value=110, help="2-hour oral glucose tolerance test")
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=10, max_value=250, value=70)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
    pedigree = st.number_input("Diabetes Pedigree Function (Family History)", min_value=0.0, max_value=5.0, value=0.45, step=0.01)
    
    # Pregnancies is only active if Female (otherwise 0)
    if sex_str == "Female":
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    else:
        pregnancies = 0
        st.markdown("<p style='color:#64748B; font-size:0.9rem; padding: 10px 0;'>Pregnancies field is automatically set to 0 for Male patient.</p>", unsafe_allow_html=True)
        
    skin = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

st.markdown('</div>', unsafe_allow_html=True)

# Evaluation logic
analyze_clicked = st.button("🔬 Analyze Patient Vitals", use_container_width=True)

if analyze_clicked:
    # Prepare input (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    input_df = pd.DataFrame([{
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': pedigree,
        'Age': age
    }])
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prob_base = model_base.predict_proba(input_scaled)[0][1] if model_base is not None else prob
    
    prediction = "⚠️ Disease Detected" if prob > 0.5 else "✅ No Disease Detected"
    
    # Save to history
    save_prediction(name, age, sex_str, "Diabetes", prediction)
    
    # Display Result
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header" style="margin-top:0;">📊 Diagnostic Assessment Results</h3>', unsafe_allow_html=True)
    
    if prediction == "⚠️ Disease Detected":
        st.markdown(f'<h4 style="color:#EF4444 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        
        # Why this happened explanation
        reasons = []
        if glucose > 100:
            reasons.append(f"your glucose level of {glucose} mg/dL is elevated above the normal range (under 100 mg/dL), which indicates that your body is experiencing insulin resistance")
        if bmi > 25.0:
            reasons.append(f"your calculated BMI of {bmi:.1f} is in the overweight/obese range, which is a major factor that blocks normal insulin response")
        if pedigree > 0.6:
            reasons.append(f"your family history pedigree score of {pedigree:.2f} shows a higher genetic risk of developing diabetes")
        if insulin > 150:
            reasons.append(f"your insulin level of {insulin} mu U/ml is abnormally high, showing your pancreas is working overtime to reduce glucose")
            
        if not reasons:
            reasons.append("the model identified high metabolic and diabetic risks based on your combined age, BMI, and glucose profile")
            
        why_text = f"This warning is displayed because " + ", and ".join(reasons[:2]) + ". These signs point to blood sugar regulation difficulties."
        st.markdown(f"<p style='color:#CBD5E1; font-size:1.05rem; line-height:1.6; margin-bottom:20px;'><strong>Why this happened:</strong> {why_text}</p>", unsafe_allow_html=True)
        
        # Recovery Plan Tabs
        tab_food, tab_meds, tab_exercises = st.tabs(["🥗 Foods to Eat", "💊 Medicines / Supplements", "🏃 Exercises"])
        with tab_food:
            st.markdown("""
            - **Non-Starchy Vegetables:** Broccoli, spinach, and kale are low in calories and carbohydrates, preventing blood sugar spikes.
            - **Avocados:** Rich in healthy monounsaturated fats that support insulin sensitivity.
            - **Eggs:** High in protein and healthy fats, which keep you full and stabilize morning glucose curves.
            - **Chia Seeds & Flaxseeds:** Packed with viscous fiber that slows down digestion and glucose absorption.
            - **Beans & Lentils:** High-fiber legumes that have a very low glycemic index, providing steady energy.
            - **Greek Yogurt:** High in protein and calcium, helping control appetite and glucose curves.
            - **Barley & Oats:** Contain beta-glucans which actively improve insulin response.
            - **Nuts (Walnuts/Almonds):** High in fiber and low in digestible carbs to prevent glycemic spikes.
            """)
        with tab_meds:
            st.info("⚠️ **Disclaimer:** Consult a qualified medical practitioner or endocrinologist before starting any medication or supplement regimen.")
            st.markdown("""
            - **Metformin:** The primary prescription medication to improve insulin sensitivity and reduce liver glucose production.
            - **Sulfonylureas:** Medications that stimulate pancreatic beta cells to produce more insulin.
            - **Insulin Therapy:** Direct insulin injections to manage blood glucose levels (commonly for Type 1 or advanced Type 2).
            - **Chromium Supplements:** A trace mineral that is believed to assist insulin action and glucose metabolism.
            - **Alpha-Lipoic Acid (ALA):** A powerful antioxidant that may improve insulin sensitivity and reduce diabetic nerve discomfort.
            """)
        with tab_exercises:
            st.markdown("""
            - **Brisk Walking:** A simple 30-minute walk daily improves glucose uptake in muscles and lowers blood sugar.
            - **Strength Training:** Building skeletal muscle increases glycogen storage capacity, improving long-term HbA1c.
            - **Stationary Cycling:** High-energy aerobic workout that burns glucose and enhances cardiovascular fitness.
            - **Swimming:** Full-body aerobic exercise that puts minimal stress on joints and actively burns calories.
            - **Yoga & Pilates:** Helps reduce cortisol levels, which in turn reduces stress-induced glucose spikes.
            """)
    else:
        st.markdown(f'<h4 style="color:#00BCD4 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        st.success("Your blood sugar markers and vital indicators point to a healthy, stable metabolism. Keep maintaining your diet and exercise habits!")

    # PDF Report Generator Button
    try:
        pdf_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }
        p_id = f"CC-{np.random.randint(1000, 9999)}"
        pdf_bytes = generate_patient_report(
            patient_name=name,
            patient_id=p_id,
            disease="Diabetes",
            patient_data=pdf_data,
            prob_primary=prob,
            prob_baseline=prob_base,
            model_primary_name="Random Forest",
            model_baseline_name="Logistic Regression"
        )
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Complete Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"ChroniCare_Diabetes_Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate report PDF: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-top:0; color:#00BCD4; font-weight:700;">📂 Section B — CSV / Excel File Upload</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV or Excel file for batch glycemic/diabetes screening", type=["csv", "xlsx"])

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
            
        prepared_df['Pregnancies'] = get_col(['Pregnancies', 'pregnancies', 'PregnancyCount'], 0).astype(int)
        prepared_df['Glucose'] = get_col(['Glucose', 'glucose', 'GlucoseLevel'], 110)
        prepared_df['BloodPressure'] = get_col(['BloodPressure', 'blood pressure', 'bloodpressure', 'BP'], 70)
        prepared_df['SkinThickness'] = get_col(['SkinThickness', 'skinthickness', 'skin thickness', 'Skin'], 20)
        prepared_df['Insulin'] = get_col(['Insulin', 'insulin', 'InsulinLevel'], 80)
        prepared_df['BMI'] = get_col(['BMI', 'bmi', 'Body Mass Index'], 28.0)
        prepared_df['DiabetesPedigreeFunction'] = get_col(['DiabetesPedigreeFunction', 'diabetespedigreefunction', 'pedigree', 'family history score'], 0.45)
        prepared_df['Age'] = get_col(['Age', 'age'], 30)
        
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
        ages_col = prepared_df['Age']
        sexes_col = get_col(['sex', 'Sex', 'gender', 'Gender'], 'Female')
        
        for idx in df.index:
            p_name = names_col.loc[idx]
            p_age = ages_col.loc[idx]
            p_sex = str(sexes_col.loc[idx])
            p_pred = predictions[idx]
            save_prediction(p_name, p_age, p_sex, "Diabetes", p_pred)
            
        st.success("Batch predictions processed and logged to patient history successfully!")
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Persistent History Section
render_history_section()
