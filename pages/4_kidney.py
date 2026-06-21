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
        <h1 style="color:#10B981; font-weight: 900; margin-bottom: 8px; font-size:3rem; letter-spacing:-0.02em;">🫘 Renal Function Assessment</h1>
        <p style="color:#94A3B8; font-size:1.15rem; margin:0; line-height:1.7; font-weight:400;">Comprehensive kidney health screening for Chronic Kidney Disease using serum chemistry and urinalysis.</p>
    </div>
""", unsafe_allow_html=True)

# Load primary and baseline models
model, scaler = load_ml_model("kidney_knn.pkl")
model_base, _ = load_ml_model("kidney_nb.pkl")

if model is None or scaler is None:
    st.warning("Model not loaded. Please add the trained model file.")
    st.stop()

# Layout: Form and File Upload
st.markdown('<div class="form-card"><h3 class="section-header" style="margin-top:0;">👤 Patient Clinical Data Input</h3><p class="section-subtitle">Enter patient vitals and renal function markers below:</p>', unsafe_allow_html=True)

# Two-column layout for the manual input form
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name", value="Alex Smith")
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex_str = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_str == "Male" else 0
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=72.0)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    
    # BMI Auto-Calculation
    bmi = 0.0
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**Computed Body Mass Index (BMI):** <span style='color:#00BCD4; font-weight:700;'>{bmi:.1f}</span>", unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours per day", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
    activity = st.selectbox("Physical Activity Level", options=["Low", "Medium", "High"])
    smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"])

with col2:
    bu = st.number_input("Blood Urea (mg/dL)", min_value=1, max_value=500, value=40, help="Standard normal range: 7-20 mg/dL")
    sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=50.0, value=1.1, step=0.1, help="Standard normal range: 0.6-1.2 mg/dL")
    sod = st.number_input("Sodium Level (mEq/L)", min_value=50, max_value=200, value=138, help="Standard normal range: 135-145 mEq/L")
    pot = st.number_input("Potassium Level (mEq/L)", min_value=1.0, max_value=15.0, value=4.5, step=0.1, help="Standard normal range: 3.5-5.0 mEq/L")
    hemo = st.number_input("Hemoglobin Level (g/dL)", min_value=1.0, max_value=25.0, value=13.5, step=0.1, help="Standard normal range: 12.0-17.5 g/dL")
    rc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=0.5, max_value=10.0, value=4.8, step=0.1)
    wc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=1000, max_value=50000, value=8000, step=100)
    
    htn_str = st.selectbox("Hypertension Status?", options=["No", "Yes"])
    htn = 1 if htn_str == "Yes" else 0
    
    dm_str = st.selectbox("Diabetes Mellitus Status?", options=["No", "Yes"])
    dm = 1 if dm_str == "Yes" else 0
    
    appet_str = st.selectbox("Appetite Quality", options=["Good", "Poor"])
    appet = 1 if appet_str == "Good" else 0
    
    pe_str = st.selectbox("Pedal Edema (Ankle Swelling)?", options=["No", "Yes"])
    pe = 1 if pe_str == "Yes" else 0

st.markdown('</div>', unsafe_allow_html=True)

# Evaluation logic
analyze_clicked = st.button("🔬 Analyze Patient Vitals", use_container_width=True)

if analyze_clicked:
    # Prepare input for prediction (24 features expected by model)
    # Defaulting missing inputs: bp=80, sg=1.020, al=0, su=0, rbc=1, pc=1, pcc=0, ba=0, bgr=120, pcv=40, cad=0, ane=0
    input_df = pd.DataFrame([{
        'age': age,
        'bp': 80,
        'sg': 1.020,
        'al': 0,
        'su': 0,
        'rbc': 1,
        'pc': 1,
        'pcc': 0,
        'ba': 0,
        'bgr': 120,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': 40,
        'wc': wc,
        'rc': rc,
        'htn': htn,
        'dm': dm,
        'cad': 0,
        'appet': appet,
        'pe': pe,
        'ane': 0
    }])
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prob_base = model_base.predict_proba(input_scaled)[0][1] if model_base is not None else prob
    
    prediction = "⚠️ Disease Detected" if prob > 0.5 else "✅ No Disease Detected"
    
    # Save to history
    save_prediction(name, age, sex_str, "Kidney Disease", prediction)
    
    # Display Result
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header" style="margin-top:0;">📊 Diagnostic Assessment Results</h3>', unsafe_allow_html=True)
    
    if prediction == "⚠️ Disease Detected":
        st.markdown(f'<h4 style="color:#EF4444 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        
        # Why this happened explanation
        reasons = []
        if sc > 1.2:
            reasons.append(f"your serum creatinine level of {sc} mg/dL is above the normal limit of 1.2 mg/dL, which points to reduced blood filtering in the kidneys")
        if bu > 20:
            reasons.append(f"your blood urea level of {bu} mg/dL is higher than the typical range of 7-20 mg/dL, suggesting accumulation of protein waste products")
        if hemo < 12.0:
            reasons.append(f"your hemoglobin level of {hemo} g/dL is low, which can indicate anemia, a common complication of kidney disease")
        if htn == 1:
            reasons.append("you have active hypertension, which places high physical pressure on the delicate vessels inside the kidneys")
            
        if not reasons:
            reasons.append("the model detected abnormal filtration and kidney stress based on your joint blood urea, creatinine, and electrolytes profile")
            
        why_text = f"This warning is displayed because " + ", and ".join(reasons[:2]) + ". These indicators suggest that your kidney function is under stress."
        st.markdown(f"<p style='color:#CBD5E1; font-size:1.05rem; line-height:1.6; margin-bottom:20px;'><strong>Why this happened:</strong> {why_text}</p>", unsafe_allow_html=True)
        
        # Recovery Plan Tabs
        tab_food, tab_meds, tab_exercises = st.tabs(["🥗 Foods to Eat", "💊 Medicines / Supplements", "🏃 Exercises"])
        with tab_food:
            st.markdown("""
            - **Cauliflower:** A low-potassium, low-sodium vegetable that provides vitamin C, folate, and fiber.
            - **Blueberries:** Low-potassium fruit packed with powerful antioxidants that protect vascular health.
            - **Red Grapes:** Containing resveratrol, which has been shown to reduce kidney inflammation.
            - **Garlic:** A great sodium-free alternative for adding flavor while providing anti-inflammatory benefits.
            - **Olive Oil:** Provides a healthy, phosphorus-free fat source for kidney-friendly cooking.
            - **Cabbage:** Low in potassium and sodium, high in vitamins K and C.
            - **Egg Whites:** High-quality, low-phosphorus protein suitable for kidney management.
            - **Buckwheat:** A nutritious whole grain low in phosphorus, ideal for renal diets.
            """)
        with tab_meds:
            st.info("⚠️ **Disclaimer:** Consult a qualified medical practitioner or nephrologist before starting any medication or supplement regimen.")
            st.markdown("""
            - **ACE Inhibitors / ARBs:** Blood pressure medications that slow the progression of kidney damage and protect nephrons.
            - **Erythropoietin (EPO) therapy:** Hormonal treatments to manage anemia by stimulating red blood cell production.
            - **Phosphate Binders:** Prescribed to prevent body accumulation of phosphorus from foods.
            - **Vitamin D & Calcium Supplements:** Used to maintain bone health when kidney function is reduced.
            - **Iron Supplements:** Administered to raise hemoglobin levels and combat fatigue from kidney strain.
            """)
        with tab_exercises:
            st.markdown("""
            - **Brisk Walking:** Increases circulation and cardiovascular health without putting excess stress on body organs.
            - **Stationary Cycling:** Safe, low-impact exercise that can be performed at a comfortable resistance level.
            - **Water Aerobics:** Light resistance training in water, which helps reduce swelling and joint pressure.
            - **Gentle Yoga & Breathing:** Helps manage blood pressure, reduces stress, and increases general flexibility.
            - **Light Stretching:** Keeps muscles active and reduces cramps, which are common in patients with kidney strain.
            """)
    else:
        st.markdown(f'<h4 style="color:#00BCD4 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        st.success("Your renal metrics and blood waste markers look healthy and stable. Keep drinking plenty of water and maintaining low-sodium dietary habits!")

    # PDF Report Generator Button
    try:
        pdf_data = {
            "age": age,
            "bp": 80,
            "bu": bu,
            "sc": sc,
            "sod": sod,
            "pot": pot,
            "hemo": hemo,
            "pcv": 40,
            "wc": wc,
            "rc": rc
        }
        p_id = f"CC-{np.random.randint(1000, 9999)}"
        pdf_bytes = generate_patient_report(
            patient_name=name,
            patient_id=p_id,
            disease="Kidney Disease",
            patient_data=pdf_data,
            prob_primary=prob,
            prob_baseline=prob_base,
            model_primary_name="K-Nearest Neighbors",
            model_baseline_name="Gaussian Naive Bayes"
        )
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Complete Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"ChroniCare_Kidney_Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate report PDF: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-top:0; color:#00BCD4; font-weight:700;">📂 Section B — CSV / Excel File Upload</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV or Excel file for batch kidney risk screening", type=["csv", "xlsx"])

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
        prepared_df['bp'] = get_col(['bp', 'blood pressure', 'bloodpressure', 'BP'], 80)
        prepared_df['sg'] = get_col(['sg', 'specific gravity', 'Specific Gravity'], 1.020)
        prepared_df['al'] = get_col(['al', 'albumin', 'Albumin'], 0)
        prepared_df['su'] = get_col(['su', 'sugar', 'Sugar'], 0)
        
        raw_rbc = get_col(['rbc', 'red blood cells', 'Red Blood Cells'], 'normal')
        prepared_df['rbc'] = raw_rbc.apply(lambda x: 1 if str(x).strip().lower().startswith('n') or str(x) == '1' else 0)
        
        raw_pc = get_col(['pc', 'pus cells', 'Pus Cells'], 'normal')
        prepared_df['pc'] = raw_pc.apply(lambda x: 1 if str(x).strip().lower().startswith('n') or str(x) == '1' else 0)
        
        raw_pcc = get_col(['pcc', 'pus cell clumps', 'Pus Cell Clumps'], 'notpresent')
        prepared_df['pcc'] = raw_pcc.apply(lambda x: 1 if str(x).strip().lower().startswith('p') or str(x) == '1' or str(x) == 'present' else 0)
        
        raw_ba = get_col(['ba', 'bacteria', 'Bacteria'], 'notpresent')
        prepared_df['ba'] = raw_ba.apply(lambda x: 1 if str(x).strip().lower().startswith('p') or str(x) == '1' or str(x) == 'present' else 0)
        
        prepared_df['bgr'] = get_col(['bgr', 'blood glucose random', 'blood glucose'], 120)
        prepared_df['bu'] = get_col(['bu', 'blood urea', 'Blood Urea'], 40)
        prepared_df['sc'] = get_col(['sc', 'serum creatinine', 'Serum Creatinine', 'creatinine'], 1.1)
        prepared_df['sod'] = get_col(['sod', 'sodium', 'Sodium Level'], 138)
        prepared_df['pot'] = get_col(['pot', 'potassium', 'Potassium Level'], 4.5)
        prepared_df['hemo'] = get_col(['hemo', 'hemoglobin', 'Hemoglobin'], 13.5)
        prepared_df['pcv'] = get_col(['pcv', 'packed cell volume', 'PCV'], 40)
        prepared_df['wc'] = get_col(['wc', 'white blood cell count', 'wbc', 'White Blood Cells'], 8000)
        prepared_df['rc'] = get_col(['rc', 'red blood cell count', 'rbc count', 'Red Blood Cells Count'], 4.8)
        
        def map_yn(val):
            return 1 if str(val).strip().lower().startswith('y') or str(val) == '1' or str(val) == 'true' else 0
            
        prepared_df['htn'] = get_col(['htn', 'hypertension', 'Hypertension'], 0).apply(map_yn)
        prepared_df['dm'] = get_col(['dm', 'diabetes mellitus', 'diabetes', 'Diabetes'], 0).apply(map_yn)
        prepared_df['cad'] = get_col(['cad', 'coronary artery disease'], 0).apply(map_yn)
        
        raw_appet = get_col(['appet', 'appetite', 'Appetite'], 'good')
        prepared_df['appet'] = raw_appet.apply(lambda x: 1 if str(x).strip().lower().startswith('g') or str(x) == '1' or str(x) == 'good' else 0)
        
        prepared_df['pe'] = get_col(['pe', 'pedal edema', 'edema', 'Pedal Edema'], 0).apply(map_yn)
        prepared_df['ane'] = get_col(['ane', 'anemia', 'Anemia'], 0).apply(map_yn)
        
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
            save_prediction(p_name, p_age, p_sex, "Kidney Disease", p_pred)
            
        st.success("Batch predictions processed and logged to patient history successfully!")
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Persistent History Section
render_history_section()
