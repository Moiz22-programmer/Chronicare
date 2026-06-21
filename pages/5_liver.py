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
        @keyframes titleSlide {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes subtitleFade {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .page-header {
            animation: titleSlide 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }
    </style>
    <div class="page-header">
        <h1 style="color:#7C3AED; font-weight: 900; margin-bottom: 8px; font-size:3rem; letter-spacing:-0.02em;">🧪 Liver Health Diagnostics</h1>
        <p style="color:#94A3B8; font-size:1.15rem; line-height:1.6;">Comprehensive hepatic evaluation and disease screening</p>
    </div>
""", unsafe_allow_html=True)

# Load primary and baseline models
model, scaler = load_ml_model("liver_xgb.pkl")
model_base, _ = load_ml_model("liver_gb.pkl")

if model is None or scaler is None:
    st.warning("Model not loaded. Please add the trained model file.")
    st.stop()

# Layout: Form and File Upload
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header" style="margin-top:0;">👤 Patient Data</h3>', unsafe_allow_html=True)

# Two-column layout for the manual input form
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name", value="Robin Green")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex_str = st.selectbox("Sex", options=["Male", "Female"])
    sex = 1 if sex_str == "Male" else 0
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=75.0)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=172.0)
    
    # BMI Auto-Calculation
    bmi = 0.0
    if height > 0:
        bmi = weight / ((height / 100) ** 2)
    st.markdown(f"**Computed Body Mass Index (BMI):** <span style='color:#00BCD4; font-weight:700;'>{bmi:.1f}</span>", unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours per day", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
    activity = st.selectbox("Physical Activity Level", options=["Medium", "Low", "High"])
    smoking = st.selectbox("Smoking Status", options=["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", options=["No", "Yes"])

with col2:
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=80.0, value=1.0, step=0.1, help="Bilirubin pigment level. Normal: 0.2-1.2 mg/dL")
    direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.1, max_value=30.0, value=0.3, step=0.1, help="Conjugated bilirubin level. Normal: 0.1-0.4 mg/dL")
    alkphos = st.number_input("Alkaline Phosphatase (IU/L)", min_value=10, max_value=3000, value=250, help="Liver enzyme. Normal: 44-147 IU/L")
    sgpt = st.number_input("SGPT / ALT (U/L)", min_value=1, max_value=3000, value=50, help="Alanine Aminotransferase. Normal: 7-56 U/L")
    sgot = st.number_input("SGOT / AST (U/L)", min_value=1, max_value=5000, value=60, help="Aspartate Aminotransferase. Normal: 10-40 U/L")
    total_proteins = st.number_input("Total Proteins (g/dL)", min_value=1.0, max_value=15.0, value=6.8, step=0.1, help="Serum protein level. Normal: 6.0-8.3 g/dL")
    albumin = st.number_input("Albumin Level (g/dL)", min_value=0.5, max_value=10.0, value=3.5, step=0.1, help="Serum albumin level. Normal: 3.5-5.0 g/dL")
    ag_ratio = st.number_input("Albumin to Globulin Ratio", min_value=0.1, max_value=5.0, value=1.1, step=0.05)

st.markdown('</div>', unsafe_allow_html=True)
# Evaluation logic
if st.button("🔬 Analyze Patient Vitals", use_container_width=True):
    analyze_clicked = True
else:
    analyze_clicked = False

if analyze_clicked:
    # Prepare input for prediction
    # Features order: Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': sex,
        'Total_Bilirubin': total_bilirubin,
        'Direct_Bilirubin': direct_bilirubin,
        'Alkaline_Phosphotase': alkphos,
        'Alamine_Aminotransferase': sgpt,
        'Aspartate_Aminotransferase': sgot,
        'Total_Protiens': total_proteins,
        'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': ag_ratio
    }])
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prob = model.predict_proba(input_scaled)[0][1]
    prob_base = model_base.predict_proba(input_scaled)[0][1] if model_base is not None else prob
    
    prediction = "⚠️ Disease Detected" if prob > 0.5 else "✅ No Disease Detected"
    
    # Save to history
    save_prediction(name, age, sex_str, "Liver Disease", prediction)
    
    # Display Result
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#00BCD4; font-weight:700;">📊 Section C — Prediction Result & Report</h3>', unsafe_allow_html=True)
    
    if prediction == "⚠️ Disease Detected":
        st.markdown(f'<h4 style="color:#EF4444 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        
        # Why this happened explanation
        reasons = []
        if total_bilirubin > 1.2:
            reasons.append(f"your total bilirubin of {total_bilirubin} mg/dL is higher than the recommended limit of 1.2 mg/dL, pointing to potential filtration blockages or bile flow issues")
        if sgpt > 56:
            reasons.append(f"your SGPT (ALT) enzyme level of {sgpt} U/L is elevated, indicating that liver cells may be experiencing inflammation or stress")
        if sgot > 40:
            reasons.append(f"your SGOT (AST) enzyme level of {sgot} U/L is above normal limits, showing signs of tissue damage or liver fatigue")
        if ag_ratio < 1.0:
            reasons.append(f"your albumin-to-globulin ratio of {ag_ratio:.2f} is low, which can occur when the liver produces less albumin due to functional strain")
            
        if not reasons:
            reasons.append("the model identified elevated hepatic stress based on the combined elevations in your ALT/AST enzymes and bilirubin levels")
            
        why_text = f"This warning is displayed because " + ", and ".join(reasons[:2]) + ". These indicators suggest that your liver function requires clinical review."
        st.markdown(f"<p style='color:#CBD5E1; font-size:1.05rem; line-height:1.6; margin-bottom:20px;'><strong>Why this happened:</strong> {why_text}</p>", unsafe_allow_html=True)
        
        # Recovery Plan Tabs
        tab_food, tab_meds, tab_exercises = st.tabs(["🥗 Foods to Eat", "💊 Medicines / Supplements", "🏃 Exercises"])
        with tab_food:
            st.markdown("""
            - **Garlic:** Contains sulfur compounds that activate liver enzymes responsible for flushing out toxins.
            - **Grapefruit:** High in antioxidants (naringenin and naringin) that naturally protect the liver by reducing inflammation.
            - **Blueberries & Cranberries:** Contain anthocyanins, antioxidants that protect the liver from cell damage and fibrosis.
            - **Grapes:** Especially red and purple grapes, which contain beneficial plant compounds like resveratrol.
            - **Beetroot Juice:** A source of nitrates and antioxidants that reduce oxidative damage and inflammation in the liver.
            - **Cruciferous Vegetables:** Broccoli and Brussels sprouts increase the liver's natural detoxification enzymes.
            - **Olive Oil:** Helps decrease fat accumulation in the liver and improves liver enzyme levels.
            - **Prickly Pear:** Historically used to protect liver cells and reduce inflammation.
            """)
        with tab_meds:
            st.info("⚠️ **Disclaimer:** Consult a qualified medical practitioner or hepatologist before starting any medication or supplement regimen.")
            st.markdown("""
            - **Milk Thistle (Silymarin):** A popular herbal supplement widely studied for its anti-inflammatory and liver-protecting properties.
            - **Ursodeoxycholic Acid (UDCA):** A prescription bile acid used to help improve bile flow and reduce liver damage in certain conditions.
            - **Vitamin E:** A fat-soluble antioxidant sometimes recommended to reduce inflammation in non-alcoholic fatty liver disease (NAFLD).
            - **N-Acetyl Cysteine (NAC):** Helps replenish glutathione, the liver's primary antioxidant, supporting natural cell repair.
            - **Omega-3 Fatty Acids:** Supplements that help reduce liver fat accumulation and support cardiovascular health.
            """)
        with tab_exercises:
            st.markdown("""
            - **Brisk Walking:** 30-40 minutes of daily walking helps reduce fatty deposits in the liver.
            - **Jogging / Running:** Moderate cardiovascular activity that enhances metabolic rate and burns visceral fat.
            - **Cycling:** A great cardiovascular exercise that promotes overall metabolic efficiency and reduces liver strain.
            - **Swimming:** Low-impact, full-body activity that improves insulin sensitivity, helping clear fat from the liver.
            - **Low-Impact Aerobics:** Sustained light movement that boosts general cardiovascular and hepatic circulation.
            """)
    else:
        st.markdown(f'<h4 style="color:#00BCD4 !important; font-size:1.6rem; font-weight:800;">{prediction}</h4>', unsafe_allow_html=True)
        st.success("Your hepatic enzyme levels and serum protein ratios are healthy and normal. Keep supporting your liver health with balanced hydration and a low-toxin diet!")

    # PDF Report Generator Button
    try:
        pdf_data = {
            "Age": age,
            "Total_Bilirubin": total_bilirubin,
            "Direct_Bilirubin": direct_bilirubin,
            "Alkaline_Phosphotase": alkphos,
            "Alamine_Aminotransferase": sgpt,
            "Aspartate_Aminotransferase": sgot,
            "Total_Protiens": total_proteins,
            "Albumin": albumin,
            "Albumin_and_Globulin_Ratio": ag_ratio
        }
        p_id = f"CC-{np.random.randint(1000, 9999)}"
        pdf_bytes = generate_patient_report(
            patient_name=name,
            patient_id=p_id,
            disease="Liver Disease",
            patient_data=pdf_data,
            prob_primary=prob,
            prob_baseline=prob_base,
            model_primary_name="XGBoost",
            model_baseline_name="Gradient Boosting"
        )
        
        st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Complete Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=f"ChroniCare_Liver_Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to generate report PDF: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

# File Upload Section
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.markdown('<h3 class="section-header" style="margin-top:0;">📂 Batch Analysis</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV or Excel file for batch hepatic risk screening", type=["csv", "xlsx"])

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
            
        prepared_df['Age'] = get_col(['Age', 'age'], 45)
        
        # Map gender
        raw_sex = get_col(['gender', 'Gender', 'sex', 'Sex'], 'Male')
        prepared_df['Gender'] = raw_sex.apply(lambda x: 1 if str(x).strip().lower().startswith('m') or str(x) == '1' else 0)
        
        prepared_df['Total_Bilirubin'] = get_col(['Total_Bilirubin', 'total bilirubin', 'bilirubin', 'Total Bilirubin'], 1.0)
        prepared_df['Direct_Bilirubin'] = get_col(['Direct_Bilirubin', 'direct bilirubin', 'Direct Bilirubin'], 0.3)
        prepared_df['Alkaline_Phosphotase'] = get_col(['Alkaline_Phosphotase', 'alkaline phosphatase', 'alkaline phosphotase', 'alkphos', 'Alkaline Phosphatase'], 250)
        prepared_df['Alamine_Aminotransferase'] = get_col(['Alamine_Aminotransferase', 'alamine aminotransferase', 'alt', 'sgpt', 'ALT/SGPT'], 50)
        prepared_df['Aspartate_Aminotransferase'] = get_col(['Aspartate_Aminotransferase', 'aspartate aminotransferase', 'ast', 'sgot', 'AST/SGOT'], 60)
        prepared_df['Total_Protiens'] = get_col(['Total_Protiens', 'total proteins', 'total protiens', 'proteins', 'Total Proteins'], 6.8)
        prepared_df['Albumin'] = get_col(['Albumin', 'albumin', 'Albumin Level'], 3.5)
        prepared_df['Albumin_and_Globulin_Ratio'] = get_col(['Albumin_and_Globulin_Ratio', 'albumin and globulin ratio', 'ag ratio', 'ag_ratio', 'A/G Ratio'], 1.1)
        
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
            save_prediction(p_name, p_age, p_sex, "Liver Disease", p_pred)
            
        st.success("Batch predictions processed and logged to patient history successfully!")
        
    except Exception as e:
        st.error(f"Failed to process file: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Persistent History Section
render_history_section()
