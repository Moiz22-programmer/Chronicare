"""
patient_report_generator.py
ChroniCare - Individual Patient Clinical Report Generator
Generates a fully detailed PDF health report for a single patient including:
  - Diagnostic risk probability (both models)
  - Out-of-range indicator analysis ("Why this happened")
  - Untreated prognosis section ("What happens if ignored")
  - Personalized care plan ("What to do")
"""

import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ─────────────────────────────────────────────
# Reference healthy ranges for each feature
# ─────────────────────────────────────────────
HEALTHY_RANGES = {
    # Diabetes
    "Pregnancies":               (0, 6,    "pregnancies"),
    "Glucose":                   (70, 99,  "mg/dL"),
    "BloodPressure":             (60, 80,  "mmHg"),
    "SkinThickness":             (10, 35,  "mm"),
    "Insulin":                   (16, 166, "μU/mL"),
    "BMI":                       (18.5, 24.9, "kg/m²"),
    "DiabetesPedigreeFunction":  (0.0, 0.6, "score"),
    "Age":                       (0, 120,  "years"),

    # Heart
    "age":         (20, 65,  "years"),
    "trestbps":    (90, 120, "mmHg"),
    "chol":        (100, 200,"mg/dL"),
    "thalach":     (100, 170,"bpm"),
    "oldpeak":     (0.0, 1.5,"ST units"),

    # Kidney
    "bp":    (60, 90,   "mmHg"),
    "bgr":   (70, 110,  "mg/dL"),
    "bu":    (7, 25,    "mg/dL"),
    "sc":    (0.6, 1.2, "mg/dL"),
    "sod":   (136, 145, "mEq/L"),
    "pot":   (3.5, 5.0, "mEq/L"),
    "hemo":  (12.0, 17.5,"g/dL"),
    "pcv":   (36, 52,   "%"),
    "wc":    (4500, 11000,"cells/μL"),
    "rc":    (4.2, 6.1, "million/μL"),

    # Liver
    "Total_Bilirubin":            (0.1, 1.2,  "mg/dL"),
    "Direct_Bilirubin":           (0.0, 0.3,  "mg/dL"),
    "Alkaline_Phosphotase":       (44, 147,   "IU/L"),
    "Alamine_Aminotransferase":   (7, 56,     "IU/L"),
    "Aspartate_Aminotransferase": (10, 40,    "IU/L"),
    "Total_Protiens":             (6.0, 8.3,  "g/dL"),
    "Albumin":                    (3.5, 5.0,  "g/dL"),
    "Albumin_and_Globulin_Ratio": (1.1, 2.5,  "ratio"),
}

FEATURE_LABELS = {
    # Diabetes
    "Pregnancies":               "Number of Pregnancies",
    "Glucose":                   "Plasma Glucose Level",
    "BloodPressure":             "Diastolic Blood Pressure",
    "SkinThickness":             "Triceps Skin Fold Thickness",
    "Insulin":                   "2-Hour Serum Insulin",
    "BMI":                       "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction":  "Diabetes Pedigree Function",

    # Heart
    "age":      "Patient Age",
    "sex":      "Gender (1=Male, 0=Female)",
    "cp":       "Chest Pain Type (0–3)",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Serum Cholesterol",
    "fbs":      "Fasting Blood Sugar >120mg/dL",
    "restecg":  "Resting ECG Result",
    "thalach":  "Maximum Heart Rate Achieved",
    "exang":    "Exercise Induced Angina",
    "oldpeak":  "ST Depression (Oldpeak)",
    "slope":    "Slope of ST Segment",
    "ca":       "Major Vessels Colored (Fluoroscopy)",
    "thal":     "Thalassemia Type",

    # Kidney
    "bp":    "Blood Pressure",
    "sg":    "Urine Specific Gravity",
    "al":    "Albuminuria Level (0–5)",
    "su":    "Glycosuria Level (0–5)",
    "rbc":   "Red Blood Cells (1=Normal)",
    "pc":    "Pus Cell (1=Normal)",
    "pcc":   "Pus Cell Clumps (1=Present)",
    "ba":    "Bacteria (1=Present)",
    "bgr":   "Blood Glucose Random",
    "bu":    "Blood Urea",
    "sc":    "Serum Creatinine",
    "sod":   "Serum Sodium",
    "pot":   "Serum Potassium",
    "hemo":  "Hemoglobin",
    "pcv":   "Packed Cell Volume",
    "wc":    "White Cell Count",
    "rc":    "Red Cell Count",
    "htn":   "Hypertension (1=Yes)",
    "dm":    "Diabetes Mellitus (1=Yes)",
    "cad":   "Coronary Artery Disease (1=Yes)",
    "appet": "Appetite (1=Good)",
    "pe":    "Pedal Edema (1=Yes)",
    "ane":   "Anemia (1=Yes)",

    # Liver
    "Age":                        "Patient Age",
    "Gender":                     "Gender (1=Male)",
    "Total_Bilirubin":            "Total Bilirubin",
    "Direct_Bilirubin":           "Direct Bilirubin",
    "Alkaline_Phosphotase":       "Alkaline Phosphatase",
    "Alamine_Aminotransferase":   "ALT (SGPT)",
    "Aspartate_Aminotransferase": "AST (SGOT)",
    "Total_Protiens":             "Total Proteins",
    "Albumin":                    "Albumin",
    "Albumin_and_Globulin_Ratio": "Albumin/Globulin Ratio",
}

DISEASE_COLORS = {
    "Diabetes":      "#3B82F6",
    "Heart Disease": "#EF4444",
    "Kidney Disease":"#10B981",
    "Liver Disease": "#7C3AED",
}

PROGNOSIS_TEXT = {
    "Diabetes": {
        "causes": (
            "Elevated blood glucose (hyperglycemia) occurs when the pancreatic beta cells fail to produce "
            "sufficient insulin, or when cells become resistant to insulin signaling. Key contributors include "
            "genetic predisposition (Diabetes Pedigree Function), elevated BMI indicating adipose tissue "
            "resistance, advanced age reducing cellular insulin sensitivity, and a sedentary, high-carbohydrate "
            "dietary lifestyle causing repeated glycemic spikes."
        ),
        "if_untreated": (
            "Without management, chronically elevated blood glucose causes progressive damage to the vascular "
            "system and nervous tissue. Likely complications over 5–10 years include: diabetic retinopathy "
            "(vision loss), nephropathy (kidney damage), peripheral neuropathy (limb numbness and pain), "
            "cardiovascular disease (2–4× elevated cardiac risk), non-healing foot ulcers, and in severe cases "
            "diabetic ketoacidosis requiring emergency hospitalization."
        ),
        "care_plan": [
            "🥗 Dietary: Follow a low-glycemic-index diet. Reduce refined carbohydrates, added sugars, and "
            "sugary beverages. Prioritize leafy vegetables, legumes, whole grains, and lean proteins.",
            "🏃 Physical Activity: Engage in at least 150 minutes per week of moderate aerobic activity "
            "(brisk walking, swimming, cycling). Resistance training 2× per week improves insulin sensitivity.",
            "📊 Monitoring: Track fasting glucose daily. Schedule HbA1c blood panel every 3 months.",
            "💊 Medical: Consult an endocrinologist for possible Metformin therapy or insulin titration. "
            "Do not adjust medications without clinical supervision.",
            "⚖️ Weight Management: A 5–10% reduction in body weight can significantly improve glycemic control "
            "and reduce insulin resistance.",
            "🚭 Lifestyle: Quit smoking (doubles cardiovascular risk in diabetics). Limit alcohol to "
            "<1 unit/day for women, <2 units/day for men.",
        ],
    },
    "Heart Disease": {
        "causes": (
            "Coronary artery disease (CAD) develops when plaque (cholesterol, fat, calcium deposits) "
            "accumulates inside coronary arteries, progressively restricting blood flow to the heart muscle. "
            "Key risk amplifiers include elevated resting blood pressure (trestbps), high serum cholesterol, "
            "asymptomatic chest pain type (type 0), exercise-induced angina, ST-segment depression (oldpeak), "
            "multiple blocked vessels (ca > 0), and a Thalassemia type of 3 (reversible defect)."
        ),
        "if_untreated": (
            "Untreated coronary artery disease carries a risk of acute myocardial infarction (heart attack), "
            "which can occur with little warning. Progression includes: stable angina → unstable angina → "
            "acute MI → heart failure. ST depression episodes indicate ischemia; if recurring without "
            "intervention, permanent myocardial scarring occurs. Risk of sudden cardiac death is significantly "
            "elevated in patients with blocked vessels and exercise angina."
        ),
        "care_plan": [
            "🥗 Dietary: Adopt the Mediterranean or DASH diet. Reduce saturated fats, trans fats, and dietary "
            "cholesterol. Consume omega-3 rich foods (salmon, flaxseed, walnuts).",
            "🏃 Exercise: Engage in cardiac rehabilitation under physician supervision. Aim for moderate "
            "aerobic activity 5 days/week. Avoid sudden high-intensity activities.",
            "💊 Medical: Discuss statin therapy for cholesterol management, antiplatelet agents (aspirin), "
            "beta-blockers for blood pressure, and ACE inhibitors with your cardiologist.",
            "📊 Monitoring: Weekly blood pressure tracking. Quarterly lipid panel and ECG monitoring.",
            "🧘 Stress Management: Chronic stress elevates cortisol which worsens cardiac markers. Practice "
            "daily breathing exercises, meditation, or yoga.",
            "🚭 Lifestyle: Stop smoking immediately (reduces cardiac risk by 50% within 1 year). Limit "
            "alcohol strictly. Maintain a healthy BMI (below 25).",
        ],
    },
    "Kidney Disease": {
        "causes": (
            "Chronic Kidney Disease (CKD) damages the nephrons — the microscopic filtration units in the "
            "kidneys — gradually reducing the Glomerular Filtration Rate (GFR). Primary causes include "
            "prolonged hypertension, diabetes mellitus (damages glomerular capillaries), albuminuria "
            "(protein leakage indicating glomerular damage), elevated serum creatinine (waste buildup), "
            "and low specific gravity (dilute urine indicating reduced concentration ability). "
            "Pedal edema and anemia are common secondary indicators of advanced CKD."
        ),
        "if_untreated": (
            "CKD progresses through 5 stages based on GFR decline. Without intervention, Stage 3–4 CKD "
            "advances to End-Stage Renal Disease (ESRD) requiring dialysis or transplantation. Electrolyte "
            "imbalances (hyperkalemia, hyponatremia) cause cardiac arrhythmias. Accumulation of uremic "
            "toxins affects the brain (uremic encephalopathy), lungs (pulmonary edema), and bones "
            "(renal osteodystrophy). Life expectancy decreases dramatically without treatment."
        ),
        "care_plan": [
            "💧 Hydration: Maintain precise fluid intake as prescribed by your nephrologist. Over-hydration "
            "and dehydration both stress the kidneys.",
            "🥗 Dietary: Adopt a renal diet — low sodium (<2g/day), low potassium, low phosphorus. Reduce "
            "high-protein foods (excess nitrogen creates urea burden on kidneys).",
            "💊 Medical: ACE inhibitors or ARBs reduce proteinuria and slow CKD progression. Erythropoietin "
            "injections address anemia. Schedule quarterly nephrology reviews.",
            "📊 Monitoring: Track serum creatinine, GFR, potassium, and albumin every 3 months. Blood "
            "pressure must remain below 130/80 mmHg.",
            "🚫 Avoidance: Strictly avoid NSAIDs (ibuprofen, naproxen) — they reduce renal blood flow and "
            "accelerate CKD. Avoid nephrotoxic antibiotics (aminoglycosides) without specialist oversight.",
            "⚖️ Comorbidity Control: Tightly control diabetes and hypertension — they are the #1 and #2 "
            "causes of CKD progression.",
        ],
    },
    "Liver Disease": {
        "causes": (
            "Liver disease occurs when hepatocytes (liver cells) are chronically damaged and replaced by "
            "scar tissue (fibrosis / cirrhosis). Key biochemical markers include elevated bilirubin (impaired "
            "bile processing), raised ALT/AST transaminase enzymes (hepatocyte damage), reduced albumin "
            "levels (impaired protein synthesis), and an abnormal A/G ratio. Common causes are alcoholic "
            "liver disease, non-alcoholic fatty liver disease (NAFLD), viral hepatitis (B and C), "
            "autoimmune hepatitis, and chronic medication hepatotoxicity."
        ),
        "if_untreated": (
            "Progressive liver fibrosis leads to cirrhosis — irreversible scarring that prevents normal "
            "liver function. Complications of untreated cirrhosis include: portal hypertension (causing "
            "esophageal varices at risk of fatal hemorrhage), ascites (fluid accumulation in the abdomen), "
            "hepatic encephalopathy (brain toxicity), spontaneous bacterial peritonitis (SBP), and "
            "hepatocellular carcinoma (liver cancer). Liver failure is fatal without transplantation."
        ),
        "care_plan": [
            "🚫 Alcohol: Complete alcohol abstinence is the single most impactful intervention for most "
            "liver conditions. Even small amounts worsen ALT/AST levels.",
            "💊 Medications: Review all current medications and supplements with a hepatologist — many are "
            "hepatotoxic (e.g. high-dose paracetamol, statins, certain herbal remedies).",
            "🥗 Dietary: Follow a low-fat, high-antioxidant diet. Include cruciferous vegetables (broccoli, "
            "Brussels sprouts). Maintain adequate protein intake to support albumin production.",
            "📊 Monitoring: Liver function tests (LFTs) including ALT, AST, bilirubin, and albumin every "
            "3 months. Liver ultrasound every 6 months for cirrhosis surveillance.",
            "⚖️ Weight Management: NAFLD (fatty liver) significantly worsens with obesity. A structured "
            "weight loss of 7–10% body weight dramatically reduces liver fat.",
            "💉 Vaccination: Get vaccinated for Hepatitis A and B if not already immune — superinfection "
            "in a diseased liver can be rapidly fatal.",
        ],
    },
}


MEDICATION_SUGGESTIONS = {
    "Diabetes": [
        "Metformin — first-line oral antihyperglycemic (consult endocrinologist)",
        "SGLT2 inhibitors (e.g., empagliflozin) — in patients with cardiovascular/renal benefit",
        "GLP-1 receptor agonists (e.g., liraglutide) — for weight loss and glycemic control",
        "Insulin therapy — short or long-acting formulations for significant hyperglycemia",
    ],
    "Heart Disease": [
        "Statins — for LDL-cholesterol lowering and plaque stabilization",
        "Antiplatelet therapy (e.g., low-dose aspirin) — secondary prevention where indicated",
        "Beta-blockers — to reduce myocardial oxygen demand and control rate",
        "ACE inhibitors / ARBs — for blood pressure and remodeling benefits",
    ],
    "Kidney Disease": [
        "ACE inhibitors or ARBs — reduce proteinuria and slow CKD progression",
        "Erythropoiesis-stimulating agents — for CKD-related anemia when indicated",
        "Phosphate binders — for hyperphosphatemia in later CKD stages",
        "SGLT2 inhibitors — emerging renal protective class for diabetic kidney disease",
    ],
    "Liver Disease": [
        "Avoid hepatotoxic medications; review all current drugs with a hepatologist",
        "Specific antiviral therapy for viral hepatitis where applicable (HBV/HCV) — specialist-led",
        "Vitamin supplementation and hepatoprotective strategies as recommended by clinician",
    ],
}


def _check_abnormal_features(patient_data: dict, disease: str) -> list:
    """Returns list of (label, value, unit, low, high, direction) for out-of-range values."""
    abnormal = []
    for col, val in patient_data.items():
        if col in HEALTHY_RANGES:
            lo, hi, unit = HEALTHY_RANGES[col]
            try:
                fval = float(val)
                if fval < lo:
                    direction = "LOW"
                    abnormal.append((FEATURE_LABELS.get(col, col), round(fval, 2), unit, lo, hi, direction))
                elif fval > hi:
                    direction = "HIGH"
                    abnormal.append((FEATURE_LABELS.get(col, col), round(fval, 2), unit, lo, hi, direction))
            except (TypeError, ValueError):
                pass
    return abnormal


def generate_patient_report(
    patient_name: str,
    patient_id: str,
    disease: str,
    patient_data: dict,
    prob_primary: float,
    prob_baseline: float,
    model_primary_name: str,
    model_baseline_name: str,
) -> bytes:
    """
    Generate a full patient PDF report and return as bytes (for st.download_button).
    """
    buffer = io.BytesIO()
    report_date = datetime.datetime.now().strftime("%B %d, %Y  |  %I:%M %p")
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    accent = DISEASE_COLORS.get(disease, "#3B82F6")
    accent_color = colors.HexColor(accent)
    
    # ── Style definitions ────────────────────────────────────────────────────
    cover_title = ParagraphStyle(
        "CoverTitle", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=30, leading=36,
        textColor=colors.HexColor("#0F172A"), alignment=TA_LEFT, spaceAfter=6,
    )
    cover_sub = ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontName="Helvetica", fontSize=13, leading=18,
        textColor=colors.HexColor("#475569"), alignment=TA_LEFT, spaceAfter=4,
    )
    cover_meta = ParagraphStyle(
        "CoverMeta", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=15,
        textColor=colors.HexColor("#64748B"), alignment=TA_LEFT,
    )
    h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontName="Helvetica-Bold", fontSize=15, leading=20,
        textColor=accent_color, spaceBefore=18, spaceAfter=8, keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=colors.HexColor("#1E3A8A"), spaceBefore=10, spaceAfter=5, keepWithNext=True,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=10, leading=15,
        textColor=colors.HexColor("#1E293B"), spaceAfter=8,
    )
    bullet = ParagraphStyle(
        "Bullet", parent=styles["BodyText"],
        fontName="Helvetica", fontSize=9.5, leading=14,
        leftIndent=18, firstLineIndent=-12,
        textColor=colors.HexColor("#1E293B"), spaceAfter=5,
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9, leading=12,
        textColor=colors.HexColor("#64748B"),
    )
    disclaimer = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
        textColor=colors.HexColor("#94A3B8"), alignment=TA_CENTER,
    )

    story = []

    # ── COVER PAGE ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * inch))

    # Header banner table
    header_data = [[
        Paragraph("🏥  ChroniCare", ParagraphStyle(
            "BrandH", fontName="Helvetica-Bold", fontSize=20,
            textColor=colors.white, leading=24,
        )),
        Paragraph("PATIENT CLINICAL REPORT", ParagraphStyle(
            "BrandTag", fontName="Helvetica-Bold", fontSize=10,
            textColor=colors.HexColor("#CBD5E1"), leading=14, alignment=TA_RIGHT,
        )),
    ]]
    header_table = Table(header_data, colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 18),
        ("RIGHTPADDING", (0, 0), (-1, -1), 18),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.35 * inch))

    story.append(Paragraph(f"{disease} Diagnostic Summary", cover_title))
    story.append(Paragraph(f"Patient: <b>{patient_name}</b>  |  ID: {patient_id}", cover_sub))
    story.append(Paragraph(f"Report Generated: {report_date}", cover_meta))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceAfter=20))

    # Risk summary card
    avg_prob = (prob_primary + prob_baseline) / 2
    risk_pct = int(avg_prob * 100)
    if avg_prob < 0.3:
        risk_label, risk_color = "LOW RISK", "#10B981"
    elif avg_prob < 0.6:
        risk_label, risk_color = "MODERATE RISK", "#F59E0B"
    else:
        risk_label, risk_color = "HIGH RISK", "#EF4444"

    risk_data = [[
        Paragraph(
            f"<b>Combined Risk Assessment</b><br/>"
            f"<font size=28 color='{risk_color}'><b>{risk_pct}%</b></font><br/>"
            f"<font size=11 color='{risk_color}'><b>{risk_label}</b></font>",
            ParagraphStyle("RiskCenter", fontName="Helvetica-Bold", fontSize=11,
                           alignment=TA_CENTER, leading=34, textColor=colors.HexColor("#1E293B"))
        ),
        Paragraph(
            f"<b>Primary Model:</b>  {model_primary_name}<br/>"
            f"Probability: <b><font color='{risk_color}'>{int(prob_primary*100)}%</font></b><br/><br/>"
            f"<b>Baseline Model:</b>  {model_baseline_name}<br/>"
            f"Probability: <b><font color='{risk_color}'>{int(prob_baseline*100)}%</font></b>",
            ParagraphStyle("RiskDetail", fontName="Helvetica", fontSize=10.5, leading=18,
                           textColor=colors.HexColor("#334155"))
        ),
    ]]
    risk_table = Table(risk_data, colWidths=[2.8 * inch, 4.2 * inch])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F8FAFC")),
        ("BACKGROUND", (1, 0), (1, 0), colors.white),
        ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor(risk_color)),
        ("LINEAFTER", (0, 0), (0, -1), 1, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph(
        "⚠️  This report is intended for informational purposes only. It is not a substitute for "
        "professional medical diagnosis, treatment, or advice. Always consult a qualified physician "
        "before making any healthcare decisions.",
        disclaimer
    ))

    story.append(PageBreak())

    # ── SECTION 1: PATIENT VITALS TABLE ─────────────────────────────────────
    story.append(Paragraph(f"1. Patient Input Vitals — {disease}", h1))
    story.append(Paragraph(
        "The following table lists all clinical parameters entered for this patient. "
        "Values marked in <font color='#EF4444'><b>red (HIGH)</b></font> or "
        "<font color='#F59E0B'><b>orange (LOW)</b></font> fall outside of established healthy reference ranges.",
        body
    ))

    vitals_header = [["Clinical Parameter", "Entered Value", "Healthy Range", "Status"]]
    vitals_rows = []
    for col, val in patient_data.items():
        label = FEATURE_LABELS.get(col, col)
        display_val = str(round(float(val), 2)) if isinstance(val, (int, float)) else str(val)
        if col in HEALTHY_RANGES:
            lo, hi, unit = HEALTHY_RANGES[col]
            range_str = f"{lo} – {hi} {unit}"
            try:
                fval = float(val)
                if fval < lo:
                    status = Paragraph("<font color='#F59E0B'><b>⬇ LOW</b></font>",
                                       ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=9, leading=12))
                elif fval > hi:
                    status = Paragraph("<font color='#EF4444'><b>⬆ HIGH</b></font>",
                                       ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=9, leading=12))
                else:
                    status = Paragraph("<font color='#10B981'><b>✓ Normal</b></font>",
                                       ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=9, leading=12))
            except (TypeError, ValueError):
                status = Paragraph("—", ParagraphStyle("s", fontSize=9, leading=12))
        else:
            range_str = "—"
            status = Paragraph("—", ParagraphStyle("s", fontSize=9, leading=12))

        vitals_rows.append([label, display_val, range_str, status])

    vitals_data = vitals_header + vitals_rows
    vitals_table = Table(vitals_data, colWidths=[2.6 * inch, 1.3 * inch, 2.0 * inch, 1.1 * inch])
    
    vitals_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ALIGN", (3, 0), (3, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]
    # Alternate row shading
    for i in range(1, len(vitals_data)):
        if i % 2 == 0:
            vitals_style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F8FAFC")))
    vitals_table.setStyle(TableStyle(vitals_style))
    story.append(vitals_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── SECTION 2: RISK FACTOR ANALYSIS ─────────────────────────────────────
    story.append(Paragraph("2. Elevated Risk Factor Analysis", h1))

    disease_key = disease
    prognosis = PROGNOSIS_TEXT.get(disease_key, {})
    causes_text = prognosis.get("causes", "")
    story.append(Paragraph("<b>Clinical Basis of Risk:</b>", h2))
    story.append(Paragraph(causes_text, body))

    # List the abnormal features specifically for this patient
    abnormal_features = _check_abnormal_features(patient_data, disease)
    if abnormal_features:
        story.append(Paragraph("<b>Out-of-Range Indicators for This Patient:</b>", h2))
        ab_header = [["Indicator", "Patient Value", "Normal Range", "Direction", "Clinical Significance"]]
        ab_rows = []
        
        significance_map = {
            # Diabetes
            "Glucose": "Elevated glucose is the primary diagnostic marker for diabetes; chronic hyperglycemia damages blood vessels.",
            "BMI": "High BMI causes adipose tissue-induced insulin resistance, worsening glycemic control.",
            "DiabetesPedigreeFunction": "Genetic predisposition increases lifetime diabetes risk 2–3 fold.",
            "Insulin": "Abnormal insulin indicates either resistance or insufficient pancreatic output.",
            # Heart
            "chol": "High cholesterol accelerates atherosclerotic plaque formation in coronary arteries.",
            "trestbps": "Elevated resting blood pressure increases cardiac workload and arterial wall stress.",
            "thalach": "Reduced max heart rate indicates poor cardiac reserve and fitness.",
            "oldpeak": "ST depression indicates myocardial ischemia during exertion.",
            # Kidney
            "sc": "Elevated creatinine directly indicates reduced kidney filtration function.",
            "bu": "High blood urea indicates impaired nitrogen excretion by the kidneys.",
            "hemo": "Low hemoglobin (anemia) is a hallmark of CKD affecting erythropoietin production.",
            "sod": "Sodium imbalance reflects impaired renal electrolyte regulation.",
            # Liver
            "Total_Bilirubin": "Elevated bilirubin indicates impaired bile processing — a key liver function.",
            "Alamine_Aminotransferase": "Raised ALT is the gold standard marker of hepatocyte injury.",
            "Aspartate_Aminotransferase": "Elevated AST alongside ALT confirms liver cell damage.",
            "Albumin": "Low albumin indicates impaired hepatic protein synthesis, a sign of chronic disease.",
        }
        
        for label, val, unit, lo, hi, direction in abnormal_features:
            sig = ""
            for key, text in significance_map.items():
                if FEATURE_LABELS.get(key, "") == label or key == label:
                    sig = text
                    break
            if not sig:
                sig = f"Value is {direction.lower()} compared to the expected healthy range of {lo}–{hi} {unit}."
            
            dir_para = Paragraph(
                f"<font color='{'#EF4444' if direction == 'HIGH' else '#F59E0B'}'><b>⬆ {direction}</b></font>"
                if direction == "HIGH" else
                f"<font color='#F59E0B'><b>⬇ {direction}</b></font>",
                ParagraphStyle("ds", fontName="Helvetica-Bold", fontSize=9, leading=12)
            )
            ab_rows.append([label, f"{val} {unit}", f"{lo}–{hi} {unit}", dir_para,
                            Paragraph(sig, ParagraphStyle("ss", fontSize=8.5, leading=12,
                                                          textColor=colors.HexColor("#334155")))])
        
        ab_data = ab_header + ab_rows
        ab_table = Table(ab_data, colWidths=[1.5*inch, 1.1*inch, 1.2*inch, 0.8*inch, 2.4*inch])
        ab_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7F1D1D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FCA5A5")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FFF5F5")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        ab_table.setStyle(TableStyle(ab_style))
        story.append(ab_table)
    else:
        story.append(Paragraph(
            "✅ All entered patient values fall within established healthy reference ranges. "
            "The predicted risk may be influenced by interaction patterns between multiple factors "
            "rather than single-value extremes.", body
        ))

    story.append(Spacer(1, 0.15 * inch))

    # ── SECTION 3: PROGNOSIS IF UNTREATED ───────────────────────────────────
    story.append(Paragraph("3. Clinical Prognosis if Untreated", h1))
    untreated = prognosis.get("if_untreated", "")
    story.append(Paragraph(untreated, body))

    # Risk urgency box
    if avg_prob >= 0.6:
        urgency_text = (
            "<b>⚠️ URGENT CLINICAL ATTENTION RECOMMENDED</b><br/>"
            "The combined model risk exceeds 60%. Immediate evaluation by a specialist "
            "is strongly advised. Do not delay scheduling a clinical appointment."
        )
        urgency_color = "#7F1D1D"
        urgency_bg = "#FEF2F2"
    elif avg_prob >= 0.3:
        urgency_text = (
            "<b>⚡ ELEVATED RISK — PROACTIVE MONITORING ADVISED</b><br/>"
            "The risk level is moderate. Schedule a clinical review within the next 4–8 weeks "
            "and begin implementing the care plan below immediately."
        )
        urgency_color = "#78350F"
        urgency_bg = "#FFFBEB"
    else:
        urgency_text = (
            "<b>✅ LOW RISK — MAINTAIN HEALTHY PRACTICES</b><br/>"
            "Risk indicators are within acceptable bounds. Continue regular annual check-ups "
            "and maintain the healthy habits outlined in the care plan below."
        )
        urgency_color = "#14532D"
        urgency_bg = "#F0FDF4"

    urgency_data = [[Paragraph(urgency_text, ParagraphStyle(
        "Urgency", fontName="Helvetica", fontSize=10.5, leading=16,
        textColor=colors.HexColor(urgency_color)
    ))]]
    urgency_table = Table(urgency_data, colWidths=[7.0 * inch])
    urgency_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(urgency_bg)),
        ("BOX", (0, 0), (-1, -1), 2, colors.HexColor(urgency_color)),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ]))
    story.append(urgency_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── SECTION 4: PERSONALIZED CARE PLAN ───────────────────────────────────
    story.append(Paragraph("4. Personalized Recovery & Prevention Care Plan", h1))
    story.append(Paragraph(
        "The following evidence-based recommendations are tailored to the identified risk factors "
        f"for {disease}. These should be reviewed with a healthcare provider before implementation.", body
    ))

    care_bullets = prognosis.get("care_plan", [])
    for item in care_bullets:
        story.append(Paragraph(f"• {item}", bullet))

    # ── Medication suggestions (informational only) ─────────────────────
    meds = MEDICATION_SUGGESTIONS.get(disease, [])
    if meds:
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("Suggested Medication Classes (Informational)", h2))
        story.append(Paragraph(
            "The following medication classes are commonly used in clinical management of this condition. "
            "These are examples for clinician discussion and NOT a prescription. Always consult a licensed provider before starting or changing any medication.",
            body
        ))
        for med in meds:
            story.append(Paragraph(f"• {med}", bullet))
    story.append(Spacer(1, 0.2 * inch))

    # ── SECTION 5: FOLLOW-UP SCHEDULE ───────────────────────────────────────
    story.append(Paragraph("5. Recommended Follow-Up Schedule", h1))

    followup_data = [
        ["Timeframe", "Action Item", "Priority"],
        ["Within 1 Week", "Share this report with your primary care physician.", "🔴 High"],
        ["Within 1 Month",
         "Schedule specialist consultation (endocrinologist / cardiologist / nephrologist / hepatologist).",
         "🔴 High" if avg_prob >= 0.5 else "🟡 Medium"],
        ["Every 3 Months", "Repeat relevant blood panels and lab tests.", "🟡 Medium"],
        ["Every 6 Months", "Full clinical vitals check and risk re-assessment.", "🟢 Routine"],
        ["Annually", "Complete physical examination and imaging where appropriate.", "🟢 Routine"],
    ]
    followup_table = Table(followup_data, colWidths=[1.6*inch, 4.1*inch, 1.3*inch])
    followup_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(followup_table)
    story.append(Spacer(1, 0.35 * inch))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"ChroniCare Diagnostic Platform  |  Report ID: {patient_id}-{disease[:3].upper()}"
        f"  |  Generated: {report_date}<br/>"
        "This document is auto-generated by AI models and is not a medical certificate. "
        "For clinical decisions, consult a licensed medical professional.",
        disclaimer
    ))

    doc.build(story)
    return buffer.getvalue()
