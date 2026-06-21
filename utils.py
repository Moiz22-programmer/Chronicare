import os
import pickle
import datetime
import pandas as pd
import streamlit as st

MODELS_DIR = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
HISTORY_FILE = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\patient_history.csv"

def load_ml_model(filename):
    """
    Loads a model and scaler tuple from the models folder.
    Returns (model, scaler) if successful, otherwise (None, None).
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"⚠️ Model not loaded. Please add the trained model file: '{filename}' in models/ directory.")
        return None, None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple) and len(data) == 2:
                return data[0], data[1]
            return data, None
    except Exception as e:
        st.error(f"Error loading model {filename}: {str(e)}")
        return None, None

def save_prediction(name, age, sex, disease_page, prediction_result):
    """
    Appends a new patient prediction record to the local patient_history.csv file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = pd.DataFrame([{
        "Patient Name": name,
        "Age": int(age),
        "Sex": str(sex),
        "Disease Page": str(disease_page),
        "Prediction Result": str(prediction_result),
        "Date & Time": timestamp
    }])
    
    try:
        if os.path.exists(HISTORY_FILE):
            try:
                df = pd.read_csv(HISTORY_FILE)
                # Keep column compatibility
                required_cols = ["Patient Name", "Age", "Sex", "Disease Page", "Prediction Result", "Date & Time"]
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = None
                df = pd.concat([df, new_record], ignore_index=True)
            except Exception:
                df = new_record
        else:
            df = new_record
            
        df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving to patient history: {str(e)}")

def render_history_section():
    """
    Renders the Patient History section in a professional card layout.
    Displays the full history, clear option, and download option.
    Uses a custom HTML table so the light/dark theme CSS can control styling.
    """
    st.markdown('<div class="glass-card" style="margin-top: 30px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#00BCD4; font-weight:700;">📋 Global Patient Diagnostics History</h3>', unsafe_allow_html=True)
    
    is_light = st.session_state.get("active_theme") == "Light"

    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if len(df) == 0:
                st.info("No patient history found. Make predictions to populate history.")
            else:
                # Build a fully-styled HTML table so CSS theme can control colours
                if is_light:
                    hdr_bg = "#F1F5F9"; hdr_color = "#0F172A"
                    row_bg = "#FFFFFF"; row_alt = "#F8FAFC"; row_hover = "#E0F7FA"
                    cell_color = "#0F172A"; border_c = "#E2E8F0"
                    tbl_border = "rgba(0,188,212,0.2)"
                else:
                    hdr_bg = "rgba(15,23,42,0.9)"; hdr_color = "#00BCD4"
                    row_bg = "rgba(10,15,30,0.6)"; row_alt = "rgba(15,23,42,0.4)"; row_hover = "rgba(0,188,212,0.08)"
                    cell_color = "#E2E8F0"; border_c = "rgba(0,188,212,0.12)"
                    tbl_border = "rgba(0,188,212,0.15)"

                # Build rows
                rows_html = ""
                for i, (_, row) in enumerate(df.iterrows()):
                    bg = row_alt if i % 2 == 0 else row_bg
                    cells = "".join(
                        f'<td style="padding:12px 16px; color:{cell_color}; border-bottom:1px solid {border_c}; font-size:0.92rem; white-space:nowrap;">{val}</td>'
                        for val in row
                    )
                    rows_html += f'<tr style="background:{bg}; transition:background 0.2s;" onmouseover="this.style.background=\'{row_hover}\'" onmouseout="this.style.background=\'{bg}\'">{cells}</tr>'

                headers = "".join(
                    f'<th style="padding:12px 16px; background:{hdr_bg}; color:{hdr_color}; font-weight:700; font-size:0.88rem; text-transform:uppercase; letter-spacing:0.05em; border-bottom:2px solid {tbl_border}; white-space:nowrap;">{col}</th>'
                    for col in df.columns
                )

                table_html = f"""
                <div style="overflow-x:auto; border:1px solid {tbl_border}; border-radius:12px; margin-bottom:20px;">
                    <table style="width:100%; border-collapse:collapse; font-family:'Outfit','Inter',sans-serif;">
                        <thead><tr>{headers}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
                """
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Action buttons in columns
                c1, c2, _ = st.columns([1, 1, 3])
                with c1:
                    # Convert to CSV bytes for download
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download History (CSV)",
                        data=csv_bytes,
                        file_name="patient_history.csv",
                        mime="text/csv",
                        key=f"dl_btn_{datetime.datetime.now().timestamp()}"
                    )
                with c2:
                    if st.button("🗑️ Clear History", type="secondary"):
                        try:
                            # Recreate empty history
                            empty_df = pd.DataFrame(columns=["Patient Name", "Age", "Sex", "Disease Page", "Prediction Result", "Date & Time"])
                            empty_df.to_csv(HISTORY_FILE, index=False)
                            st.success("History cleared successfully!")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error clearing history: {str(ex)}")
        except Exception as e:
            st.error(f"Failed to read patient history: {str(e)}")
            if st.button("Reinitialize History File"):
                df = pd.DataFrame(columns=["Patient Name", "Age", "Sex", "Disease Page", "Prediction Result", "Date & Time"])
                df.to_csv(HISTORY_FILE, index=False)
                st.rerun()
    else:
        st.info("No patient history found. Make predictions to populate history.")
        # Create the file initially
        try:
            df = pd.DataFrame(columns=["Patient Name", "Age", "Sex", "Disease Page", "Prediction Result", "Date & Time"])
            df.to_csv(HISTORY_FILE, index=False)
        except Exception:
            pass
            
    st.markdown('</div>', unsafe_allow_html=True)

