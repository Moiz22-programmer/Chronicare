import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from styles import apply_custom_styles

st.set_page_config(page_title="ChroniCare - Data Explorer", page_icon="🔍", layout="wide")
apply_custom_styles()

st.markdown("""
    <h1 style="color:#60A5FA; font-weight: 800; margin-bottom: 2px; font-size:2.8rem; letter-spacing:-0.02em;">🔍 Interactive Clinical Data Explorer</h1>
    <p style="color:#94A3B8; font-size:1.15rem; margin-bottom: 30px; line-height:1.6;">Examine the statistical distributions, demographic patterns, and chemical correlations of our four chronic disease datasets.</p>
""", unsafe_allow_html=True)

# Select dataset to load
sel_dataset = st.selectbox(
    "Select Clinical Dataset to Analyze",
    options=["Diabetes (PIMA)", "Heart Disease (Cleveland)", "Chronic Kidney Disease (CKD)", "Liver Patient (ILPD)"]
)

data_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\data"

# Map to file path and target column
dataset_mapping = {
    "Diabetes (PIMA)": ("diabetes.csv", "Outcome"),
    "Heart Disease (Cleveland)": ("heart.csv", "target"),
    "Chronic Kidney Disease (CKD)": ("kidney.csv", "class"),
    "Liver Patient (ILPD)": ("liver.csv", "Dataset")
}

file_name, target_col = dataset_mapping[sel_dataset]
csv_path = os.path.join(data_dir, file_name)

if not os.path.exists(csv_path):
    st.error("Error: Clinical CSV data not found. Please verify data generator has run.")
    st.stop()

# Load data
df = pd.read_csv(csv_path)

col_desc, col_corr = st.columns([2, 3])

with col_desc:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📈 Statistical Summary</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:0.85rem; color:#94A3B8;">Overview of descriptive statistics for <b>{sel_dataset}</b> ({len(df)} patient records).</p>', unsafe_allow_html=True)
    
    st.dataframe(df.describe().T.style.format("{:.2f}").background_gradient(cmap='Blues'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📊 Feature Distribution Histograms</h3>', unsafe_allow_html=True)
    
    # Choose feature to plot
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove target col
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    sel_feature = st.selectbox("Select Clinical Parameter to Graph", options=numeric_cols)
    
    fig_hist = px.histogram(
        df,
        x=sel_feature,
        color=df[target_col].astype(str),
        barmode='overlay',
        color_discrete_sequence=["#10B981", "#EF4444"],
        labels={target_col: 'Diagnostic Status', 'color': 'Outcome'},
        opacity=0.75
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor="rgba(100,100,100,0.1)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_corr:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">🔥 Biological Correlation Heatmap</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Correlations indicate pairwise relationships. Values near +1.0 or -1.0 suggest powerful clinical associations.</p>', unsafe_allow_html=True)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    fig_heat = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0
    )
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
