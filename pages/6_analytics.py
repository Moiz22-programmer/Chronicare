import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from styles import apply_custom_styles

st.set_page_config(page_title="ChroniCare - Patient Analytics", page_icon="🧬", layout="wide")
apply_custom_styles()

st.markdown("""
    <h1 style="color:#60A5FA; font-weight: 800; margin-bottom: 2px; font-size:2.8rem; letter-spacing:-0.02em;">🧬 Unsupervised Patient Risk Analytics</h1>
    <p style="color:#94A3B8; font-size:1.15rem; margin-bottom: 30px; line-height:1.6;">Visualizing patient metabolic cohorts through K-Means clustering, PCA dimensionality reduction, and DBSCAN anomaly detection.</p>
""", unsafe_allow_html=True)

models_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
dbscan_path = os.path.join(models_dir, "dbscan_diabetes.pkl")

if not os.path.exists(dbscan_path):
    st.error("Error: Analytics data not found. Please ensure that models are trained.")
    st.stop()

# Load unsupervised results
with open(dbscan_path, "rb") as f:
    dbscan_data = pickle.load(f)

pca_coords = dbscan_data['pca_coords']
kmeans_labels = dbscan_data['kmeans_labels']
dbscan_labels = dbscan_data['dbscan_labels']
original_features = dbscan_data['original_features']

# Combine into a single pandas dataframe for Plotly
df_plot = original_features.copy()
df_plot['PC1'] = pca_coords[:, 0]
df_plot['PC2'] = pca_coords[:, 1]
df_plot['PC3'] = pca_coords[:, 2]
df_plot['KMeans_Cluster'] = kmeans_labels.astype(str)
df_plot['DBSCAN_Anomaly'] = np.where(dbscan_labels == -1, "Outlier", "Normal")

col_charts, col_stats = st.columns([3, 2])

with col_charts:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">🔮 3D PCA Cohort Space (K-Means)</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Each point represents a patient record mapped down from 8 dimensions. Color groups represent K-Means segmentation clusters.</p>', unsafe_allow_html=True)
    
    fig_pca = px.scatter_3d(
        df_plot,
        x='PC1',
        y='PC2',
        z='PC3',
        color='KMeans_Cluster',
        color_discrete_sequence=px.colors.qualitative.G10,
        labels={'KMeans_Cluster': 'Metabolic Group'},
        hover_data=['Glucose', 'BMI', 'Age', 'BloodPressure'],
        opacity=0.8
    )
    fig_pca.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(100,100,100,0.15)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(100,100,100,0.15)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(100,100,100,0.15)"),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">⚠️ DBSCAN Anomaly Detection Map</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">DBSCAN identifies patients showing highly atypical clinical profiles (outliers marked in red).</p>', unsafe_allow_html=True)
    
    fig_db = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='DBSCAN_Anomaly',
        color_discrete_map={"Normal": "#10B981", "Outlier": "#EF4444"},
        labels={'DBSCAN_Anomaly': 'Status'},
        hover_data=['Glucose', 'BMI', 'Age', 'BloodPressure']
    )
    fig_db.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_db, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📈 Clinical Group Characteristics</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Understand what each K-Means cluster represents based on the average clinical markers of its members.</p>', unsafe_allow_html=True)
    
    # Calculate group means
    df_grouped = df_plot.groupby('KMeans_Cluster')[['Age', 'Glucose', 'BMI', 'BloodPressure', 'Pregnancies']].mean()
    df_grouped.index.name = "Metabolic Group"
    
    # Beautify index display names
    df_grouped.index = df_grouped.index.map(lambda x: {
        "0": "Group 0 (Low Risk / Maintenance)",
        "1": "Group 1 (Moderate Risk / Elevated)",
        "2": "Group 2 (Severe Diabetic Markers)"
    }[x])
    
    st.dataframe(df_grouped.style.format("{:.1f}").background_gradient(cmap='Blues'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top:0; color:#60A5FA; font-weight:700;">💡 Clinical Segmentation Insights</h4>
            <div style="font-size:0.95rem; color:#E2E8F0; line-height: 1.8;">
                <p>
                    <b>Group 0:</b> Features patients with optimal glycemic averages (Glucose ~95) and normal BMI levels. 
                    Recommended for general metabolic maintenance checkups.
                </p>
                <p>
                    <b>Group 1:</b> Demonstrates prediabetic markers (Glucose ~125) and elevated BMI. 
                    Indicates target group for proactive lifestyle changes.
                </p>
                <p>
                    <b>Group 2:</b> Reflects diabetic clinical indicators (Glucose ≥155) combined with high BMI. 
                    Indicates high probability group where active clinical intervention is highly indicated.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
