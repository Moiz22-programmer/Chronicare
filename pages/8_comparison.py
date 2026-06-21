import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from styles import apply_custom_styles

st.set_page_config(page_title="ChroniCare - Algorithm Comparison", page_icon="📊", layout="wide")
apply_custom_styles()

st.markdown("""
    <h1 style="color:#60A5FA; font-weight: 800; margin-bottom: 2px; font-size:2.8rem; letter-spacing:-0.02em;">📊 Diagnostic Performance Comparisons</h1>
    <p style="color:#94A3B8; font-size:1.15rem; margin-bottom: 30px; line-height:1.6;">Compare the primary ML classifiers against their corresponding standard baseline configurations using Accuracy, Precision, Recall, F1, and ROC curves.</p>
""", unsafe_allow_html=True)

models_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
metrics_path = os.path.join(models_dir, "model_metrics.pkl")

if not os.path.exists(metrics_path):
    st.error("Error: Model training metrics file not found. Please ensure that models are trained.")
    st.stop()

with open(metrics_path, "rb") as f:
    metrics = pickle.load(f)

# Build a tidy comparative dataframe
comparison_data = []
for model_key, scores in metrics.items():
    disease, algo = model_key.split('_')
    comparison_data.append({
        'Disease Module': disease,
        'Algorithm': algo,
        'Accuracy': scores['accuracy'],
        'Precision': scores['precision'],
        'Recall': scores['recall'],
        'F1-Score': scores['f1'],
        'ROC AUC': scores['auc']
    })
df_metrics = pd.DataFrame(comparison_data)

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📈 Dynamic Metrics Table</h3>', unsafe_allow_html=True)
st.dataframe(df_metrics.style.format({
    'Accuracy': "{:.2%}",
    'Precision': "{:.2%}",
    'Recall': "{:.2%}",
    'F1-Score': "{:.3f}",
    'ROC AUC': "{:.3f}"
}).background_gradient(cmap='Blues'))
st.markdown('</div>', unsafe_allow_html=True)

# Select Metric to Compare
sel_metric = st.selectbox("Select Benchmark Variable to Compare", options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])

col_bar, col_roc = st.columns([3, 2])

with col_bar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📊 Algorithms Comparison ({sel_metric})</h3>', unsafe_allow_html=True)
    
    fig_bar = px.bar(
        df_metrics,
        x='Disease Module',
        y=sel_metric,
        color='Algorithm',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.G10,
        text_auto='.2%' if sel_metric in ['Accuracy', 'Precision', 'Recall'] else '.3f'
    )
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor="rgba(100,100,100,0.1)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_roc:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📈 ROC Performance Curves</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">ROC Curve outlines diagnostic trade-offs between true positive rates and false positive rates. Higher curves indicate superior classification.</p>', unsafe_allow_html=True)
    
    # Overlay ROC Curves in a single Plotly figure
    fig_roc = go.Figure()
    
    # Add a diagonal baseline reference line
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='grey'),
        name='Random Choice'
    ))
    
    colors_list = ['#3B82F6', '#EF4444', '#10B981', '#7C3AED', '#EC4899', '#F59E0B', '#14B8A6', '#6366F1']
    for idx, (model_key, scores) in enumerate(metrics.items()):
        fpr = scores['fpr']
        tpr = scores['tpr']
        auc_val = scores['auc']
        
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=colors_list[idx % len(colors_list)], width=2),
            name=f"{model_key} (AUC: {auc_val:.2f})"
        ))
        
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(gridcolor="rgba(100,100,100,0.1)"),
        yaxis=dict(gridcolor="rgba(100,100,100,0.1)"),
        legend=dict(font=dict(size=8), yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
