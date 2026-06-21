import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from styles import apply_custom_styles

st.set_page_config(page_title="ChroniCare - RL Advisor", page_icon="🤖", layout="wide")
apply_custom_styles()

st.markdown("""
    <h1 style="color:#60A5FA; font-weight: 800; margin-bottom: 2px; font-size:2.8rem; letter-spacing:-0.02em;">🤖 Reinforcement Learning Lifestyle Advisor</h1>
    <p style="color:#94A3B8; font-size:1.15rem; margin-bottom: 30px; line-height:1.6;">Interactive sequential treatment optimizer showing patient recovery pathways guided by a custom Q-Learning policy.</p>
""", unsafe_allow_html=True)

# Load Q-table
models_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
q_table_path = os.path.join(models_dir, "q_table.npy")

if not os.path.exists(q_table_path):
    st.error("Error: Trained Q-Table not found. Please train models.")
    st.stop()

q_table = np.load(q_table_path)

actions_map = {
    0: "Carb Reduction & Diet Plan",
    1: "Incorporate Daily Aerobic Exercise",
    2: "Establish Rigorous Vitals Monitoring",
    3: "Clinical Consultant Session"
}

# State helpers
def encode_state(g, bmi, bp):
    return g * 9 + bmi * 3 + bp

def decode_state(state):
    g = state // 9
    bmi = (state % 9) // 3
    bp = state % 3
    return g, bmi, bp

# State transition function
def transition_state(action):
    curr = st.session_state['sim_state']
    g, bmi, bp = decode_state(curr)
    
    next_g, next_bmi, next_bp = g, bmi, bp
    reward = 0
    
    # Matching train.py transition logic
    if action == 0:  # Carb Reduction
        if g > 0:
            next_g = max(0, g - 1)
            reward += 15
        else:
            reward += 5
    elif action == 1:  # Exercise
        if bmi > 0:
            next_bmi = max(0, bmi - 1)
            reward += 15
        if bp > 0:
            next_bp = max(0, bp - 1)
            reward += 10
        else:
            reward += 5
    elif action == 2:  # Monitor Vitals
        reward += 8
    elif action == 3:  # Clinical Consultation
        if g == 2 or bmi == 2 or bp == 2:
            next_g = max(0, g - 1)
            next_bp = max(0, bp - 1)
            reward += 20
        else:
            reward += 5
            
    # Biological state penalties
    reward -= (next_g * 6 + next_bmi * 4 + next_bp * 5)
    
    next_state = encode_state(next_g, next_bmi, next_bp)
    
    st.session_state['sim_state'] = next_state
    st.session_state['sim_reward'] += reward
    
    state_desc = f"Glucose:{['Normal','Elevated','High'][next_g]} | BMI:{['Normal','Overweight','Obese'][next_bmi]} | BP:{['Normal','Elevated','High'][next_bp]}"
    st.session_state['sim_history'].append({
        'action': actions_map[action],
        'reward': reward,
        'next_state': state_desc
    })

def run_auto_optimization():
    # Automatically execute optimal policy until recovered (0, 0, 0)
    steps = 0
    while steps < 8:
        curr = st.session_state['sim_state']
        g, bmi, bp = decode_state(curr)
        if g == 0 and bmi == 0 and bp == 0:
            break
        best_a = np.argmax(q_table[curr])
        transition_state(best_a)
        steps += 1

# Pre-populate using evaluated metrics if available
default_g = 0
default_bmi = 0
default_bp = 0

if 'glucose' in st.session_state:
    gl = st.session_state['glucose']
    default_g = 0 if gl < 100 else (1 if gl <= 125 else 2)
if 'bmi' in st.session_state:
    bm = st.session_state['bmi']
    default_bmi = 0 if bm < 25 else (1 if bm < 30 else 2)
if 'bp' in st.session_state:
    bpp = st.session_state['bp']
    default_bp = 0 if bpp < 80 else (1 if bpp < 90 else 2)

st.sidebar.markdown("### 🧬 Current Simulation State")
sel_g = st.sidebar.selectbox("Glucose State", options=[0, 1, 2], format_func=lambda x: ["Normal (<100)", "Elevated (100-125)", "High (>=126)"][x], index=default_g)
sel_bmi = st.sidebar.selectbox("BMI State", options=[0, 1, 2], format_func=lambda x: ["Normal (<25)", "Overweight (25-29.9)", "Obese (>=30)"][x], index=default_bmi)
sel_bp = st.sidebar.selectbox("Blood Pressure State", options=[0, 1, 2], format_func=lambda x: ["Normal (<80)", "Elevated (80-89)", "High (>=90)"][x], index=default_bp)

# Store simulated interactive state in session
if 'sim_state' not in st.session_state:
    st.session_state['sim_state'] = encode_state(sel_g, sel_bmi, sel_bp)
    st.session_state['sim_history'] = []
    st.session_state['sim_reward'] = 0

# If sidebar parameters changed, reset simulation state
new_state = encode_state(sel_g, sel_bmi, sel_bp)
if st.sidebar.button("Reset Simulation Vitals"):
    st.session_state['sim_state'] = new_state
    st.session_state['sim_history'] = []
    st.session_state['sim_reward'] = 0
    st.rerun()

curr_state = st.session_state['sim_state']
g, bmi, bp = decode_state(curr_state)

col_agent, col_console = st.columns([3, 2])

with col_agent:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">🏥 Current Patient Physiological State</h3>', unsafe_allow_html=True)
    
    # State cards with glowing ring pulses
    c1, c2, c3 = st.columns(3)
    with c1:
        color = "rgba(16, 185, 129, 0.12)" if g == 0 else ("rgba(245, 158, 11, 0.12)" if g == 1 else "rgba(239, 68, 68, 0.12)")
        txt_color = "#10B981" if g == 0 else ("#F59E0B" if g == 1 else "#EF4444")
        pulse_class = "pulse-healthy" if g == 0 else ("" if g == 1 else "pulse-danger")
        st.markdown(f"""
            <div class="{pulse_class}" style="background:{color}; color:{txt_color}; padding:18px; border-radius:16px; text-align:center; border:1px solid {txt_color}40; transition: all 0.3s ease;">
                <h5 style="margin:0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; color:{txt_color}; opacity:0.8;">Glucose Index</h5>
                <p style="font-size:1.4rem; font-weight:800; margin:8px 0 0 0; color:{txt_color};">{["Normal", "Elevated", "High Diabetic"][g]}</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        color = "rgba(16, 185, 129, 0.12)" if bmi == 0 else ("rgba(245, 158, 11, 0.12)" if bmi == 1 else "rgba(239, 68, 68, 0.12)")
        txt_color = "#10B981" if bmi == 0 else ("#F59E0B" if bmi == 1 else "#EF4444")
        pulse_class = "pulse-healthy" if bmi == 0 else ("" if bmi == 1 else "pulse-danger")
        st.markdown(f"""
            <div class="{pulse_class}" style="background:{color}; color:{txt_color}; padding:18px; border-radius:16px; text-align:center; border:1px solid {txt_color}40; transition: all 0.3s ease;">
                <h5 style="margin:0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; color:{txt_color}; opacity:0.8;">BMI Index</h5>
                <p style="font-size:1.4rem; font-weight:800; margin:8px 0 0 0; color:{txt_color};">{["Normal", "Overweight", "Obese"][bmi]}</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        color = "rgba(16, 185, 129, 0.12)" if bp == 0 else ("rgba(245, 158, 11, 0.12)" if bp == 1 else "rgba(239, 68, 68, 0.12)")
        txt_color = "#10B981" if bp == 0 else ("#F59E0B" if bp == 1 else "#EF4444")
        pulse_class = "pulse-healthy" if bp == 0 else ("" if bp == 1 else "pulse-danger")
        st.markdown(f"""
            <div class="{pulse_class}" style="background:{color}; color:{txt_color}; padding:18px; border-radius:16px; text-align:center; border:1px solid {txt_color}40; transition: all 0.3s ease;">
                <h5 style="margin:0; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; color:{txt_color}; opacity:0.8;">Blood Pressure</h5>
                <p style="font-size:1.4rem; font-weight:800; margin:8px 0 0 0; color:{txt_color};">{["Normal", "Elevated", "High Pressure"][bp]}</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">🧠 Learned Agent Policy (Q-Values)</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#94A3B8;">Q-values represent expected cumulative recovery rewards for each decision. High Q-value indicates optimal action.</p>', unsafe_allow_html=True)
    
    q_vals = q_table[curr_state]
    df_q = pd.DataFrame({
        'Action': [actions_map[0], actions_map[1], actions_map[2], actions_map[3]],
        'Value (Q-Score)': q_vals
    })
    
    fig_q = px.bar(df_q, y='Action', x='Value (Q-Score)', orientation='h', color='Value (Q-Score)', color_continuous_scale='Blues')
    fig_q.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(gridcolor="rgba(100,100,100,0.1)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_q, use_container_width=True)
    
    best_action = np.argmax(q_vals)
    st.success(f"💡 **AI Recommendation**: Apply **{actions_map[best_action]}** to maximize biological recovery trajectory.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color:#F8FAFC; font-weight:700; margin-bottom:15px;">🎮 Apply Next Simulation Step</h3>', unsafe_allow_html=True)
    col_act_btns = st.columns(2)
    with col_act_btns[0]:
        if st.button("Apply Recommended Action 🚀"):
            # Simulate step
            transition_state(best_action)
            st.rerun()
            
    with col_act_btns[1]:
        # User manual choice override
        selected_manual = st.selectbox("Manual Intervention Selection", options=[0, 1, 2, 3], format_func=lambda x: actions_map[x])
        if st.button("Execute Manual Choice ⚙️"):
            transition_state(selected_manual)
            st.rerun()

with col_console:
    st.markdown('<div class="glass-card" style="background:#0b0f19; border: 1px solid rgba(255, 255, 255, 0.08); color:white;">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0; color:#60A5FA; font-weight:700;">📟 Clinical Recovery Console</h3>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255, 255, 255, 0.08); padding-bottom:8px; margin-bottom:15px;">
            <span style="color:#94A3B8;">Cumulative Reward</span>
            <span style="font-weight:800; color:#34D399; font-size:1.2rem;">{st.session_state['sim_reward']}</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h5 style="color:#F8FAFC; margin-bottom:10px;">📝 Trajectory Logs:</h5>', unsafe_allow_html=True)
    if not st.session_state['sim_history']:
        st.markdown("<p style='color:#94A3B8; font-style:italic;'>No history logs. Set states and execute actions to log metrics.</p>", unsafe_allow_html=True)
    else:
        for idx, item in enumerate(st.session_state['sim_history']):
            st.markdown(
                f"""
                <div style='background:rgba(255, 255, 255, 0.03); border-radius:10px; padding:10px 14px; margin-bottom:10px; border-left: 4px solid #60A5FA; border: 1px solid rgba(255,255,255,0.06);'>
                    <span style='font-size:0.75rem; color:#94A3B8;'>Step {idx+1}</span><br/>
                    <span style='font-size:0.9rem; font-weight:700; color:#F8FAFC;'>{item['action']}</span><br/>
                    <span style='font-size:0.8rem; color:#34D399; font-weight:600;'>Reward: +{item['reward']}</span> | 
                    <span style='font-size:0.8rem; color:#CBD5E1;'>Next State: {item['next_state']}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
    if st.session_state['sim_history']:
        if st.button("Auto-Optimize to Healthy Recovery 🏆"):
            run_auto_optimization()
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
