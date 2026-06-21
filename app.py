import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="MatRisk AI Lab", layout="wide", page_icon="🏗️")

# Custom CSS for LARGER typography, bolding, and dashboard background
st.markdown("""
    <style>
    /* Increase base font size for the whole app */
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }
    
    /* Massive, bold headers */
    .main-header {font-size: 3.5rem !important; color: #0F172A; font-weight: 900 !important; margin-bottom: 0px; font-family: 'Inter', sans-serif;}
    .sub-header {font-size: 1.5rem !important; color: #475569; font-weight: 600 !important; margin-top: 5px; margin-bottom: 2rem;}
    
    /* Bold and enlarge metric components */
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        font-weight: 800 !important;
        color: #1E293B !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 900 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {border-radius: 8px; overflow: hidden; border: 2px solid #E2E8F0;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏗️ MatRisk Lab</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header"><strong>AI-Driven Infrastructure & Commodity Simulation Engine</strong></p>', unsafe_allow_html=True)

# ==========================================
# 2. DATA INGESTION & CACHING
# ==========================================
@st.cache_data
def load_datasets():
    """Loads datasets and includes synthetic fallbacks if files are missing."""
    data = {}
    try:
        data['ds3'] = pd.read_csv("DS3_infrastructure_bridges_5000.csv")
        data['ds6'] = pd.read_csv("DS6_historical_failures_2000.csv")
    except FileNotFoundError:
        data['ds3'] = pd.DataFrame({
            'bridge_id': [f'BR-{i:04d}' for i in range(1, 101)],
            'bridge_type': np.random.choice(['Highway', 'Pedestrian', 'Interstate'], 100),
            'material': np.random.choice(['Steel (A36)', 'Reinforced Concrete', 'FRP Composite'], 100),
            'age_years': np.random.randint(10, 80, 100),
            'design_life_years': 100,
            'corrosion_rate_mm_yr': np.random.uniform(0.01, 0.1, 100),
            'replacement_cost_M': np.random.uniform(1, 50, 100)
        })
        data['ds6'] = pd.DataFrame({
            'event_id': [f'FE-{i:04d}' for i in range(1, 501)],
            'material': np.random.choice(['Steel (A36)', 'Reinforced Concrete', 'FRP Composite'], 500),
            'age_at_event_years': np.random.randint(20, 90, 500),
            'corrosion_rate_mm_yr': np.random.uniform(0.02, 0.15, 500),
            'loss_ratio': np.random.uniform(0.1, 1.0, 500)
        })
    return data

datasets = load_datasets()
df_portfolio = datasets['ds3']
df_failures = datasets['ds6']

# ==========================================
# 3. AI MODEL TRAINING 
# ==========================================
@st.cache_resource
def train_rul_model(df_fail):
    df = df_fail.copy()
    le = LabelEncoder()
    df['material_encoded'] = le.fit_transform(df['material'])
    
    X = df[['material_encoded', 'corrosion_rate_mm_yr']]
    y_age = df['age_at_event_years']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y_age)
    return model, le

rf_model, material_encoder = train_rul_model(df_failures)

# ==========================================
# 4. SIMULATION ENGINE
# ==========================================
def calculate_portfolio_metrics(df, budget_m, corrosion_mult, quality_drop):
    sim_df = df.copy()
    sim_df['sim_corrosion_rate'] = sim_df['corrosion_rate_mm_yr'] * corrosion_mult
    
    known_materials = material_encoder.classes_
    sim_df['safe_material'] = sim_df['material'].apply(lambda x: x if x in known_materials else known_materials[0])
    sim_df['mat_encoded'] = material_encoder.transform(sim_df['safe_material'])
    
    X_pred = sim_df[['mat_encoded', 'sim_corrosion_rate']].rename(columns={
        'mat_encoded': 'material_encoded',
        'sim_corrosion_rate': 'corrosion_rate_mm_yr'
    })
    
    predicted_failure_age = rf_model.predict(X_pred)
    predicted_failure_age = predicted_failure_age * (1 - (quality_drop / 100))
    
    budget_impact = (budget_m / 10) * 5 
    sim_df['predicted_rul'] = np.maximum((predicted_failure_age + budget_impact) - sim_df['age_years'], 0)
    sim_df['expected_loss'] = sim_df['replacement_cost_M'] * (sim_df['sim_corrosion_rate'] * 10)
    
    total_coq = sim_df['expected_loss'].sum()
    total_value = sim_df['replacement_cost_M'].sum()
    npv = (total_value - total_coq - budget_m) / ((1 + 0.05) ** 5) 
    avg_rul = sim_df['predicted_rul'].mean()
    
    return sim_df, npv, avg_rul, total_coq

# ==========================================
# 5. STREAMLIT UI: SIDEBAR & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## **🕹️ Simulation Controls**")
    st.markdown("**Adjust macro-parameters below to stress-test the portfolio.**")
    
    st.divider()
    
    scenario = st.selectbox("**🌍 Inject Stress Scenario**", [
        "Baseline (Normal Operations)", 
        "Climate-Accelerated Corrosion (+30% Rate)", 
        "Steel Quality Degradation (-15% Quality)",
        "Rare Earth Supply Crisis"
    ])

    default_corr, default_qual = 1.0, 0.0
    if "Climate" in scenario: default_corr = 1.3
    elif "Quality" in scenario: default_qual = 15.0

    st.divider()
    budget = st.slider("**💰 Annual Maintenance Budget ($M)**", 1.0, 50.0, 10.0, 1.0)
    corrosion_multiplier = st.slider("**🌧️ Climate Corrosion Multiplier**", 1.0, 3.0, default_corr, 0.1)
    quality_drop = st.slider("**🏗️ Material Quality Drop (%)**", 0.0, 30.0, float(default_qual), 1.0)

results_df, port_npv, port_rul, port_coq = calculate_portfolio_metrics(df_portfolio, budget, corrosion_multiplier, quality_drop)

# ==========================================
# 6. STREAMLIT UI: DASHBOARD VISUALIZATIONS
# ==========================================

# Top Row: KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    with st.container(border=True):
        st.metric("📈 Projected NPV", f"${port_npv:,.2f} M", f"{-(port_coq/port_npv)*100:.1f}% Impact" if port_npv > 0 else "Critical", delta_color="inverse")
with col2:
    with st.container(border=True):
        st.metric("⏳ Average RUL", f"{port_rul:.1f} Yrs", f"{(port_rul - 35.0):.1f} vs Target" if port_rul else None)
with col3:
    with st.container(border=True):
        st.metric("💸 Cost of Quality", f"${port_coq:,.2f} M")
with col4:
    with st.container(border=True):
        st.metric("🌉 Assets at Risk", f"{len(results_df[results_df['predicted_rul'] < 10])} Bridges", "Urgent")

st.markdown("---")

# Middle Row: Charts (Proper 3:2 Column Ratio for better layout)
col_chart1, col_chart2 = st.columns([3, 2])

with col_chart1:
    st.markdown("### **📉 Accelerated Degradation Curve**")
    years = np.arange(0, 21)
    degradation = [max(port_rul - (y * corrosion_multiplier) + (budget/20), 0) for y in years]
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=years, y=degradation, mode='lines', fill='tozeroy', 
                                  name='Predicted RUL', line=dict(color='#3B82F6', width=5),
                                  fillcolor='rgba(59, 130, 246, 0.2)'))
    fig_line.add_hline(y=10, line_dash="dash", line_color="#EF4444", line_width=3,
                       annotation_text="CRITICAL ACTION ZONE (RUL < 10 Yrs)", 
                       annotation_position="bottom left", annotation_font=dict(size=14, color="#EF4444", family="Arial Black"))
    
    fig_line.update_layout(xaxis_title="<b>Years from Present</b>", yaxis_title="<b>Remaining Useful Life (Years)</b>", 
                           template="plotly_white", margin=dict(l=0, r=0, t=10, b=0), height=400,
                           font=dict(size=14))
    st.plotly_chart(fig_line, use_container_width=True)

with col_chart2:
    st.markdown("### **🧩 Risk by Topology & Material**")
    treemap_data = results_df.groupby(['bridge_type', 'material']).agg({'expected_loss': 'sum'}).reset_index()
    
    fig_tree = px.treemap(treemap_data, path=[px.Constant("Portfolio"), 'bridge_type', 'material'], 
                          values='expected_loss', color='expected_loss', color_continuous_scale='Reds',
                          hover_data=['expected_loss'])
    fig_tree.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=400, font=dict(size=16, family="Arial", color="black"))
    st.plotly_chart(fig_tree, use_container_width=True)

# Bottom Row: Dataframe
st.markdown("### **📋 Asset-Level Risk Analytics**")

display_cols = ['bridge_id', 'bridge_type', 'material', 'age_years', 'sim_corrosion_rate', 'predicted_rul', 'expected_loss']

# Applying Pandas Background Gradients to make the table visually intuitive
styled_df = (results_df[display_cols]
             .sort_values('expected_loss', ascending=False)
             .head(20)
             .style
             .background_gradient(subset=['expected_loss'], cmap='Reds', vmin=0)
             .background_gradient(subset=['predicted_rul'], cmap='RdYlGn', vmin=0, vmax=50)
             .format({"sim_corrosion_rate": "{:.3f}", "predicted_rul": "{:.1f} yrs", "expected_loss": "${:.2f}M"}))

st.dataframe(styled_df, use_container_width=True, height=500)