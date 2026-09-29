import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import time
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RiskPulse DeepML — Credit Risk Studio 🛡️⚡",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hallmark Modern-Minimal Design System (High-Contrast Tokens & Entrance Motion)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: #090D16;
        color: #F8FAFC;
    }

    /* Input Field Label Styling - Bright, High Contrast */
    label[data-testid="stWidgetLabel"], 
    .stWidgetLabel p, 
    label, 
    div[data-baseweb="select"] span,
    .stNumberInput p,
    .stSelectbox p,
    .stFileUploader p {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em !important;
    }

    /* Form Input Fields Background & Border */
    div[data-baseweb="input"] input, 
    div[data-baseweb="select"] > div {
        background-color: #121826 !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input:focus, 
    div[data-baseweb="select"] > div:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
    }

    /* Keyframe Animations for Smooth Motion */
    @keyframes slideUpFade {
        0% {
            opacity: 0;
            transform: translateY(24px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.1); }
        50% { box-shadow: 0 0 20px rgba(56, 189, 248, 0.35); }
        100% { box-shadow: 0 0 5px rgba(56, 189, 248, 0.1); }
    }

    .animated-result {
        animation: slideUpFade 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* Header Banner */
    .hallmark-header {
        background: linear-gradient(135deg, #121826 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hallmark-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hallmark-subtitle {
        color: #CBD5E1;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .hallmark-card {
        background: #121826;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        animation: slideUpFade 0.5s ease-out forwards;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .hallmark-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .card-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    /* Decision Hero Cards with Motion */
    .decision-approved {
        background: rgba(16, 185, 129, 0.1);
        border: 1.5px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.25);
    }
    .decision-rejected {
        background: rgba(244, 63, 94, 0.1);
        border: 1.5px solid rgba(244, 63, 94, 0.4);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        box-shadow: 0 8px 25px -5px rgba(244, 63, 94, 0.25);
    }
    
    .status-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .badge-success { background: rgba(16, 185, 129, 0.25); color: #34D399; }
    .badge-warning { background: rgba(245, 158, 11, 0.25); color: #FBBF24; }
    .badge-danger { background: rgba(244, 63, 94, 0.25); color: #F87171; }
</style>
""", unsafe_allow_html=True)

# API Endpoint Config (Ensure no trailing slash)
API_URL = os.getenv("API_URL", "http://localhost:8000").strip().rstrip('/')

# Header
st.markdown("""
<div class="hallmark-header">
    <div class="hallmark-title">RiskPulse DeepML 🛡️⚡</div>
    <div class="hallmark-subtitle">Calibrated Machine Learning Credit Risk Assessment & Explainable AI (SHAP) Studio</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Engine Status")
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        if res.status_code == 200:
            health = res.json()
            st.markdown('<span class="status-badge badge-success">ONLINE</span>', unsafe_allow_html=True)
            st.markdown(f"**Loaded Threshold**: `{health['threshold']}`")
            st.markdown(f"**API Version**: `{health['version']}`")
        else:
            st.markdown('<span class="status-badge badge-warning">DEGRADED</span>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<span class="status-badge badge-danger">OFFLINE</span>', unsafe_allow_html=True)
        st.caption(f"Backend URL: `{API_URL}`")

    st.markdown("---")
    st.markdown("### 📋 Model Spec")
    st.markdown("- **Core Model**: XGBoost Classifier")
    st.markdown("- **Calibration**: Platt Scaling (Sigmoid)")
    st.markdown("- **Optimization**: 5-Fold Stratified CV")
    st.markdown("- **Explainability**: SHAP TreeExplainer")

tab1, tab2, tab3 = st.tabs([
    "🎯 Single Applicant Assessor",
    "📊 Portfolio Batch Analytics",
    "🎛️ Threshold & Calibration Simulator"
])

# TAB 1: SINGLE APPLICANT ASSESSOR
with tab1:
    st.subheader("Applicant Financial Profile & Risk Assessment")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### 👤 Demographic Info")
        person_age = st.number_input("Age (Years)", min_value=18, max_value=100, value=28)
        person_income = st.number_input("Annual Income ($)", min_value=0, max_value=5000000, value=65000, step=1000)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        person_emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)

    with c2:
        st.markdown("##### 💳 Loan Specifications")
        loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Credit Risk Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Requested Amount ($)", min_value=500, max_value=1000000, value=10000, step=500)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.14, step=0.1)

    with c3:
        st.markdown("##### 📜 Credit History & Ratios")
        loan_percent_income = round(loan_amnt / (person_income + 1e-5), 2)
        st.metric("Loan-to-Income Ratio", f"{loan_percent_income * 100:.1f}%")
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["N", "Y"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=4)

    st.markdown(" ")
    if st.button("⚡ Evaluate Credit Risk & Generate SHAP Drivers", type="primary", use_container_width=True):
        payload = {
            "person_age": int(person_age),
            "person_income": float(person_income),
            "person_home_ownership": person_home_ownership,
            "person_emp_length": float(person_emp_length),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": int(cb_person_cred_hist_length)
        }

        # Analyzing Status with Progress Animation
        with st.status("🔍 Analyzing applicant profile & calculating SHAP drivers...", expanded=True) as status_box:
            st.write("📡 Connecting to RiskPulse DeepML Microservice...")
            time.sleep(0.3)
            st.write("⚡ Computing Platt-calibrated default probability...")
            time.sleep(0.3)
            
            try:
                exp_res = requests.post(f"{API_URL}/explain", json=payload, timeout=10)
                st.write("🔍 Extracting local SHAP feature attributions...")
                time.sleep(0.2)
                
                if exp_res.status_code == 200:
                    status_box.update(label="✅ Credit Risk Assessment Complete!", state="complete", expanded=False)
                    result = exp_res.json()
                    prob = result["default_probability"]
                    is_def = result["is_default"]
                    category = result["risk_category"]
                    drivers = result["drivers"]

                    # Animated Result Section Container
                    st.markdown('<div class="animated-result">', unsafe_allow_html=True)
                    st.markdown("---")
                    st.subheader("Automated Credit Risk Assessment")

                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.markdown(f"""
                        <div class="hallmark-card">
                            <div class="card-label">Calibrated Default Probability</div>
                            <div class="card-value" style="color: {'#F43F5E' if prob > 0.45 else '#10B981'};">{prob * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m2:
                        st.markdown(f"""
                        <div class="hallmark-card">
                            <div class="card-label">Assigned Risk Category</div>
                            <div class="card-value">{category}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with m3:
                        if is_def == 1:
                            st.markdown("""
                            <div class="decision-rejected">
                                <h3 style="color: #F43F5E; margin:0;">❌ REJECT LOAN</h3>
                                <p style="color: #CBD5E1; margin:4px 0 0 0; font-size:0.85rem;">High Predicted Default Risk</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="decision-approved">
                                <h3 style="color: #10B981; margin:0;">✅ APPROVE LOAN</h3>
                                <p style="color: #CBD5E1; margin:4px 0 0 0; font-size:0.85rem;">Acceptable Default Risk Profile</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # SHAP Feature Drivers Chart with Motion
                    st.markdown("#### 🔍 SHAP Local Feature Attribution Drivers")
                    drivers_df = pd.DataFrame(drivers)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#F43F5E' if imp > 0 else '#10B981' for imp in drivers_df['impact']]
                    
                    ax.barh(drivers_df['feature'], drivers_df['impact'], color=colors)
                    ax.axvline(0, color='#64748B', linewidth=1, linestyle='--')
                    ax.set_xlabel('SHAP Impact on Default Risk')
                    ax.set_facecolor('#121826')
                    fig.patch.set_facecolor('#121826')
                    ax.tick_params(colors='#F8FAFC')
                    ax.xaxis.label.set_color('#F8FAFC')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#334155')
                    ax.spines['bottom'].set_color('#334155')

                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    status_box.update(label="❌ Assessment Failed", state="error", expanded=True)
                    st.error(f"Error from API ({exp_res.status_code}): {exp_res.text}")
            except Exception as e:
                status_box.update(label="❌ Connection Failed", state="error", expanded=True)
                st.error(f"Connection failed: {str(e)}")


# TAB 2: PORTFOLIO BATCH ANALYTICS
with tab2:
    st.subheader("Batch Loan Portfolio Risk Analytics")
    uploaded_file = st.file_uploader("Upload Applicants CSV File", type=["csv"])

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.markdown(f"**Loaded Batch**: `{len(df_batch)} records`")
        st.dataframe(df_batch.head(3), use_container_width=True)

        if st.button("🚀 Score Portfolio Batch", type="primary"):
            with st.status("📊 Scoring batch loan portfolio...", expanded=True) as batch_status:
                try:
                    records = df_batch.to_dict(orient="records")
                    res = requests.post(f"{API_URL}/predict/batch", json={"applications": records})
                    if res.status_code == 200:
                        batch_status.update(label="✅ Batch Scoring Complete!", state="complete", expanded=False)
                        batch_res = res.json()
                        preds_df = pd.DataFrame(batch_res["predictions"])
                        final_df = pd.concat([df_batch, preds_df], axis=1)

                        total = len(final_df)
                        approved = (final_df['is_default'] == 0).sum()
                        rejected = (final_df['is_default'] == 1).sum()
                        avg_prob = final_df['default_probability'].mean()

                        st.markdown('<div class="animated-result">', unsafe_allow_html=True)
                        st.markdown("---")
                        b1, b2, b3, b4 = st.columns(4)

                        with b1:
                            st.metric("Total Scored", total)
                        with b2:
                            st.metric("Approved Loans", approved, delta=f"{approved/total*100:.1f}% approval")
                        with b3:
                            st.metric("Rejected Loans", rejected, delta=f"-{rejected/total*100:.1f}% rejection")
                        with b4:
                            st.metric("Portfolio Avg Risk", f"{avg_prob*100:.2f}%")

                        st.dataframe(final_df, use_container_width=True)

                        csv_export = final_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Scored Credit Risk CSV",
                            data=csv_export,
                            file_name="riskpulse_scored_portfolio.csv",
                            mime="text/csv"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        batch_status.update(label="❌ Batch Failed", state="error", expanded=True)
                        st.error(f"Batch API error ({res.status_code}): {res.text}")
                except Exception as e:
                    batch_status.update(label="❌ Execution Error", state="error", expanded=True)
                    st.error(f"Batch scoring failed: {str(e)}")


# TAB 3: THRESHOLD & CALIBRATION SIMULATOR
with tab3:
    st.subheader("Classification Threshold & Risk Trade-off Simulator")
    st.markdown("""
    Adjust the decision threshold to simulate business impact on approval rates, precision, and default rate.
    Lowering the threshold prioritizes **Risk Avoidance** (higher recall on defaults), while raising it prioritizes **Loan Volume Growth**.
    """)

    sim_threshold = st.slider("Simulated Decision Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    sim_c1, sim_c2 = st.columns(2)

    with sim_c1:
        st.markdown(f"### Threshold Setting: `{sim_threshold:.2f}`")
        if sim_threshold < 0.35:
            st.info("🛡️ **Conservative Stance**: Higher rejection rate, minimizes default exposure.")
        elif sim_threshold <= 0.55:
            st.success("⚖️ **Balanced Stance**: Optimal F1-score balance between volume and risk.")
        else:
            st.warning("⚡ **Aggressive Stance**: Maximizes loan volume, higher risk of default leakage.")

    with sim_c2:
        # Generate simulated curve preview
        thresh_vals = np.linspace(0.1, 0.9, 20)
        est_approval = 1.0 / (1.0 + np.exp(4 * (thresh_vals - 0.45)))
        est_default_rate = 0.05 + 0.30 * (1 - thresh_vals)**2

        fig_sim, ax_sim = plt.subplots(figsize=(8, 3.5))
        ax_sim.plot(thresh_vals, est_approval * 100, label='Approval Rate %', color='#38BDF8', linewidth=2)
        ax_sim.plot(thresh_vals, est_default_rate * 100, label='Portfolio Default Rate %', color='#F43F5E', linewidth=2)
        ax_sim.axvline(sim_threshold, color='#FBBF24', linestyle='--', label=f'Current: {sim_threshold:.2f}')
        
        ax_sim.set_xlabel('Decision Threshold')
        ax_sim.set_ylabel('Percentage (%)')
        ax_sim.legend(facecolor='#121826', edgecolor='none', labelcolor='#F8FAFC')
        ax_sim.set_facecolor('#121826')
        fig_sim.patch.set_facecolor('#121826')
        ax_sim.tick_params(colors='#F8FAFC')
        ax_sim.xaxis.label.set_color('#F8FAFC')
        ax_sim.yaxis.label.set_color('#F8FAFC')
        ax_sim.spines['top'].set_visible(False)
        ax_sim.spines['right'].set_visible(False)
        ax_sim.spines['left'].set_color('#334155')
        ax_sim.spines['bottom'].set_color('#334155')

        st.pyplot(fig_sim)
