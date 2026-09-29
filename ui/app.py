import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(
    page_title="RiskPulse DeepML 🛡️⚡",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3B82F6;
    }
    .risk-low {
        color: #16A34A;
        font-weight: bold;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint Config
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.markdown('<p class="main-title">RiskPulse DeepML 🛡️⚡</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Calibrated Machine Learning Credit Risk Assessment & Loan Default Forecasting Platform</p>', unsafe_allow_html=True)

# Sidebar System Health
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            health = response.json()
            st.success(f"Backend API: {health['status'].upper()}")
            st.info(f"Loaded Threshold: {health['threshold']}")
        else:
            st.warning("Backend API reachable but returned error.")
    except Exception:
        st.error(f"Cannot connect to API at `{API_URL}`. Ensure FastAPI backend is running.")

    st.markdown("---")
    st.markdown("### 📌 Model Information")
    st.markdown("- **Algorithm**: XGBoost Classifier")
    st.markdown("- **Calibration**: Platt Scaling (Sigmoid)")
    st.markdown("- **Features**: 11 Demographic & Credit Variables")

tab1, tab2 = st.tabs(["📝 Single Loan Assessment", "📊 Batch Loan Scoring"])

with tab1:
    st.subheader("Applicant Financial & Demographic Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("Applicant Age", min_value=18, max_value=100, value=28)
        person_income = st.number_input("Annual Income ($)", min_value=0, max_value=5000000, value=65000, step=1000)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        person_emp_length = st.number_input("Employment Duration (Years)", min_value=0.0, max_value=50.0, value=4.0, step=0.5)

    with col2:
        loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Assigned Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=1000000, value=10000, step=500)
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.14, step=0.1)

    with col3:
        loan_percent_income = round(loan_amnt / (person_income + 1e-5), 2)
        st.metric("Loan-to-Income Ratio", f"{loan_percent_income * 100:.1f}%")
        cb_person_default_on_file = st.selectbox("Historical Default on File", ["N", "Y"])
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, value=4)

    if st.button("🔍 Assess Credit Risk", type="primary", use_container_width=True):
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

        try:
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                result = res.json()
                st.markdown("---")
                st.subheader("Risk Assessment Summary")

                r_col1, r_col2, r_col3 = st.columns(3)
                
                with r_col1:
                    prob = result["default_probability"]
                    st.metric("Calibrated Default Probability", f"{prob * 100:.2f}%")

                with r_col2:
                    category = result["risk_category"]
                    st.metric("Assigned Risk Level", category)

                with r_col3:
                    decision = "REJECT LOAN (High Default Risk)" if result["is_default"] == 1 else "APPROVE LOAN (Acceptable Risk)"
                    st.metric("Credit Decision", decision)

                st.progress(prob)
            else:
                st.error(f"Error from backend API: {res.text}")
        except Exception as e:
            st.error(f"Failed to communicate with API: {str(e)}")

with tab2:
    st.subheader("Batch Loan Portfolio Risk Evaluation")
    uploaded_file = st.file_uploader("Upload CSV Applicant File", type=["csv"])

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Uploaded Sample Preview:", df_upload.head(3))

        if st.button("🚀 Score Entire Batch"):
            try:
                records = df_upload.to_dict(orient="records")
                res = requests.post(f"{API_URL}/predict/batch", json={"applications": records})
                if res.status_code == 200:
                    batch_res = res.json()
                    preds_df = pd.DataFrame(batch_res["predictions"])
                    final_df = pd.concat([df_upload, preds_df], axis=1)

                    st.success(f"Processed {batch_res['total_processed']} applications successfully!")
                    st.dataframe(final_df)

                    csv_data = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Scored CSV", csv_data, "scored_credit_risk.csv", "text/csv")
                else:
                    st.error(f"Batch prediction error: {res.text}")
            except Exception as e:
                st.error(f"Batch execution failed: {str(e)}")
